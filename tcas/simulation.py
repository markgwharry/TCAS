"""Core simulation routines for ACAS/TCAS residual risk analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .constants import (
    DEFAULT_ALIM_FT,
    DEFAULT_RESP_THRESH_FPM,
    G,
    MS_PER_FPM,
    PL_ACCEL_G,
    PL_DELAY_S,
    PL_IAS_KT,
    PL_VS_CAP,
    PL_VS_FPM,
)
from .geometry import ias_to_tas, relative_closure_kt, sample_headings, time_to_go_from_geometry
from .kinematics import delta_h_piecewise, integrate_altitude_from_vs, vs_time_series
from .sampling import (
    apply_surveillance_noise,
    baseline_dh_ft,
    sample_altitudes_and_h0,
    sample_pilot_response_cat,
    sample_tgo_with_trigger,
)
from .surrogate import late_override_monitor, surrogate_decision_stream

SCENARIO_REL_BOUNDS = {
    "Head-on": (150.0, 210.0),
    "Crossing": (60.0, 120.0),
    "Overtaking": (0.0, 30.0),
}


@dataclass
class NonComplianceConfig:
    p_opposite: float = 0.02
    p_leveloff: float = 0.03
    p_persist: float = 0.01
    jitter: bool = True
    ta_only: bool = False


@dataclass
class BatchSimulationInputs:
    n_runs: int
    seed: int
    scenario: str
    r_min: float
    r_max: float
    hdg1_min: float = 0.0
    hdg1_max: float = 360.0
    hdg2_min: Optional[float] = None
    hdg2_max: Optional[float] = None
    rel_min: Optional[float] = None
    rel_max: Optional[float] = None
    ra_trigger_mode: str = "Scenario-calibrated (recommended)"
    tgo_cap: float = 60.0
    use_distrib: bool = True
    alim_ft: float = DEFAULT_ALIM_FT
    baseline: str = "IDEAL 1500 fpm (ACASA 2002)"
    dt: float = 0.1
    resp_thr: float = DEFAULT_RESP_THRESH_FPM
    cat_vs: float = 1500.0
    cat_cap: float = 2000.0
    cat_ag_nom: float = 0.25
    cat_td_nom: float = 5.0
    cat_tas_min: float = 420.0
    cat_tas_max: float = 470.0
    h0_mean: float = 250.0
    h0_sd: float = 100.0
    h0_lo: float = 100.0
    h0_hi: float = 500.0
    p_miss: float = 0.0
    noncomp: NonComplianceConfig = field(default_factory=NonComplianceConfig)


@dataclass
class RunResult:
    run: int
    scenario: str
    FL_PL: int
    FL_CAT: int
    PL_TAS: float
    CAT_TAS: float
    PLhdg: float
    CAThdg: float
    R0NM: float
    closurekt: float
    tgos: float
    plDelay: float
    plAccel_g: float
    catDelay: float
    catAccel_g: float
    intruder_mode: str
    h0ft: float
    missCPAft: float
    minPredMiss_bothResp_ft: float
    ALIMbreach_CPA: bool
    ALIMbreach_ANY_predCPA: bool
    dhPLft: float
    dhCATft: float
    dhbaselineft: float
    ratiobaseoverPL: float
    unresolvedRRpct: float
    eventtype_first: str
    eventtime_first: float
    eventcause_first: str
    eventtype: str
    eventtimes: float
    eventcause: str


def run_single_case(
    rng: np.random.Generator,
    inputs: BatchSimulationInputs,
) -> RunResult:
    FL_pl, FL_cat, h0 = sample_altitudes_and_h0(
        rng, inputs.h0_mean, inputs.h0_sd, inputs.h0_lo, inputs.h0_hi
    )
    PL_TAS = ias_to_tas(PL_IAS_KT, FL_pl * 100.0)
    cat_tas_lo = min(inputs.cat_tas_min, inputs.cat_tas_max)
    cat_tas_hi = max(inputs.cat_tas_min, inputs.cat_tas_max)
    CAT_TAS = float(rng.uniform(cat_tas_lo, cat_tas_hi))

    if inputs.scenario == "Custom":
        assert inputs.hdg2_min is not None and inputs.hdg2_max is not None
        h1 = float(rng.uniform(inputs.hdg1_min, inputs.hdg1_max))
        h2 = float(rng.uniform(inputs.hdg2_min, inputs.hdg2_max))
    else:
        rel_min = inputs.rel_min
        rel_max = inputs.rel_max
        if rel_min is None or rel_max is None:
            rel_min, rel_max = SCENARIO_REL_BOUNDS.get(inputs.scenario, (0.0, 360.0))
        h1, h2 = sample_headings(
            rng,
            inputs.scenario,
            inputs.hdg1_min,
            inputs.hdg1_max,
            rel_min,
            rel_max,
        )
    r0 = float(rng.uniform(min(inputs.r_min, inputs.r_max), max(inputs.r_min, inputs.r_max)))
    vcl = relative_closure_kt(PL_TAS, h1, CAT_TAS, h2)
    tgo_geom = time_to_go_from_geometry(r0, vcl)
    if inputs.ra_trigger_mode.startswith("Scenario"):
        tgo = sample_tgo_with_trigger(rng, inputs.scenario, tgo_geom, FL_pl, FL_cat, cap_s=inputs.tgo_cap)
    else:
        if tgo_geom is None:
            tgo_geom = 30.0
        tgo = float(np.clip(tgo_geom, 6.0, inputs.tgo_cap))

    pl_td_k = PL_DELAY_S
    pl_ag_k = PL_ACCEL_G
    if inputs.use_distrib:
        cat_td_k, cat_ag_k = sample_pilot_response_cat(rng)
    else:
        cat_td_k, cat_ag_k = inputs.cat_td_nom, inputs.cat_ag_nom

    dh_pl = delta_h_piecewise(tgo, pl_td_k, pl_ag_k, PL_VS_FPM)
    dh_cat = delta_h_piecewise(tgo, cat_td_k, cat_ag_k, inputs.cat_vs)
    dh_base = baseline_dh_ft(tgo, mode=inputs.baseline)
    ratio = (dh_base / dh_pl) if dh_pl > 1e-6 else np.nan
    unresolved_rr = 1.1 * ratio

    times, vs_pl = vs_time_series(tgo, inputs.dt, pl_td_k, pl_ag_k, PL_VS_FPM, sense=+1, cap_fpm=PL_VS_CAP)
    _, vs_ca = vs_time_series(tgo, inputs.dt, cat_td_k, cat_ag_k, inputs.cat_vs, sense=-1, cap_fpm=inputs.cat_cap)

    noncomp = inputs.noncomp
    mode = "BASE"
    if noncomp.ta_only:
        mode = "TA_ONLY"
        times = np.arange(0.0, tgo + 1e-9, inputs.dt)
        vs_ca = np.zeros_like(times) + rng.normal(0.0, 100.0, size=times.size)
    else:
        p1 = noncomp.p_opposite
        p2 = noncomp.p_opposite + noncomp.p_leveloff
        p3 = p2 + noncomp.p_persist
        if noncomp.jitter:
            j1 = float(np.clip(rng.uniform(0.5, 1.5), 0.0, 2.0))
            j2 = float(np.clip(rng.uniform(0.5, 1.5), 0.0, 2.0))
            j3 = float(np.clip(rng.uniform(0.5, 1.5), 0.0, 2.0))
            p1 *= j1
            p2 = p1 + noncomp.p_leveloff * j2
            p3 = p2 + noncomp.p_persist * j3
        u = rng.uniform()
        if u < p1:
            mode = "OPPOSITE"
            wrong_win = (times >= cat_td_k) & (times <= cat_td_k + 6.0)
            vs_ca[wrong_win] = -vs_ca[wrong_win]
        elif u < p2:
            mode = "LEVELOFF"
            hold_win = (times >= cat_td_k) & (times <= cat_td_k + 8.0)
            vs_ca[hold_win] = 0.0
        elif u < p3:
            mode = "PERSIST"
            vs_ca[:] = np.clip(vs_ca, -250.0, +250.0)

    vs_pl_noisy, vs_ca_noisy = apply_surveillance_noise(rng, times, vs_pl, vs_ca, p_miss=inputs.p_miss)
    z_pl_dec = integrate_altitude_from_vs(times, vs_pl_noisy)
    z_ca_dec = integrate_altitude_from_vs(times, vs_ca_noisy)
    sep_now_dec = h0 + (z_pl_dec - z_ca_dec)
    evs, t_strengthen = surrogate_decision_stream(
        times,
        vs_pl_noisy,
        vs_ca_noisy,
        sep_now_dec,
        t_cpa_s=tgo,
        alim_ft=inputs.alim_ft,
        resp_thr=inputs.resp_thr,
    )
    eventtype_first = evs[0][0] if evs else "NONE"
    eventtime_first = evs[0][1] if evs else np.nan
    eventcause_first = evs[0][2] if evs else "N/A"

    if any(e[0] == "STRENGTHEN" for e in evs):
        t_ev = next(e[1] for e in evs if e[0] == "STRENGTHEN")
        post_mask = times > t_ev
        te = np.clip(times - t_ev, 0, None)
        a = cat_ag_k * G
        new_target_fpm = min(inputs.cat_cap, inputs.cat_vs + 500.0)
        v_target = new_target_fpm * MS_PER_FPM
        v_after = np.minimum(a * te, v_target) / MS_PER_FPM
        v_after *= -1
        vs_ca[post_mask] = v_after[post_mask]

    vs_pl_noisy2, vs_ca_noisy2 = apply_surveillance_noise(rng, times, vs_pl, vs_ca, p_miss=inputs.p_miss)
    z_pl_dec2 = integrate_altitude_from_vs(times, vs_pl_noisy2)
    z_ca_dec2 = integrate_altitude_from_vs(times, vs_ca_noisy2)
    sep_now_dec2 = h0 + (z_pl_dec2 - z_ca_dec2)
    evs = late_override_monitor(
        evs,
        t_strengthen,
        times,
        vs_pl_noisy2,
        vs_ca_noisy2,
        sep_now_dec2,
        t_cpa_s=tgo,
        alim_ft=inputs.alim_ft,
        resp_thr=inputs.resp_thr,
        intruder_delay_s=cat_td_k,
        noncomp_grace_s=2.0,
        ta_only=(mode == "TA_ONLY"),
    )
    eventtype_final = evs[-1][0] if evs else "NONE"
    eventtime_final = evs[-1][1] if evs else np.nan
    eventcause_final = evs[-1][2] if evs else "N/A"

    z_pl = integrate_altitude_from_vs(times, vs_pl)
    z_ca = integrate_altitude_from_vs(times, vs_ca)
    sep_now = h0 + (z_pl - z_ca)
    tgo_series = np.clip(tgo - times, a_min=0.0, a_max=None)
    pred_miss_series = np.abs(sep_now + ((vs_pl - vs_ca) * (tgo_series / 60.0)))

    def first_compliance_time(times, vs_fpm, sense, thr=DEFAULT_RESP_THRESH_FPM):
        ok = (vs_fpm * sense) >= thr
        if ok.any():
            idx = np.argmax(ok)
            return times[idx]
        return np.inf

    t_own_ok = first_compliance_time(times, vs_pl, +1, thr=inputs.resp_thr)
    t_int_ok = first_compliance_time(times, vs_ca, -1, thr=inputs.resp_thr)
    start_t = max(t_own_ok, t_int_ok) + 0.5
    mask_any = times >= start_t
    min_pred_miss = float(np.min(pred_miss_series[mask_any])) if mask_any.any() else float(np.min(pred_miss_series))
    miss_cpa = float(abs(h0 + (z_pl[-1] - z_ca[-1])))
    breach_cpa = bool(miss_cpa < inputs.alim_ft)
    breach_any = bool(min_pred_miss < inputs.alim_ft)

    return RunResult(
        run=0,
        scenario=inputs.scenario,
        FL_PL=FL_pl,
        FL_CAT=FL_cat,
        PL_TAS=PL_TAS,
        CAT_TAS=CAT_TAS,
        PLhdg=h1,
        CAThdg=h2,
        R0NM=r0,
        closurekt=vcl,
        tgos=tgo,
        plDelay=pl_td_k,
        plAccel_g=pl_ag_k,
        catDelay=cat_td_k,
        catAccel_g=cat_ag_k,
        intruder_mode=mode,
        h0ft=h0,
        missCPAft=miss_cpa,
        minPredMiss_bothResp_ft=min_pred_miss,
        ALIMbreach_CPA=breach_cpa,
        ALIMbreach_ANY_predCPA=breach_any,
        dhPLft=dh_pl,
        dhCATft=dh_cat,
        dhbaselineft=dh_base,
        ratiobaseoverPL=ratio,
        unresolvedRRpct=unresolved_rr,
        eventtype_first=eventtype_first,
        eventtime_first=eventtime_first,
        eventcause_first=eventcause_first,
        eventtype=eventtype_final,
        eventtimes=eventtime_final,
        eventcause=eventcause_final,
    )


def run_batch_simulation(inputs: BatchSimulationInputs) -> list[RunResult]:
    rng = np.random.default_rng(inputs.seed)
    results: list[RunResult] = []
    for k in range(inputs.n_runs):
        result = run_single_case(rng, inputs)
        result.run = k + 1
        results.append(result)
    return results

