"""Surrogate RA decision logic (two-phase)."""

from __future__ import annotations

import numpy as np

from .constants import ALIM_MARGIN_FT, DEFAULT_RESP_THRESH_FPM


def dwell_fn(tgo_s: float) -> float:
    return float(np.clip(0.8 + 0.05 * (tgo_s - 12.0), 0.8, 1.8))


def predicted_miss_at_cpa(
    sep_now_ft: float, vs_own_fpm: float, vs_int_fpm: float, t_go_s: float
) -> float:
    return abs(sep_now_ft + (vs_own_fpm - vs_int_fpm) * (t_go_s / 60.0))


def surrogate_decision_stream(
    times: np.ndarray,
    vs_own: np.ndarray,
    vs_int: np.ndarray,
    sep_now: np.ndarray,
    t_cpa_s: float,
    alim_ft: float,
    resp_thr: float = DEFAULT_RESP_THRESH_FPM,
):
    events: list[tuple[str, float, str]] = []
    t_strengthen = np.nan
    min_eval_time = 0.2
    for i, t in enumerate(times):
        t_go = max(0.0, t_cpa_s - t)
        if t_go <= 0.0:
            break
        own_ok = vs_own[i] >= resp_thr
        if not events and t >= min_eval_time and own_ok:
            miss_pred = predicted_miss_at_cpa(sep_now[i], vs_own[i], vs_int[i], t_go)
            if miss_pred < (alim_ft - ALIM_MARGIN_FT):
                if t_go < 6.0:
                    events.append(("REVERSE", float(t), "ALIM_SHORTFALL_LATE"))
                    return events, t_strengthen
                events.append(("STRENGTHEN", float(t), "ALIM_SHORTFALL_EARLY"))
                t_strengthen = float(t)
                break
    return events, t_strengthen


def late_override_monitor(
    events: list[tuple[str, float, str]],
    t_strengthen: float,
    times: np.ndarray,
    vs_own: np.ndarray,
    vs_int: np.ndarray,
    sep_now: np.ndarray,
    t_cpa_s: float,
    alim_ft: float,
    resp_thr: float,
    intruder_delay_s: float,
    noncomp_grace_s: float,
    ta_only: bool,
) -> list[tuple[str, float, str]]:
    have_strengthen = any(e[0] == "STRENGTHEN" for e in events)
    if not have_strengthen:
        return events
    t_str = next(e[1] for e in events if e[0] == "STRENGTHEN")
    for i, t in enumerate(times):
        t_go = max(0.0, t_cpa_s - t)
        if t_go <= 0.0:
            break
        own_ok = vs_own[i] >= resp_thr
        int_ok = ((-vs_int[i]) >= resp_thr)
        if (not ta_only) and own_ok:
            if (t >= (intruder_delay_s + noncomp_grace_s)) and ((t - t_str) >= dwell_fn(t_go)):
                if not int_ok:
                    events.append(("REVERSE", float(t), "INTRUDER_NONCOMPL_AFTER_DWELL"))
                    return events
        if own_ok and (t_go < 6.0):
            miss_pred = predicted_miss_at_cpa(sep_now[i], vs_own[i], vs_int[i], t_go)
            if miss_pred < (alim_ft - ALIM_MARGIN_FT):
                events.append(("REVERSE", float(t), "ALIM_SHORTFALL_LATE"))
                return events
    return events

