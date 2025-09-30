from __future__ import annotations

import io
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from . import (
    BatchSimulationInputs,
    NonComplianceConfig,
    PL_ACCEL_G,
    PL_DELAY_S,
    PL_IAS_KT,
    PL_VS_CAP,
    PL_VS_FPM,
    baseline_dh_ft,
    delta_h_piecewise,
    ias_to_tas,
    relative_closure_kt,
    run_batch_simulation,
    time_to_go_from_geometry,
    wilson_ci,
)


SCENARIO_REL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Head-on": (150.0, 210.0),
    "Crossing": (60.0, 120.0),
    "Overtaking": (0.0, 30.0),
}

STATE_KEY_RESULTS = "tcas_last_results"
STATE_KEY_INPUTS = "tcas_last_inputs"
STATE_KEY_TIMESTAMP = "tcas_last_run_ts"


def init_state() -> None:
    if STATE_KEY_RESULTS not in st.session_state:
        st.session_state[STATE_KEY_RESULTS] = None
    if STATE_KEY_INPUTS not in st.session_state:
        st.session_state[STATE_KEY_INPUTS] = None
    if STATE_KEY_TIMESTAMP not in st.session_state:
        st.session_state[STATE_KEY_TIMESTAMP] = None


def normalise_for_json(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: normalise_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [normalise_for_json(v) for v in data]
    if isinstance(data, tuple):
        return [normalise_for_json(v) for v in data]
    if isinstance(data, np.generic):
        return data.item()
    return data


@st.cache_data(show_spinner=True)
def run_batch_cached(payload: Dict[str, Any]) -> pd.DataFrame:
    payload_copy = dict(payload)
    noncomp_payload = dict(payload_copy.pop("noncomp"))
    batch_inputs = BatchSimulationInputs(
        noncomp=NonComplianceConfig(**noncomp_payload),
        **payload_copy,
    )
    results = run_batch_simulation(batch_inputs)
    rows = [asdict(r) for r in results]
    return pd.DataFrame(rows)


def render_sidebar() -> Tuple[Dict[str, Any], NonComplianceConfig]:
    with st.sidebar:
        st.header("Global settings")
        alim_ft = st.number_input("ALIM (ft)", value=600.0, step=50.0, min_value=200.0)
        baseline = st.selectbox(
            "Baseline for RR scaling",
            ["IDEAL 1500 fpm (ACASA 2002)", "STANDARD 1500 fpm (EUROCONTROL 2018)"],
        )
        dt = st.number_input("Time step (s)", value=0.10, step=0.05, min_value=0.01, format="%.2f")
        resp_thr = st.number_input(
            "Meaningful response threshold (fpm)", value=300.0, step=50.0, min_value=50.0
        )
        st.divider()
        st.subheader("Performance-limited (PL) — fixed")
        st.caption(
            f"Delay {PL_DELAY_S:.1f} s, accel {PL_ACCEL_G:.2f} g, target ±{PL_VS_FPM} fpm (cap {PL_VS_CAP})"
        )
        st.caption(f"Speed {PL_IAS_KT:.0f} KIAS → TAS computed per flight level.")
        st.divider()
        st.subheader("CAT (non-PL) — batch parameters")
        cat_vs = st.number_input("CAT target VS (fpm)", value=1500.0, step=100.0, min_value=200.0)
        cat_cap = st.number_input("CAT performance cap (fpm)", value=2000.0, step=100.0, min_value=500.0)
        cat_ag_nom = st.number_input(
            "CAT accel nominal (g)", value=0.25, step=0.01, format="%.2f", min_value=0.05
        )
        cat_td_nom = st.number_input("CAT delay nominal (s)", value=5.0, step=0.5, min_value=0.0)
        cat_tas_min = st.number_input("CAT TAS min (kt)", value=420.0, step=5.0, min_value=200.0)
        cat_tas_max = st.number_input("CAT TAS max (kt)", value=470.0, step=5.0, min_value=200.0)
        with st.expander("RA trigger & surveillance/noise"):
            ra_trigger_mode = st.selectbox(
                "RA→CPA mode",
                ["Scenario-calibrated (recommended)", "Geometry-derived"],
            )
            tgo_cap = st.number_input("Max RA→CPA cap (s)", value=60.0, step=5.0, min_value=10.0)
            p_miss = st.slider(
                "P(missing cycle) per time-step (surrogate only)", 0.0, 0.20, 0.00, 0.01
            )
        with st.expander("Intruder (CAT) non-compliance priors"):
            p_opposite = st.slider("P(opposite-sense) per run", 0.0, 0.10, 0.02, 0.005)
            p_leveloff = st.slider("P(level-off / follow ATC) per run", 0.0, 0.10, 0.03, 0.005)
            p_persist = st.slider("P(persistent weak <300 fpm) per run", 0.0, 0.05, 0.01, 0.005)
            jitter = st.checkbox("Jitter priors per run (±50%)", value=True)
            ta_only = st.checkbox("TA-only / unequipped intruder (sensitivity)", value=False)
        with st.expander("Initial vertical miss (at RA)"):
            h0_mean = st.number_input("h0 mean (ft)", value=250.0, step=25.0, min_value=0.0)
            h0_sd = st.number_input("h0 std dev (ft)", value=100.0, step=25.0, min_value=0.0)
            h0_lo = st.number_input("h0 min (ft)", value=100.0, step=25.0, min_value=0.0)
            h0_hi = st.number_input("h0 max (ft)", value=500.0, step=25.0, min_value=0.0)
        st.divider()
    config = {
        "alim_ft": alim_ft,
        "baseline": baseline,
        "dt": dt,
        "resp_thr": resp_thr,
        "cat_vs": cat_vs,
        "cat_cap": cat_cap,
        "cat_ag_nom": cat_ag_nom,
        "cat_td_nom": cat_td_nom,
        "cat_tas_min": cat_tas_min,
        "cat_tas_max": cat_tas_max,
        "ra_trigger_mode": ra_trigger_mode,
        "tgo_cap": tgo_cap,
        "p_miss": p_miss,
        "h0_mean": h0_mean,
        "h0_sd": h0_sd,
        "h0_lo": h0_lo,
        "h0_hi": h0_hi,
    }
    noncomp = NonComplianceConfig(
        p_opposite=p_opposite,
        p_leveloff=p_leveloff,
        p_persist=p_persist,
        jitter=jitter,
        ta_only=ta_only,
    )
    return config, noncomp


def validate_config(config: Dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config["dt"] <= 0:
        errors.append("Time step must be greater than zero.")
    if config["cat_tas_min"] > config["cat_tas_max"]:
        errors.append("CAT TAS min cannot exceed max.")
    if config["h0_lo"] > config["h0_hi"]:
        errors.append("h0 min cannot exceed h0 max.")
    if not (config["h0_lo"] <= config["h0_mean"] <= config["h0_hi"]):
        errors.append("h0 mean must fall between h0 min and h0 max.")
    if config["tgo_cap"] < 6.0:
        errors.append("Max RA→CPA cap must be at least 6 seconds.")
    return errors


def render_spot_check(config: Dict[str, Any]) -> None:
    st.subheader("Single-run spot check")
    c1, c2, c3 = st.columns(3)
    with c1:
        spot_FL_pl = st.number_input(
            "Spot FL (PL)", value=200, step=10, min_value=150, max_value=300, key="spot_fl_pl"
        )
    with c2:
        spot_FL_cat = st.number_input(
            "Spot FL (CAT)", value=200, step=10, min_value=150, max_value=300, key="spot_fl_cat"
        )
    with c3:
        spot_h0 = st.number_input(
            "Spot initial vertical miss h0 (ft)", value=250.0, step=25.0, key="spot_h0"
        )
    pl_tas = ias_to_tas(PL_IAS_KT, spot_FL_pl * 100.0)
    cat_tas_mid = 0.5 * (config["cat_tas_min"] + config["cat_tas_max"])
    cat_tas = cat_tas_mid
    v_closure = relative_closure_kt(pl_tas, 0.0, cat_tas, 180.0)
    tgo_geom = time_to_go_from_geometry(8.0, v_closure)
    t_cpa = float(np.clip(tgo_geom if tgo_geom is not None else 30.0, 20.0, config["tgo_cap"]))
    dh_pl = delta_h_piecewise(t_cpa, PL_DELAY_S, PL_ACCEL_G, PL_VS_FPM)
    dh_cat = delta_h_piecewise(t_cpa, config["cat_td_nom"], config["cat_ag_nom"], config["cat_vs"])
    dh_base = baseline_dh_ft(t_cpa, mode=config["baseline"])
    ratio = dh_base / dh_pl if dh_pl > 1e-6 else np.nan
    unres_rr = 1.1 * ratio
    spot_tab = pd.DataFrame(
        {
            "Aircraft": ["PL (ownship)", "CAT (intruder)", "Baseline"],
            "Δh @ CPA (ft)": [dh_pl, dh_cat, dh_base],
        }
    )
    st.dataframe(spot_tab, use_container_width=True)
    st.caption(
        f"Scaled unresolved RR ≈ {unres_rr:,.3f}% (ratio {ratio:,.3f}); t_go ≈ {t_cpa:.1f}s; h0 = {spot_h0:.0f} ft"
    )


def render_batch_form(config: Dict[str, Any], disable: bool = False) -> Tuple[bool, Dict[str, Any]]:
    st.markdown("---")
    st.header("Batch Monte Carlo")
    with st.form("batch_form", clear_on_submit=False):
        n_runs = st.number_input("Number of runs", min_value=1, max_value=100000, value=5000, step=100)
        seed = st.number_input("Random seed", value=42, step=1)
        scenario = st.selectbox(
            "Scenario",
            list(SCENARIO_REL_BOUNDS.keys()) + ["Custom"],
        )
        r_min = st.number_input("Initial range min (NM)", value=5.0, step=0.5, min_value=0.5)
        r_max = st.number_input("Initial range max (NM)", value=12.0, step=0.5, min_value=1.0)
        if scenario == "Custom":
            hdg1_min = st.number_input("PL heading min (deg)", value=0.0, step=5.0)
            hdg1_max = st.number_input("PL heading max (deg)", value=360.0, step=5.0)
            hdg2_min = st.number_input("CAT heading min (deg)", value=0.0, step=5.0)
            hdg2_max = st.number_input("CAT heading max (deg)", value=360.0, step=5.0)
        else:
            hdg1_min, hdg1_max = 0.0, 360.0
            hdg2_min = hdg2_max = None
        use_distrib = st.checkbox(
            "CAT response: use mixture distributions (recommended)", value=True
        )
        submitted = st.form_submit_button("Run batch", disabled=disable)
    form_data = {
        "n_runs": int(n_runs),
        "seed": int(seed),
        "scenario": scenario,
        "r_min": r_min,
        "r_max": r_max,
        "hdg1_min": hdg1_min,
        "hdg1_max": hdg1_max,
        "hdg2_min": hdg2_min,
        "hdg2_max": hdg2_max,
        "use_distrib": use_distrib,
    }
    return submitted, form_data


def validate_form(form_data: Dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if form_data["r_min"] > form_data["r_max"]:
        errors.append("Initial range min cannot exceed max.")
    if form_data["scenario"] == "Custom":
        if form_data["hdg1_min"] > form_data["hdg1_max"]:
            errors.append("PL heading min cannot exceed max.")
        if form_data["hdg2_min"] > form_data["hdg2_max"]:
            errors.append("CAT heading min cannot exceed max.")
    return errors


def build_inputs(
    config: Dict[str, Any],
    noncomp: NonComplianceConfig,
    form_data: Dict[str, Any],
) -> BatchSimulationInputs:
    scenario = form_data["scenario"]
    rel_min, rel_max = SCENARIO_REL_BOUNDS.get(scenario, (None, None))
    return BatchSimulationInputs(
        n_runs=form_data["n_runs"],
        seed=form_data["seed"],
        scenario=scenario,
        r_min=form_data["r_min"],
        r_max=form_data["r_max"],
        hdg1_min=form_data["hdg1_min"],
        hdg1_max=form_data["hdg1_max"],
        hdg2_min=form_data.get("hdg2_min"),
        hdg2_max=form_data.get("hdg2_max"),
        rel_min=rel_min,
        rel_max=rel_max,
        ra_trigger_mode=config["ra_trigger_mode"],
        tgo_cap=config["tgo_cap"],
        use_distrib=form_data["use_distrib"],
        alim_ft=config["alim_ft"],
        baseline=config["baseline"],
        dt=config["dt"],
        resp_thr=config["resp_thr"],
        cat_vs=config["cat_vs"],
        cat_cap=config["cat_cap"],
        cat_ag_nom=config["cat_ag_nom"],
        cat_td_nom=config["cat_td_nom"],
        cat_tas_min=config["cat_tas_min"],
        cat_tas_max=config["cat_tas_max"],
        h0_mean=config["h0_mean"],
        h0_sd=config["h0_sd"],
        h0_lo=config["h0_lo"],
        h0_hi=config["h0_hi"],
        p_miss=config["p_miss"],
        noncomp=noncomp,
    )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.subheader("Explore batch")
        tgo_min = float(df["tgos"].min())
        tgo_max = float(df["tgos"].max())
        if tgo_min == tgo_max:
            st.caption(f"tgo fixed at {tgo_min:.1f}s")
            tgo_lo, tgo_hi = tgo_min, tgo_max
        else:
            tgo_lo, tgo_hi = st.slider(
                "tgo window (s)",
                min_value=float(max(6.0, tgo_min)),
                max_value=float(tgo_max),
                value=(float(tgo_min), float(tgo_max)),
                key="filter_tgo",
            )
        modes = sorted(m for m in df["intruder_mode"].dropna().unique())
        selected_modes = st.multiselect(
            "Intruder modes",
            modes,
            default=modes,
            key="filter_modes",
        )
        events = sorted(e for e in df["eventtype"].dropna().unique())
        selected_events = st.multiselect(
            "RA outcomes",
            events,
            default=events,
            key="filter_events",
        )
        alt_min = int(df["FL_PL"].min())
        alt_max = int(df["FL_PL"].max())
        if alt_min == alt_max:
            st.caption(f"PL flight level fixed at FL{alt_min}")
            alt_range = (alt_min, alt_max)
        else:
            alt_range = st.slider(
                "PL flight level window",
                min_value=alt_min,
                max_value=alt_max,
                value=(alt_min, alt_max),
                step=5,
                key="filter_alt",
            )
        breach_any = st.checkbox(
            "Only ALIM breaches (ANY predicted CPA)", value=False, key="filter_any"
        )
        breach_cpa = st.checkbox(
            "Only ALIM breaches @ CPA", value=False, key="filter_cpa"
        )
        st.divider()
    view = df[df["tgos"].between(tgo_lo, tgo_hi)]
    if selected_modes:
        view = view[view["intruder_mode"].isin(selected_modes)]
    if selected_events:
        view = view[view["eventtype"].isin(selected_events)]
    if alt_range:
        view = view[view["FL_PL"].between(*alt_range)]
    if breach_any:
        view = view[view["ALIMbreach_ANY_predCPA"]]
    if breach_cpa:
        view = view[view["ALIMbreach_CPA"]]
    return view


def render_metrics(view: pd.DataFrame, total_runs: int) -> None:
    n = len(view)
    if n == 0:
        st.info("No runs match the current filters.")
        return
    k_rev = int((view["eventtype"] == "REVERSE").sum())
    k_str = int((view["eventtype"] == "STRENGTHEN").sum())
    k_cpa = int(view["ALIMbreach_CPA"].sum())
    k_any = int(view["ALIMbreach_ANY_predCPA"].sum())
    p_rev = k_rev / n
    p_str = k_str / n
    p_cpa = k_cpa / n
    p_any = k_any / n
    lo_rev, hi_rev = wilson_ci(k_rev, n)
    lo_str, hi_str = wilson_ci(k_str, n)
    lo_cpa, hi_cpa = wilson_ci(k_cpa, n)
    lo_any, hi_any = wilson_ci(k_any, n)
    mean_rr = view["unresolvedRRpct"].mean()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("P(Reversal)", f"{100 * p_rev:,.2f}%", f"{100 * lo_rev:,.2f}–{100 * hi_rev:,.2f}%")
    c2.metric("P(Strengthen)", f"{100 * p_str:,.2f}%", f"{100 * lo_str:,.2f}–{100 * hi_str:,.2f}%")
    c3.metric("Mean RR", f"{mean_rr:.3f}%")
    c4.metric("P(ALIM breach @CPA)", f"{100 * p_cpa:,.2f}%", f"{100 * lo_cpa:,.2f}–{100 * hi_cpa:,.2f}%")
    c5.metric("P(ALIM breach ANY)", f"{100 * p_any:,.2f}%", f"{100 * lo_any:,.2f}–{100 * hi_any:,.2f}%")
    st.caption(f"{n} runs match filters (out of {total_runs}).")


def render_percentiles(view: pd.DataFrame) -> None:
    values = view["unresolvedRRpct"].dropna().to_numpy()
    if values.size == 0:
        return
    percentiles = [5, 25, 50, 75, 95]
    rr_percentiles = [np.percentile(values, p) for p in percentiles]
    table = pd.DataFrame(
        {
            "Percentile": percentiles,
            "Unresolved RR (%)": [f"{val:.3f}" for val in rr_percentiles],
        }
    )
    st.subheader("Unresolved RR percentiles")
    st.table(table)


def render_plot(fig: plt.Figure, label: str, key: str) -> None:
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        f"Download {label}",
        data=buf,
        file_name=f"{key}.png",
        mime="image/png",
        key=f"dl_{key}",
    )
    plt.close(fig)


def render_plots(view: pd.DataFrame, config: Dict[str, Any]) -> None:
    st.subheader("Visualisations")
    vals = view["unresolvedRRpct"].dropna().to_numpy()
    if vals.size:
        fig_ecdf, ax_ecdf = plt.subplots(figsize=(6, 3))
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)
        ax_ecdf.plot(x, y)
        ax_ecdf.set_xlabel("Unresolved RR (%)")
        ax_ecdf.set_ylabel("ECDF")
        ax_ecdf.grid(True, alpha=0.3)
        render_plot(fig_ecdf, "unresolved-rr-ecdf", "ecdf_rr")
    if not view.empty:
        fig_hist, ax_hist = plt.subplots(figsize=(6, 3))
        for lab in sorted(view["eventtype"].dropna().unique()):
            sub = view.loc[view["eventtype"] == lab, "tgos"].dropna()
            if not sub.empty:
                ax_hist.hist(sub, bins=24, histtype="step", label=lab)
        ax_hist.set_xlabel("tgo (s)")
        ax_hist.set_ylabel("Count")
        ax_hist.grid(True, alpha=0.3)
        if len(ax_hist.lines) or len(ax_hist.patches):
            ax_hist.legend()
        render_plot(fig_hist, "tgo-by-event", "tgo_hist")
    if "minPredMiss_bothResp_ft" in view.columns and not view.empty:
        fig_miss, ax_miss = plt.subplots(figsize=(6, 3))
        ax_miss.hist(view["minPredMiss_bothResp_ft"].dropna(), bins=30)
        ax_miss.axvline(
            config["alim_ft"], color="k", ls="--", alpha=0.7, label=f"ALIM = {config['alim_ft']:.0f} ft"
        )
        ax_miss.set_xlabel("Min predicted miss (ft) after both responding")
        ax_miss.set_ylabel("Count")
        ax_miss.grid(True, alpha=0.3)
        ax_miss.legend()
        render_plot(fig_miss, "min-predicted-miss", "min_pred_miss")


def render_event_breakdown(view: pd.DataFrame) -> None:
    cause_counts = view["eventcause"].value_counts()
    mode_counts = view["intruder_mode"].value_counts()
    if cause_counts.empty and mode_counts.empty:
        return
    st.subheader("Categorical breakdowns")
    col1, col2 = st.columns(2)
    if not cause_counts.empty:
        with col1:
            fig_cause, ax_cause = plt.subplots(figsize=(6, 3))
            ax_cause.bar(cause_counts.index, cause_counts.values)
            ax_cause.set_ylabel("Count")
            ax_cause.set_title("Final RA causes")
            ax_cause.tick_params(axis="x", rotation=20)
            render_plot(fig_cause, "event-causes", "event_causes")
    if not mode_counts.empty:
        with col2:
            fig_mode, ax_mode = plt.subplots(figsize=(6, 3))
            ax_mode.bar(mode_counts.index, mode_counts.values, color="#7f7f7f")
            ax_mode.set_ylabel("Count")
            ax_mode.set_title("Intruder modes")
            ax_mode.tick_params(axis="x", rotation=20)
            render_plot(fig_mode, "intruder-modes", "intruder_modes")


def render_downloads(view: pd.DataFrame) -> None:
    st.subheader("Downloads")
    csv_buf = io.BytesIO()
    view.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    st.download_button(
        "Download filtered CSV",
        data=csv_buf,
        file_name="tcas_filtered_results.csv",
        mime="text/csv",
        key="dl_filtered_csv",
    )
    payload = st.session_state.get(STATE_KEY_INPUTS)
    if payload:
        json_buf = io.BytesIO()
        json_buf.write(json.dumps(normalise_for_json(payload), indent=2).encode("utf-8"))
        json_buf.seek(0)
        st.download_button(
            "Download batch configuration (JSON)",
            data=json_buf,
            file_name="tcas_batch_config.json",
            mime="application/json",
            key="dl_config_json",
        )


def display_results(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    st.markdown("---")
    st.header("Batch results")
    view = apply_filters(df)
    total = len(df)
    st.caption(f"Showing {len(view)} of {total} runs after filters.")
    if view.empty:
        st.warning("No runs match the current filters.")
        return
    render_metrics(view, total)
    render_percentiles(view)
    render_plots(view, config)
    render_event_breakdown(view)
    render_downloads(view)
    st.subheader("Preview of filtered results")
    st.dataframe(view.head(200), use_container_width=True)


def render_methodology() -> None:
    with st.expander("Methodology & assumptions"):
        st.markdown(
            """
**Model scope**
- Two-aircraft encounters in Class A airspace (FL150–FL300) with a fixed performance-limited ownship and a variable CAT intruder.
- Surrogate RA taxonomy v7.1 with two-phase strengthen/reverse monitoring and ALIM shortfall logic.
- Baseline residual risk scales unresolved probability by Δh ratio × 1.1% reference risk.

**Implementation notes**
- Core Monte Carlo logic lives in `tcas.simulation` and is reused by this Streamlit UI and the CLI (`python -m tcas.cli`).
- Cached batch runs reuse prior results when inputs match, improving iteration speed.
- Sidebar validation guards against inconsistent parameter ranges before launching a batch.

**Outputs**
- KPIs include reversal/strengthen probabilities, ALIM breaches at CPA and for any predicted miss, and the unresolved residual risk distribution.
- Download buttons provide CSV results, configuration JSON, and plot images for reports or regulator briefings.
            """
        )


def main() -> None:
    st.set_page_config(page_title="ACAS/TCAS Residual Risk", layout="wide")
    init_state()
    st.title("ACAS/TCAS v7.1 — Residual Risk & RA Taxonomy (Batch Monte Carlo)")
    st.markdown(
        """
Two aircraft in **Class A (FL150–FL300)**: one **performance-limited (PL)** is **fixed**, the **CAT intruder** varies in speed, headings, delay, and compliance. Configure parameters on the left, spot-check a scenario, then run batch Monte Carlo to quantify residual risk metrics.
        """
    )
    config, noncomp = render_sidebar()
    config_errors = validate_config(config)
    for err in config_errors:
        st.sidebar.error(f"⚠️ {err}")
    render_spot_check(config)
    submitted, form_data = render_batch_form(config, disable=bool(config_errors))
    if submitted:
        form_errors = validate_form(form_data)
        if form_errors:
            for err in form_errors:
                st.error(f"• {err}")
        elif config_errors:
            st.error("Resolve configuration issues in the sidebar before running a batch.")
        else:
            inputs = build_inputs(config, noncomp, form_data)
            payload = asdict(inputs)
            df = run_batch_cached(payload)
            st.session_state[STATE_KEY_RESULTS] = df
            st.session_state[STATE_KEY_INPUTS] = payload
            st.session_state[STATE_KEY_TIMESTAMP] = datetime.utcnow().isoformat(timespec="seconds")
            st.success(f"Completed {len(df)} runs.")
    df = st.session_state.get(STATE_KEY_RESULTS)
    if df is not None and not df.empty:
        display_results(df, config)
    else:
        st.info("Run a batch to see results.")
    render_methodology()


if __name__ == "__main__":
    main()

