"""Command-line interface for running ACAS/TCAS batch simulations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from .simulation import BatchSimulationInputs, NonComplianceConfig, run_batch_simulation
from .statistics import wilson_ci


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "noncomp" in data and not isinstance(data["noncomp"], dict):
        raise ValueError("'noncomp' must be a mapping when provided in JSON config")
    return data


def _build_inputs(args: argparse.Namespace) -> BatchSimulationInputs:
    base_config: Dict[str, Any]
    if args.config:
        base_config = _load_config(args.config)
    else:
        base_config = {}
    overrides: Dict[str, Any] = {}
    if args.n_runs is not None:
        overrides["n_runs"] = args.n_runs
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.scenario is not None:
        overrides["scenario"] = args.scenario
    if args.r_min is not None:
        overrides["r_min"] = args.r_min
    if args.r_max is not None:
        overrides["r_max"] = args.r_max
    if args.use_distrib is not None:
        overrides["use_distrib"] = args.use_distrib
    if args.ta_only is not None:
        base_config.setdefault("noncomp", {})["ta_only"] = args.ta_only
    merged = {**base_config, **overrides}
    noncomp_cfg = merged.get("noncomp", {})
    merged["noncomp"] = NonComplianceConfig(**noncomp_cfg)
    required = {"n_runs", "seed", "scenario", "r_min", "r_max"}
    missing = sorted(required - merged.keys())
    if missing:
        raise ValueError(f"Missing required config values: {', '.join(missing)}")
    return BatchSimulationInputs(**merged)


def _summarise(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    summary: Dict[str, Any] = {"runs": n}
    if n == 0:
        return summary
    k_rev = int((df["eventtype"] == "REVERSE").sum())
    k_str = int((df["eventtype"] == "STRENGTHEN").sum())
    k_cpa = int(df["ALIMbreach_CPA"].sum())
    k_any = int(df["ALIMbreach_ANY_predCPA"].sum())
    summary.update(
        {
            "p_reverse": k_rev / n,
            "p_strengthen": k_str / n,
            "p_alim_cpa": k_cpa / n,
            "p_alim_any": k_any / n,
            "p_reverse_ci": wilson_ci(k_rev, n),
            "p_strengthen_ci": wilson_ci(k_str, n),
            "p_alim_cpa_ci": wilson_ci(k_cpa, n),
            "p_alim_any_ci": wilson_ci(k_any, n),
            "mean_unresolved_rr": float(df["unresolvedRRpct"].mean()),
        }
    )
    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    def fmt_pct(value: float) -> str:
        return f"{100 * value:,.2f}%"

    print(f"Runs completed: {summary.get('runs', 0)}")
    if summary.get("runs", 0) == 0:
        return
    print(f"P(Reversal):      {fmt_pct(summary['p_reverse'])}  CI {summary['p_reverse_ci']}")
    print(f"P(Strengthen):    {fmt_pct(summary['p_strengthen'])}  CI {summary['p_strengthen_ci']}")
    print(f"P(ALIM breach @CPA): {fmt_pct(summary['p_alim_cpa'])}  CI {summary['p_alim_cpa_ci']}")
    print(f"P(ALIM breach ANY):  {fmt_pct(summary['p_alim_any'])}  CI {summary['p_alim_any_ci']}")
    print(f"Mean unresolved RR: {summary['mean_unresolved_rr']:.3f}%")


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACAS/TCAS batch simulations without Streamlit.")
    parser.add_argument("--config", type=Path, help="Path to JSON config with BatchSimulationInputs fields.")
    parser.add_argument("--n-runs", type=int, dest="n_runs", help="Override number of Monte Carlo runs.")
    parser.add_argument("--seed", type=int, help="Override RNG seed.")
    parser.add_argument("--scenario", choices=["Head-on", "Crossing", "Overtaking", "Custom"], help="Scenario type.")
    parser.add_argument("--r-min", type=float, dest="r_min", help="Override minimum initial range (NM).")
    parser.add_argument("--r-max", type=float, dest="r_max", help="Override maximum initial range (NM).")
    parser.add_argument(
        "--use-distrib",
        dest="use_distrib",
        action="store_true",
        default=None,
        help="Force use of CAT response mixture distributions.",
    )
    parser.add_argument(
        "--no-use-distrib",
        dest="use_distrib",
        action="store_false",
        help="Force nominal CAT response parameters (no mixture).",
    )
    parser.add_argument(
        "--ta-only",
        dest="ta_only",
        action="store_true",
        default=None,
        help="Force intruder TA-only mode (unequipped).",
    )
    parser.add_argument(
        "--no-ta-only",
        dest="ta_only",
        action="store_false",
        help="Ensure intruder is equipped (disable TA-only override).",
    )
    parser.add_argument("--output", type=Path, help="Output CSV path for run results.")
    parser.add_argument("--summary", action="store_true", help="Print summary statistics to stdout.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inputs = _build_inputs(args)
    except Exception as exc:  # pragma: no cover - error formatting
        print(f"Error: {exc}")
        return 2
    results = run_batch_simulation(inputs)
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    if args.output:
        _save_csv(df, args.output)
    summary = _summarise(df)
    if args.summary:
        _print_summary(summary)
    elif not args.output:
        _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
