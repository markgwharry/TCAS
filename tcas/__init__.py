"""Reusable components for the ACAS/TCAS residual risk simulator."""

from .constants import (
    ALIM_MARGIN_FT,
    DEFAULT_ALIM_FT,
    DEFAULT_RESP_THRESH_FPM,
    FT_PER_M,
    G,
    MS_PER_FPM,
    PL_ACCEL_G,
    PL_DELAY_S,
    PL_IAS_KT,
    PL_VS_CAP,
    PL_VS_FPM,
    Z_95,
)
from .kinematics import delta_h_piecewise, integrate_altitude_from_vs, vs_time_series
from .geometry import (
    ias_to_tas,
    relative_closure_kt,
    sample_headings,
    time_to_go_from_geometry,
)
from .sampling import (
    baseline_dh_ft,
    sample_altitudes_and_h0,
    sample_pilot_response_cat,
    sample_tgo_with_trigger,
    apply_surveillance_noise,
)
from .surrogate import (
    dwell_fn,
    late_override_monitor,
    predicted_miss_at_cpa,
    surrogate_decision_stream,
)
from .statistics import wilson_ci
from .simulation import (
    BatchSimulationInputs,
    NonComplianceConfig,
    RunResult,
    run_batch_simulation,
    run_single_case,
)

__all__ = [
    "ALIM_MARGIN_FT",
    "BatchSimulationInputs",
    "DEFAULT_ALIM_FT",
    "DEFAULT_RESP_THRESH_FPM",
    "FT_PER_M",
    "G",
    "MS_PER_FPM",
    "NonComplianceConfig",
    "PL_ACCEL_G",
    "PL_DELAY_S",
    "PL_IAS_KT",
    "PL_VS_CAP",
    "PL_VS_FPM",
    "RunResult",
    "Z_95",
    "apply_surveillance_noise",
    "baseline_dh_ft",
    "delta_h_piecewise",
    "dwell_fn",
    "ias_to_tas",
    "integrate_altitude_from_vs",
    "late_override_monitor",
    "predicted_miss_at_cpa",
    "relative_closure_kt",
    "run_batch_simulation",
    "run_single_case",
    "sample_altitudes_and_h0",
    "sample_headings",
    "sample_pilot_response_cat",
    "sample_tgo_with_trigger",
    "surrogate_decision_stream",
    "time_to_go_from_geometry",
    "vs_time_series",
    "wilson_ci",
]

