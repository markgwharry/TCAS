"""Sampling utilities for responses, altitudes, and noise."""

from __future__ import annotations

import numpy as np

from .constants import ALIM_MARGIN_FT, DEFAULT_RESP_THRESH_FPM
from .kinematics import delta_h_piecewise


def baseline_dh_ft(t_cpa_s: float, mode: str = "IDEAL") -> float:
    if mode.startswith("IDEAL"):
        return delta_h_piecewise(t_cpa_s, t_delay_s=1.0, a_g=0.25, v_f_fpm=1500)
    return delta_h_piecewise(t_cpa_s, t_delay_s=5.0, a_g=0.25, v_f_fpm=1500)


def sample_pilot_response_cat(rng: np.random.Generator) -> tuple[float, float]:
    u = rng.uniform()
    if u < 0.70:
        delay = max(0.0, rng.normal(4.5, 1.0))
        accel = max(0.05, rng.normal(0.25, 0.03))
    else:
        delay = max(0.0, rng.normal(8.5, 1.5))
        accel = max(0.05, rng.normal(0.10, 0.02))
    return float(delay), float(accel)


def sample_altitudes_and_h0(
    rng: np.random.Generator,
    h0_mean: float = 250.0,
    h0_sd: float = 100.0,
    h0_lo: float = 100.0,
    h0_hi: float = 500.0,
) -> tuple[int, int, float]:
    FL_pl = int(rng.integers(150, 301))
    FL_cat = int(rng.integers(150, 301))
    h0 = float(np.clip(rng.normal(h0_mean, h0_sd), h0_lo, h0_hi))
    return FL_pl, FL_cat, h0


def sample_tgo_with_trigger(
    rng: np.random.Generator,
    scenario: str,
    tgo_geom: float | None,
    FL_pl: int,
    FL_cat: int,
    cap_s: float = 60.0,
) -> float:
    base = {"Head-on": (25.0, 5.0), "Crossing": (22.0, 6.0), "Overtaking": (30.0, 8.0)}
    mu, sd = base.get(scenario, (25.0, 6.0))
    if FL_pl >= 250 and FL_cat >= 250:
        mu += 2.0
    lo, hi = 12.0, min(tgo_geom if tgo_geom is not None else cap_s, cap_s)
    if hi <= lo:
        return float(max(8.0, min(tgo_geom or 30.0, cap_s)))
    return float(np.clip(rng.normal(mu, sd), lo, hi))


def apply_surveillance_noise(
    rng: np.random.Generator,
    times: np.ndarray,
    vs_own: np.ndarray,
    vs_int: np.ndarray,
    p_miss: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    vs_own_noisy = vs_own.copy()
    vs_int_noisy = vs_int.copy()
    for i in range(1, len(times)):
        if rng.uniform() < p_miss:
            vs_own_noisy[i] = vs_own_noisy[i - 1]
            vs_int_noisy[i] = vs_int_noisy[i - 1]
    return vs_own_noisy, vs_int_noisy

