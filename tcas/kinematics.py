"""Kinematic helpers for vertical maneuvering."""

from __future__ import annotations

import numpy as np

from .constants import G, FT_PER_M, MS_PER_FPM


def delta_h_piecewise(t_cpa_s: float, t_delay_s: float, a_g: float, v_f_fpm: float) -> float:
    """Compute achieved altitude change by CPA using a simple ramp model."""

    a = a_g * G
    v_f_mps = v_f_fpm * MS_PER_FPM
    if t_cpa_s <= t_delay_s:
        return 0.0
    t = t_cpa_s - t_delay_s
    t_ramp = v_f_mps / a if a > 0 else np.inf
    if t <= t_ramp:
        dh_m = 0.5 * a * (t**2)
    else:
        dh_m = 0.5 * a * (t_ramp**2) + v_f_mps * (t - t_ramp)
    return dh_m * FT_PER_M


def vs_time_series(
    t_end_s: float,
    dt_s: float,
    t_delay_s: float,
    a_g: float,
    v_f_fpm: float,
    sense: int,
    cap_fpm: float | None = None,
):
    a = a_g * G
    v_target = v_f_fpm if cap_fpm is None else min(v_f_fpm, cap_fpm)
    times = np.arange(0.0, t_end_s + 1e-9, dt_s)
    vs = np.zeros_like(times, dtype=float)
    for i, t in enumerate(times):
        if t <= t_delay_s:
            vs[i] = 0.0
        else:
            te = t - t_delay_s
            v_mps = min(a * te, v_target * MS_PER_FPM)
            vs[i] = sense * (v_mps / MS_PER_FPM)
    return times, vs


def integrate_altitude_from_vs(times_s: np.ndarray, vs_fpm: np.ndarray) -> np.ndarray:
    dt = np.diff(times_s, prepend=times_s[0])
    fps = vs_fpm / 60.0
    z = np.cumsum(fps * dt)
    z[0] = 0.0
    return z

