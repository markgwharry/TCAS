"""Geometric helpers for encounter setup."""

from __future__ import annotations

import numpy as np


def ias_to_tas(ias_kt: float, pressure_alt_ft: float) -> float:
    """Rough ISA conversion from indicated to true airspeed for FL150–FL300."""

    sigma = (1.0 - 6.875e-6 * pressure_alt_ft) ** 4.256
    sigma = max(1e-3, sigma)
    return ias_kt / np.sqrt(sigma)


def relative_closure_kt(v1_kt: float, hdg1_deg: float, v2_kt: float, hdg2_deg: float) -> float:
    th1, th2 = np.deg2rad(hdg1_deg), np.deg2rad(hdg2_deg)
    v1 = np.array([v1_kt * np.sin(th1), v1_kt * np.cos(th1)])
    v2 = np.array([v2_kt * np.sin(th2), v2_kt * np.cos(th2)])
    return float(np.linalg.norm(v1 - v2))


def time_to_go_from_geometry(r0_nm: float, v_closure_kt: float) -> float | None:
    if v_closure_kt <= 1e-6:
        return None
    return 3600.0 * (r0_nm / v_closure_kt)


def sample_headings(
    rng: np.random.Generator,
    scenario: str,
    hdg1_min: float,
    hdg1_max: float,
    rel_min: float | None = None,
    rel_max: float | None = None,
    hdg2_min: float | None = None,
    hdg2_max: float | None = None,
):
    h1 = float(rng.uniform(hdg1_min, hdg1_max))
    if scenario == "Custom":
        assert hdg2_min is not None and hdg2_max is not None
        h2 = float(rng.uniform(hdg2_min, hdg2_max))
    else:
        rel = float(rng.uniform(rel_min, rel_max)) if (rel_min is not None and rel_max is not None) else 0.0
        dirsign = 1 if rng.uniform() < 0.5 else -1
        h2 = (h1 + dirsign * rel) % 360.0
    return h1, h2

