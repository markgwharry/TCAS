"""Statistical helpers."""

from __future__ import annotations

from .constants import Z_95


def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = z * ((phat * (1 - phat) + (z * z) / (4 * n)) / n) ** 0.5 / denom
    return (max(0.0, center - half), min(1.0, center + half))

