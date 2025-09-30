"""Simulation constants for the ACAS/TCAS residual risk model."""

from __future__ import annotations

G = 9.80665
FT_PER_M = 3.28084
MS_PER_FPM = 0.00508

DEFAULT_ALIM_FT = 600.0
DEFAULT_RESP_THRESH_FPM = 300.0
ALIM_MARGIN_FT = 100.0

Z_95 = 1.96

# Performance-limited (PL) — fixed
PL_DELAY_S = 0.1
PL_ACCEL_G = 0.10
PL_VS_FPM = 500
PL_VS_CAP = 500
PL_IAS_KT = 120.0

