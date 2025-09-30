import numpy as np

from tcas.constants import DEFAULT_RESP_THRESH_FPM
from tcas.surrogate import late_override_monitor, predicted_miss_at_cpa, surrogate_decision_stream


def test_predicted_miss_at_cpa_symmetry():
    miss = predicted_miss_at_cpa(100.0, 500.0, -500.0, 10.0)
    assert miss >= 0


def test_surrogate_decision_strengthen():
    times = np.linspace(0, 20, num=201)
    vs_own = np.full_like(times, DEFAULT_RESP_THRESH_FPM + 50.0)
    vs_int = np.full_like(times, -(DEFAULT_RESP_THRESH_FPM + 50.0))
    sep_now = np.linspace(200.0, -400.0, num=times.size)
    events, t_strengthen = surrogate_decision_stream(times, vs_own, vs_int, sep_now, t_cpa_s=20.0, alim_ft=600.0)
    assert events
    assert events[0][0] in {"STRENGTHEN", "REVERSE"}


def test_late_override_monitor_reverse():
    times = np.linspace(0, 20, num=201)
    vs_own = np.full_like(times, DEFAULT_RESP_THRESH_FPM + 100.0)
    vs_int = np.full_like(times, -(DEFAULT_RESP_THRESH_FPM - 150.0))
    sep_now = np.linspace(200.0, -500.0, num=times.size)
    events = [("STRENGTHEN", 5.0, "ALIM_SHORTFALL_EARLY")]
    events = late_override_monitor(
        events,
        t_strengthen=5.0,
        times=times,
        vs_own=vs_own,
        vs_int=vs_int,
        sep_now=sep_now,
        t_cpa_s=20.0,
        alim_ft=600.0,
        resp_thr=DEFAULT_RESP_THRESH_FPM,
        intruder_delay_s=4.0,
        noncomp_grace_s=1.0,
        ta_only=False,
    )
    assert events[-1][0] in {"REVERSE", "STRENGTHEN"}
