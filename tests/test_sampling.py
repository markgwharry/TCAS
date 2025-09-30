import numpy as np

from tcas.sampling import baseline_dh_ft, sample_pilot_response_cat, sample_tgo_with_trigger


def test_baseline_dh_ft_modes():
    t = 20.0
    ideal = baseline_dh_ft(t, mode="IDEAL")
    standard = baseline_dh_ft(t, mode="STANDARD")
    assert ideal > standard


def test_sample_pilot_response_cat_range():
    rng = np.random.default_rng(123)
    delays = []
    accels = []
    for _ in range(1000):
        delay, accel = sample_pilot_response_cat(rng)
        assert delay >= 0
        assert accel >= 0.05
        delays.append(delay)
        accels.append(accel)
    assert min(delays) >= 0
    assert min(accels) >= 0.05


def test_sample_tgo_with_trigger_bounds():
    rng = np.random.default_rng(321)
    for _ in range(50):
        tgo = sample_tgo_with_trigger(rng, "Head-on", tgo_geom=30.0, FL_pl=200, FL_cat=210, cap_s=40.0)
        assert 12.0 <= tgo <= 40.0
