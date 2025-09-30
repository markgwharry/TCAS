import numpy as np

from tcas.kinematics import delta_h_piecewise, integrate_altitude_from_vs, vs_time_series


def test_delta_h_piecewise_before_delay():
    assert delta_h_piecewise(0.05, t_delay_s=0.1, a_g=0.2, v_f_fpm=1500) == 0.0


def test_delta_h_piecewise_ramp_phase():
    dh = delta_h_piecewise(5.0, t_delay_s=0.5, a_g=0.25, v_f_fpm=1500)
    assert dh > 0


def test_vs_time_series_reaches_target():
    times, vs = vs_time_series(5.0, 0.5, t_delay_s=1.0, a_g=0.25, v_f_fpm=1000, sense=1)
    assert times[0] == 0.0
    assert vs[0] == 0.0
    assert np.isclose(vs[-1], 1000, atol=1e-6)


def test_integrate_altitude_from_vs():
    times = np.array([0.0, 1.0, 2.0])
    vs = np.array([0.0, 600.0, 600.0])
    z = integrate_altitude_from_vs(times, vs)
    assert np.isclose(z[-1], 20.0)
