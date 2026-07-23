"""
Unit / characterization tests for the horizontal wind-driven impaction model
(heliosoil.horizontal_impaction).

Mirrors the style of tests/test_model_components.py: small synthetic inputs,
independently re-derived reference values, tight tolerances for deterministic
formulas, and a looser end-to-end smoke test for the stochastic MLE fit.
"""

import types
import numpy as np
import pytest

import heliosoil.base_models as smb
from heliosoil.horizontal_impaction import ConstantMeanWindBase, ConstantMeanWindDeposition, wind_projection_factors, parse_orientation_names

RTOL = 1e-10
ATOL = 0.0


def _wind_helios_stub(**arrays):
    """Minimal stand-in for Heliostats, carrying only what calculate_delta_soiled_area
    (and, where present, compute_soiling_factor/predict_soiling_factor) touch."""
    helios = types.SimpleNamespace(delta_soiled_area={}, delta_soiled_area_variance={})
    for name, value in arrays.items():
        setattr(helios, name, value)
    return helios


# ---------------------------------------------------------------------------
# 1. wind_projection_factors geometry
# ---------------------------------------------------------------------------


def test_wind_projection_factors_directional_limits():
    tilt = np.array([[45.0]])
    azimuth = np.array([[90.0]])  # mirror faces East

    # Wind coming from the East hits the mirror face head-on: windward only.
    pw, pl = wind_projection_factors(tilt, azimuth, np.array([90.0]))
    np.testing.assert_allclose(pw, np.sin(np.radians(45.0)), rtol=RTOL)
    np.testing.assert_allclose(pl, 0.0, atol=1e-12)

    # Wind coming from the West hits the back of the mirror: leeward only.
    pw2, pl2 = wind_projection_factors(tilt, azimuth, np.array([270.0]))
    np.testing.assert_allclose(pw2, 0.0, atol=1e-12)
    np.testing.assert_allclose(pl2, np.sin(np.radians(45.0)), rtol=RTOL)

    # Wind perpendicular to the mirror normal: no impaction on either face.
    pw3, pl3 = wind_projection_factors(tilt, azimuth, np.array([0.0]))
    np.testing.assert_allclose(pw3, 0.0, atol=1e-12)
    np.testing.assert_allclose(pl3, 0.0, atol=1e-12)

    # Zero tilt: sin(0) kills the horizontal term regardless of wind alignment.
    pw4, pl4 = wind_projection_factors(np.array([[0.0]]), azimuth, np.array([90.0]))
    np.testing.assert_allclose(pw4, 0.0, atol=1e-12)
    np.testing.assert_allclose(pl4, 0.0, atol=1e-12)


def test_wind_projection_factors_broadcasting():
    tilt = np.full((3, 4), 30.0)
    azimuth = np.zeros((3, 4))
    wind_dir = np.array([0.0, 90.0, 180.0, 270.0])

    pw, pl = wind_projection_factors(tilt, azimuth, wind_dir)
    assert pw.shape == (3, 4)
    assert pl.shape == (3, 4)

    s = np.sin(np.radians(30.0))
    expected_pw = np.tile([s, 0.0, 0.0, 0.0], (3, 1))
    expected_pl = np.tile([0.0, 0.0, s, 0.0], (3, 1))
    np.testing.assert_allclose(pw, expected_pw, atol=1e-12)
    np.testing.assert_allclose(pl, expected_pl, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. parse_orientation_names
# ---------------------------------------------------------------------------


def test_parse_orientation_names():
    names = ["ON_M1_T00", "OE_M2_T85", "OS_M2_T30", "OW_M4_T05"]
    az = parse_orientation_names(names)
    np.testing.assert_allclose(az, [0.0, 90.0, 180.0, 270.0])


def test_parse_orientation_names_invalid_raises():
    with pytest.raises(ValueError):
        parse_orientation_names(["XN_M1_T00"])
    with pytest.raises(ValueError):
        parse_orientation_names(["OXX_M1_T00"])


# ---------------------------------------------------------------------------
# 3. ConstantMeanWindBase.calculate_delta_soiled_area
# ---------------------------------------------------------------------------


def test_constant_mean_wind_reduces_to_base_when_omega_zero():
    """omega_windward = omega_leeward = 0 must reproduce ConstantMeanBase exactly."""
    f = 0
    mu_tilde = 0.6
    density = 40.0
    tilt = np.array([[10.0, 45.0, 70.0]])
    azimuth = np.array([[0.0, 0.0, 0.0]])
    dust_conc = np.array([30.0, 55.0, 90.0])
    wind_dir = np.array([0.0, 90.0, 180.0])
    wind_speed = np.array([3.0, 4.0, 5.0])

    model = ConstantMeanWindBase()
    model.mu_tilde = mu_tilde
    model.sigma_dep = None
    model.omega_windward = 0.0
    model.omega_leeward = 0.0
    model.sigma_dep_gamma = None
    model.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})

    sim_in = types.SimpleNamespace(
        time={f: None},
        dust=types.SimpleNamespace(TSP={f: density}),
        dust_concentration={f: dust_conc},
        dust_type={f: "TSP"},
        wind_direction={f: wind_dir},
        wind_speed={f: wind_speed},
    )

    model.calculate_delta_soiled_area(sim_in, verbose=False)

    alpha = dust_conc / density
    expected = alpha[None, :] * np.cos(np.radians(tilt)) * mu_tilde
    np.testing.assert_allclose(model.helios.delta_soiled_area[f], expected, rtol=RTOL, atol=ATOL)


def test_constant_mean_wind_delta_soiled_area_windward_and_leeward():
    f = 0
    mu_tilde = 0.4
    density = 40.0
    omega_windward = 0.3
    omega_leeward = 0.1

    # Two mirrors both facing East (azimuth=90); wind from East (windward for both)
    # at t=0, wind from West (leeward for both) at t=1.
    tilt = np.array([[50.0, 50.0], [50.0, 50.0]])
    azimuth = np.array([[90.0, 90.0], [90.0, 90.0]])
    wind_dir = np.array([90.0, 270.0])
    wind_speed = np.array([4.0, 6.0])
    dust_conc = np.array([40.0, 60.0])

    model = ConstantMeanWindBase()
    model.mu_tilde = mu_tilde
    model.sigma_dep = None
    model.omega_windward = omega_windward
    model.omega_leeward = omega_leeward
    model.sigma_dep_gamma = None
    model.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})

    sim_in = types.SimpleNamespace(
        time={f: None},
        dust=types.SimpleNamespace(TSP={f: density}),
        dust_concentration={f: dust_conc},
        dust_type={f: "TSP"},
        wind_direction={f: wind_dir},
        wind_speed={f: wind_speed},
    )

    model.calculate_delta_soiled_area(sim_in, verbose=False)

    alpha = dust_conc / density
    theta = np.radians(tilt)
    expected_col0 = alpha[0] * (np.cos(theta[:, 0]) * mu_tilde + wind_speed[0] * np.sin(theta[:, 0]) * omega_windward)
    expected_col1 = alpha[1] * (np.cos(theta[:, 1]) * mu_tilde + wind_speed[1] * np.sin(theta[:, 1]) * omega_leeward)

    np.testing.assert_allclose(model.helios.delta_soiled_area[f][:, 0], expected_col0, rtol=RTOL)
    np.testing.assert_allclose(model.helios.delta_soiled_area[f][:, 1], expected_col1, rtol=RTOL)


def test_constant_mean_wind_delta_soiled_area_variance():
    f = 0
    density = 40.0
    tilt = np.array([[30.0, 60.0]])
    azimuth = np.array([[0.0, 0.0]])
    dust_conc = np.array([50.0, 70.0])
    wind_dir = np.array([0.0, 180.0])
    wind_speed = np.array([4.0, 5.0])
    sigma_dep = 0.02
    sigma_dep_gamma = 0.01

    model = ConstantMeanWindBase()
    model.mu_tilde = 0.5
    model.sigma_dep = sigma_dep
    model.omega_windward = 0.03
    model.omega_leeward = 0.01
    model.sigma_dep_gamma = sigma_dep_gamma
    model.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})

    sim_in = types.SimpleNamespace(
        time={f: None},
        dust=types.SimpleNamespace(TSP={f: density}),
        dust_concentration={f: dust_conc},
        dust_type={f: "TSP"},
        wind_direction={f: wind_dir},
        wind_speed={f: wind_speed},
    )

    model.calculate_delta_soiled_area(sim_in, verbose=False)

    alpha = dust_conc / density
    theta = np.radians(tilt)
    delta_gamma = np.radians(azimuth - wind_dir[None, :])
    p_w = np.sin(theta) * np.maximum(0.0, np.cos(delta_gamma))
    p_l = np.sin(theta) * np.maximum(0.0, -np.cos(delta_gamma))
    expected_var = alpha**2 * (sigma_dep**2 * np.cos(theta) ** 2 + sigma_dep_gamma**2 * wind_speed[None, :] ** 2 * (p_w + p_l) ** 2)
    np.testing.assert_allclose(model.helios.delta_soiled_area_variance[f], expected_var, rtol=RTOL, atol=ATOL)


def test_random_delta_soiled_area_matches_calculate_delta_soiled_area_mean():
    """ConstantMeanBase.random_delta_soiled_area calls calculate_delta_soiled_area
    positionally as (sim_in, mu_tilde, sigma_dep, verbose); on ConstantMeanWindBase
    that would silently misroute `verbose` into the omega_windward slot unless
    overridden. Check that the mean produced by random_delta_soiled_area matches a
    direct calculate_delta_soiled_area call (i.e. omega_windward/leeward survive)."""
    f = 0
    density = 40.0
    tilt = np.array([[30.0, 60.0]])
    azimuth = np.array([[0.0, 0.0]])
    dust_conc = np.array([50.0, 70.0])
    wind_dir = np.array([0.0, 180.0])
    wind_speed = np.array([4.0, 5.0])

    model = ConstantMeanWindBase()
    model.mu_tilde = 0.5
    model.sigma_dep = 0.02
    model.omega_windward = 0.03
    model.omega_leeward = 0.01
    model.sigma_dep_gamma = 0.01
    model.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})

    sim_in = types.SimpleNamespace(
        time={f: None},
        dust=types.SimpleNamespace(TSP={f: density}),
        dust_concentration={f: dust_conc},
        dust_type={f: "TSP"},
        wind_direction={f: wind_dir},
        wind_speed={f: wind_speed},
    )

    model.calculate_delta_soiled_area(sim_in, verbose=False)
    expected_mean = model.helios.delta_soiled_area[f].copy()

    np.random.seed(0)
    model.random_delta_soiled_area(sim_in, verbose=False)
    actual_mean = model.helios.delta_soiled_area[f]

    np.testing.assert_allclose(actual_mean, expected_mean, rtol=RTOL, atol=ATOL)


def test_calculate_delta_soiled_area_requires_azimuth():
    f = 0
    model = ConstantMeanWindBase()
    model.mu_tilde = 0.5
    model.sigma_dep = None
    model.omega_windward = 0.1
    model.omega_leeward = 0.05
    model.sigma_dep_gamma = None
    model.helios = _wind_helios_stub(tilt={f: np.array([[10.0]])}, azimuth={f: None})

    sim_in = types.SimpleNamespace(
        time={f: None},
        dust=types.SimpleNamespace(TSP={f: 40.0}),
        dust_concentration={f: np.array([50.0])},
        dust_type={f: "TSP"},
        wind_direction={f: np.array([0.0])},
        wind_speed={f: np.array([3.0])},
    )
    with pytest.raises(ValueError, match="azimuth"):
        model.calculate_delta_soiled_area(sim_in, verbose=False)


def test_calculate_delta_soiled_area_requires_wind_data():
    f = 0
    model = ConstantMeanWindBase()
    model.mu_tilde = 0.5
    model.sigma_dep = None
    model.omega_windward = 0.1
    model.omega_leeward = 0.05
    model.sigma_dep_gamma = None
    model.helios = _wind_helios_stub(tilt={f: np.array([[10.0]])}, azimuth={f: np.array([[0.0]])})

    sim_in = types.SimpleNamespace(
        time={f: None},
        dust=types.SimpleNamespace(TSP={f: 40.0}),
        dust_concentration={f: np.array([50.0])},
        dust_type={f: "TSP"},
        wind_direction={},
        wind_speed={},
    )
    with pytest.raises(ValueError, match="[Ww]ind"):
        model.calculate_delta_soiled_area(sim_in, verbose=False)


# ---------------------------------------------------------------------------
# 4. ConstantMeanWindDeposition.helios_angles: azimuth from mirror-name parsing
# ---------------------------------------------------------------------------


def test_helios_angles_populates_azimuth_from_mirror_names():
    f = 0
    N_times = 5

    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    model.helios = smb.Heliostats()
    model.mu_tilde = None
    model.sigma_dep = None
    model.omega_windward = None
    model.omega_leeward = None
    model.sigma_dep_gamma = None

    sim_in = types.SimpleNamespace(time={f: np.arange(N_times)}, files=[f])
    ref_dat = types.SimpleNamespace(
        files=[f],
        mirror_names={f: ["ON_M1_T00", "OE_M2_T30", "OS_M3_T45", "OW_M4_T60"]},
        tilts={f: np.array([[0.0], [30.0], [45.0], [60.0]])},
        reflectometer_acceptance_angle=[12.5e-3],
        reflectometer_incidence_angle=[15.0],
    )

    model.helios_angles(sim_in, ref_dat, verbose=False, second_surface=True)

    np.testing.assert_allclose(model.helios.azimuth[f][:, 0], [0.0, 90.0, 180.0, 270.0])
    np.testing.assert_allclose(model.helios.tilt[f][:, 0], [0.0, 30.0, 45.0, 60.0])
    assert model.helios.azimuth[f].shape == (4, N_times)


def test_helios_angles_explicit_orientations_override():
    f = 0
    N_times = 3

    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    model.helios = smb.Heliostats()
    model.mu_tilde = None
    model.sigma_dep = None
    model.omega_windward = None
    model.omega_leeward = None
    model.sigma_dep_gamma = None

    sim_in = types.SimpleNamespace(time={f: np.arange(N_times)}, files=[f])
    ref_dat = types.SimpleNamespace(
        files=[f],
        mirror_names={f: ["mirror_A", "mirror_B"]},  # not parseable, must be ignored
        tilts={f: np.array([[20.0], [40.0]])},
        reflectometer_acceptance_angle=[12.5e-3],
        reflectometer_incidence_angle=[15.0],
    )

    model.helios_angles(sim_in, ref_dat, verbose=False, orientations=[45.0, 135.0])

    np.testing.assert_allclose(model.helios.azimuth[f][:, 0], [45.0, 135.0])


# ---------------------------------------------------------------------------
# 5. End-to-end MLE fit-recovery smoke test
# ---------------------------------------------------------------------------


def test_fit_mle_recovers_parameters_on_synthetic_data():
    """Generates reflectance data from known parameters plus measurement noise,
    then checks that fit_mle recovers mu_tilde and omega_windward to the right
    order of magnitude. Loose tolerances: this is a smoke test for the fitting
    machinery (5-parameter update/transform/likelihood wiring), not a precise
    statistical recovery guarantee."""
    rng = np.random.default_rng(42)

    f = 0
    N_helios = 4
    N_times = 240  # 10 days, hourly
    azimuths = np.array([0.0, 90.0, 180.0, 270.0])  # N, E, S, W
    tilt_values = np.full(N_helios, 45.0)

    true_mu_tilde = 4.0e-4
    true_omega_windward = 6.0e-5
    true_omega_leeward = 2.0e-5
    density = 40.0
    nominal_reflectance = 0.95
    incidence_angle = 15.0
    inc_ref_factor = np.float64(2.0 / np.cos(np.radians(incidence_angle)))

    dust_conc = rng.uniform(20.0, 80.0, size=N_times)
    wind_dir = rng.uniform(0.0, 360.0, size=N_times)
    wind_speed = rng.uniform(1.0, 6.0, size=N_times)

    tilt = np.tile(tilt_values[:, None], (1, N_times))
    azimuth = np.tile(azimuths[:, None], (1, N_times))

    # --- generate ground truth ---
    truth = ConstantMeanWindBase()
    truth.mu_tilde = true_mu_tilde
    truth.sigma_dep = None
    truth.omega_windward = true_omega_windward
    truth.omega_leeward = true_omega_leeward
    truth.sigma_dep_gamma = None
    truth.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})

    sim_in = types.SimpleNamespace(
        time={f: np.arange(N_times)},
        files=[f],
        dust=types.SimpleNamespace(TSP={f: density}),
        dust_concentration={f: dust_conc},
        dust_type={f: "TSP"},
        wind_direction={f: wind_dir},
        wind_speed={f: wind_speed},
    )

    truth.calculate_delta_soiled_area(sim_in, verbose=False)
    cumulative = np.cumsum(truth.helios.delta_soiled_area[f], axis=1)
    soiling_factor_true = 1 - cumulative * inc_ref_factor  # clean start (rho0 = nominal)

    # --- synthesize daily reflectance measurements with i.i.d. noise ---
    meas_every = 24
    pred_idx = np.arange(meas_every - 1, N_times, meas_every)
    meas_noise_std = 0.0005
    rho_true_at_meas = nominal_reflectance * soiling_factor_true[:, pred_idx]
    average = (rho_true_at_meas + rng.normal(0.0, meas_noise_std, size=rho_true_at_meas.shape)).T
    sigma_of_the_mean = np.full(average.shape, meas_noise_std)

    ref_dat = types.SimpleNamespace(
        files=[f],
        times={f: pred_idx},
        prediction_indices={f: list(pred_idx)},
        average={f: average},
        sigma_of_the_mean={f: sigma_of_the_mean},
        rho0={f: np.full(N_helios, nominal_reflectance)},
    )

    # --- fit ---
    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    model.mu_tilde = None
    model.sigma_dep = None
    model.omega_windward = None
    model.omega_leeward = None
    model.sigma_dep_gamma = None
    model.helios = _wind_helios_stub(
        tilt={f: tilt},
        azimuth={f: azimuth},
        inc_ref_factor={f: inc_ref_factor},
        nominal_reflectance=nominal_reflectance,
        soiling_factor={},
        soiling_factor_prediction_variance={},
    )

    # Bypass the base class's scalar-warm-start (its bounds of [1, 1000] assume a
    # differently-scaled deposition parameter and are unsuitable for this
    # synthetic mu_tilde ~ 1e-4 scale); supply an order-of-magnitude x0 instead.
    x0 = np.array([1e-3, 1e-4, 1e-4, 1e-4, 1e-4])
    x_hat, x_cov = model.fit_mle(sim_in, ref_dat, verbose=False, x0=x0, transform_to_original_scale=True)

    mu_hat, omega_w_hat, omega_l_hat, sigma_dep_hat, sigma_gamma_hat = x_hat

    assert np.all(np.isfinite(x_hat))
    assert abs(mu_hat - true_mu_tilde) / true_mu_tilde < 0.4
    assert omega_w_hat > 0
    assert abs(omega_w_hat - true_omega_windward) / true_omega_windward < 0.6
