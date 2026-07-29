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
from heliosoil.horizontal_impaction import (
    ConstantMeanWindBase,
    ConstantMeanWindDeposition,
    wind_projection_factors,
    wind_tangential_factor,
    wind_retention_factors,
    parse_orientation_names,
)

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
    ConstantMeanWindBase.__init__(model)  # default components=[gravitational, normal_wind]
    model.helios = smb.Heliostats()

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
    ConstantMeanWindBase.__init__(model)  # default components=[gravitational, normal_wind]
    model.helios = smb.Heliostats()

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
    ConstantMeanWindBase.__init__(model)  # default components=[gravitational, normal_wind]
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


def test_fit_mle_recovers_parameters_with_reference_mirror_drift():
    """Same synthetic setup as test_fit_mle_recovers_parameters_on_synthetic_data,
    but the "true" data now includes a slow, deterministic reference-mirror-derived
    drift in the nominal reflectance (as heliosoil.base_models.ReflectanceMeasurements
    would produce from a real "_Ref" column). Confirms fit_mle still recovers the
    true wind/deposition parameters once that drift is threaded through via
    reflectance_data.nominal_reflectance/rho0_index (heliosoil.utilities.
    _nominal_reflectance_series/_nominal_reflectance_anchor), at the same loose
    tolerances as the no-drift smoke test above."""
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
    soiling_factor_true = 1 - cumulative * inc_ref_factor  # clean start (rho0 = nominal reflectance at t0)

    # Deterministic reference-mirror drift: a slow +/-1.5% sinusoidal wobble in the
    # instrument's nominal reading, shared identically across every mirror (matching
    # the physical model: one reflectometer, drifting the same way for every
    # measurement in a round).
    drift = 1.0 + 0.015 * np.sin(2 * np.pi * np.arange(N_times) / N_times)
    nominal_reflectance_full = nominal_reflectance * drift

    meas_every = 24
    pred_idx = np.arange(meas_every - 1, N_times, meas_every)
    meas_noise_std = 0.0005
    r0_at_meas = nominal_reflectance_full[None, pred_idx]  # (1, N_meas), broadcasts over mirrors
    rho_true_at_meas = r0_at_meas * soiling_factor_true[:, pred_idx]
    average = (rho_true_at_meas + rng.normal(0.0, meas_noise_std, size=rho_true_at_meas.shape)).T
    sigma_of_the_mean = np.full(average.shape, meas_noise_std)

    # nominal_reflectance_series: identical drift value across mirrors at each
    # measurement time (one shared reflectometer), matching how ReflectanceMeasurements
    # populates it from a real reference column.
    nominal_reflectance_series = np.tile(nominal_reflectance_full[pred_idx][:, None], (1, N_helios))
    rho0_index = np.zeros(N_helios, dtype=int)  # anchor at the first measurement, matching the clean start above

    ref_dat = types.SimpleNamespace(
        files=[f],
        times={f: pred_idx},
        prediction_indices={f: list(pred_idx)},
        average={f: average},
        sigma_of_the_mean={f: sigma_of_the_mean},
        rho0={f: nominal_reflectance_series[0, :].copy()},  # exact match at the anchor -> cumulative_soil0 = 0
        nominal_reflectance={f: nominal_reflectance_series},
        rho0_index={f: rho0_index},
    )

    # --- fit ---
    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanWindBase.__init__(model)  # default components=[gravitational, normal_wind]
    model.helios = _wind_helios_stub(
        tilt={f: tilt},
        azimuth={f: azimuth},
        inc_ref_factor={f: inc_ref_factor},
        nominal_reflectance=nominal_reflectance,
        soiling_factor={},
        soiling_factor_prediction_variance={},
    )

    x0 = np.array([1e-3, 1e-4, 1e-4, 1e-4, 1e-4])
    x_hat, x_cov = model.fit_mle(sim_in, ref_dat, verbose=False, x0=x0, transform_to_original_scale=True)

    mu_hat, omega_w_hat, omega_l_hat, sigma_dep_hat, sigma_gamma_hat = x_hat

    assert np.all(np.isfinite(x_hat))
    assert abs(mu_hat - true_mu_tilde) / true_mu_tilde < 0.4
    assert omega_w_hat > 0
    assert abs(omega_w_hat - true_omega_windward) / true_omega_windward < 0.6


# ---------------------------------------------------------------------------
# 6. wind_tangential_factor geometry
# ---------------------------------------------------------------------------


def test_wind_tangential_factor_limits():
    # tilt=0: mirror horizontal, all wind is tangential to the mirror plane.
    t0 = wind_tangential_factor(np.array([[0.0]]), np.array([[90.0]]), np.array([90.0]))
    np.testing.assert_allclose(t0, 1.0, rtol=RTOL)

    # tilt=90, Delta_gamma=0: wind blows straight into the mirror face -> no
    # tangential component.
    t1 = wind_tangential_factor(np.array([[90.0]]), np.array([[90.0]]), np.array([90.0]))
    np.testing.assert_allclose(t1, 0.0, atol=1e-12)

    # tilt=90, Delta_gamma=90: wind blows along the mirror's edge -> fully tangential.
    t2 = wind_tangential_factor(np.array([[90.0]]), np.array([[90.0]]), np.array([0.0]))
    np.testing.assert_allclose(t2, 1.0, rtol=RTOL)


def test_wind_tangential_factor_broadcasting_and_complementarity():
    tilt = np.full((3, 4), 30.0)
    azimuth = np.zeros((3, 4))
    wind_dir = np.array([0.0, 90.0, 180.0, 270.0])

    t = wind_tangential_factor(tilt, azimuth, wind_dir)
    assert t.shape == (3, 4)

    # Orthogonal decomposition: (p_windward+p_leeward)^2 + tangential^2 == 1.
    p_w, p_l = wind_projection_factors(tilt, azimuth, wind_dir)
    np.testing.assert_allclose((p_w + p_l) ** 2 + t**2, 1.0, rtol=RTOL)


# ---------------------------------------------------------------------------
# 7. wind_retention_factors geometry
# ---------------------------------------------------------------------------


def test_wind_retention_factors_directional_and_tilt_limits():
    azimuth = np.array([[90.0]])  # mirror faces East
    s45 = np.sin(np.radians(45.0)) * np.cos(np.radians(45.0))

    # tilt=45, wind from East: windward only, scaled by sin(45)*cos(45).
    pw, pl = wind_retention_factors(np.array([[45.0]]), azimuth, np.array([90.0]))
    np.testing.assert_allclose(pw, s45, rtol=RTOL)
    np.testing.assert_allclose(pl, 0.0, atol=1e-12)

    # tilt=45, wind from West: leeward only.
    pw2, pl2 = wind_retention_factors(np.array([[45.0]]), azimuth, np.array([270.0]))
    np.testing.assert_allclose(pw2, 0.0, atol=1e-12)
    np.testing.assert_allclose(pl2, s45, rtol=RTOL)

    # The defining feature: retention vanishes on BOTH faces at tilt=0 (flat, no
    # windward area) AND tilt=90 (vertical, nothing retained), even head-on.
    for t in (0.0, 90.0):
        pw3, pl3 = wind_retention_factors(np.array([[t]]), azimuth, np.array([90.0]))
        np.testing.assert_allclose(pw3, 0.0, atol=1e-12)
        np.testing.assert_allclose(pl3, 0.0, atol=1e-12)

    # Wind perpendicular to the mirror normal: no impaction on either face.
    pw4, pl4 = wind_retention_factors(np.array([[45.0]]), azimuth, np.array([0.0]))
    np.testing.assert_allclose(pw4, 0.0, atol=1e-12)
    np.testing.assert_allclose(pl4, 0.0, atol=1e-12)

    # Tilt past vertical (face pointing downward): cos(tilt) < 0 is clipped to 0.
    pw5, pl5 = wind_retention_factors(np.array([[120.0]]), azimuth, np.array([90.0]))
    np.testing.assert_allclose(pw5, 0.0, atol=1e-12)
    np.testing.assert_allclose(pl5, 0.0, atol=1e-12)


def test_wind_retention_factors_equal_cos_tilt_times_normal():
    """For tilt in [0, 90], the retention factors are exactly cos(tilt) times the
    normal-impaction factors -- the mechanism only reshapes the tilt-dependence
    (sin -> sin*cos), keeping the same windward/leeward directional split."""
    tilt = np.full((3, 4), 35.0)
    azimuth = np.zeros((3, 4))
    wind_dir = np.array([0.0, 90.0, 180.0, 270.0])

    pw_ret, pl_ret = wind_retention_factors(tilt, azimuth, wind_dir)
    pw_n, pl_n = wind_projection_factors(tilt, azimuth, wind_dir)
    assert pw_ret.shape == (3, 4)
    c = np.cos(np.radians(35.0))
    np.testing.assert_allclose(pw_ret, c * pw_n, rtol=RTOL)
    np.testing.assert_allclose(pl_ret, c * pl_n, rtol=RTOL)


# ---------------------------------------------------------------------------
# 8. Modular components: gravitational / normal_wind / tangential_wind /
#    impaction_retention
# ---------------------------------------------------------------------------


def test_parameter_names_and_model_name_for_combinations():
    m1 = ConstantMeanWindBase(components=["gravitational", "normal_wind"])
    assert m1.model_name == "constant-mean_gravitational_normal-wind"
    assert m1.parameter_names == ["mu_tilde", "omega_windward", "omega_leeward", "sigma_dep", "sigma_dep_gamma"]

    m2 = ConstantMeanWindBase()  # default must match m1 exactly
    assert m2.model_name == m1.model_name
    assert m2.parameter_names == m1.parameter_names

    m3 = ConstantMeanWindBase(components=["gravitational", "tangential_wind"])
    assert m3.model_name == "constant-mean_gravitational_tangential-wind"
    assert m3.parameter_names == ["mu_tilde", "omega_tangential", "sigma_dep", "sigma_dep_tan"]

    # Input order must not matter: canonical order is enforced.
    m4 = ConstantMeanWindBase(components=["tangential_wind", "gravitational", "normal_wind"])
    assert m4.model_name == "constant-mean_gravitational_normal-wind_tangential-wind"
    assert m4.parameter_names == ["mu_tilde", "omega_windward", "omega_leeward", "omega_tangential", "sigma_dep", "sigma_dep_gamma", "sigma_dep_tan"]

    m5 = ConstantMeanWindBase(components=["normal_wind"])
    assert m5.model_name == "constant-mean_normal-wind"
    assert m5.parameter_names == ["omega_windward", "omega_leeward", "sigma_dep_gamma"]

    m6 = ConstantMeanWindBase(components=["gravitational", "impaction_retention"])
    assert m6.model_name == "constant-mean_gravitational_impaction-retention"
    assert m6.parameter_names == ["mu_tilde", "omega_ret_windward", "omega_ret_leeward", "sigma_dep", "sigma_dep_ret"]

    # All four components, passed out of order: canonical order + full param layout.
    m7 = ConstantMeanWindBase(components=["impaction_retention", "tangential_wind", "gravitational", "normal_wind"])
    assert m7.model_name == ("constant-mean_gravitational_normal-wind_tangential-wind_impaction-retention")
    assert m7.parameter_names == [
        "mu_tilde",
        "omega_windward",
        "omega_leeward",
        "omega_tangential",
        "omega_ret_windward",
        "omega_ret_leeward",
        "sigma_dep",
        "sigma_dep_gamma",
        "sigma_dep_tan",
        "sigma_dep_ret",
    ]


def test_unknown_component_raises():
    with pytest.raises(ValueError, match="Unknown wind component"):
        ConstantMeanWindBase(components=["gravitational", "not_a_real_component"])


def test_transform_scale_respects_component_log_mask():
    """mu_tilde and both sigmas are log-transformed; omega_tangential (possibly
    negative -- scouring) stays linear. Verified through the public transform_scale
    API rather than poking at the private mask directly."""
    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanWindBase.__init__(model, components=["gravitational", "tangential_wind"])

    x = np.array([0.01, -0.005, 0.02, 0.01])  # [mu_tilde, omega_tangential, sigma_dep, sigma_dep_tan]
    y = model.transform_scale(x, direction="forward")
    np.testing.assert_allclose(y[1], x[1])  # omega_tangential: unchanged (linear)
    np.testing.assert_allclose(y[0], np.log(x[0]))
    np.testing.assert_allclose(y[2], np.log(x[2]))
    np.testing.assert_allclose(y[3], np.log(x[3]))

    x_back = model.transform_scale(y, direction="inverse")
    np.testing.assert_allclose(x_back, x, rtol=RTOL)


def test_gravitational_only_reproduces_constant_mean_base_and_skips_wind_requirement():
    """components=["gravitational"] must reproduce ConstantMeanBase's output exactly
    and must NOT require azimuth or wind data (no helios.azimuth, no
    wind_direction/wind_speed on simulation_inputs at all)."""
    f = 0
    mu_tilde = 0.6
    density = 40.0
    tilt = np.array([[10.0, 45.0, 70.0]])
    dust_conc = np.array([30.0, 55.0, 90.0])

    model = ConstantMeanWindBase(components=["gravitational"])
    model.mu_tilde = mu_tilde
    model.sigma_dep = None
    model.helios = _wind_helios_stub(tilt={f: tilt})  # deliberately no azimuth

    sim_in = types.SimpleNamespace(
        time={f: None},
        dust=types.SimpleNamespace(TSP={f: density}),
        dust_concentration={f: dust_conc},
        dust_type={f: "TSP"},
        # deliberately no wind_direction/wind_speed attributes
    )

    model.calculate_delta_soiled_area(sim_in, verbose=False)  # must not raise

    alpha = dust_conc / density
    expected = alpha[None, :] * np.cos(np.radians(tilt)) * mu_tilde
    np.testing.assert_allclose(model.helios.delta_soiled_area[f], expected, rtol=RTOL, atol=ATOL)


def test_tangential_wind_delta_soiled_area_mean_and_variance():
    f = 0
    density = 40.0
    tilt = np.array([[40.0, 65.0]])
    azimuth = np.array([[90.0, 90.0]])
    dust_conc = np.array([45.0, 55.0])
    wind_dir = np.array([30.0, 200.0])
    wind_speed = np.array([3.5, 5.5])
    omega_tan = 0.02
    sigma_tan = 0.015

    model = ConstantMeanWindBase(components=["gravitational", "tangential_wind"])
    model.mu_tilde = 0.0  # isolate the tangential contribution
    model.sigma_dep = None
    model.omega_tangential = omega_tan
    model.sigma_dep_tan = sigma_tan
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
    t = np.sqrt(1 - (np.sin(theta) * np.cos(delta_gamma)) ** 2)
    expected_mean = alpha[None, :] * wind_speed[None, :] * t * omega_tan
    expected_var = (alpha[None, :] * wind_speed[None, :] * t) ** 2 * sigma_tan**2

    np.testing.assert_allclose(model.helios.delta_soiled_area[f], expected_mean, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(model.helios.delta_soiled_area_variance[f], expected_var, rtol=RTOL, atol=ATOL)


def test_impaction_retention_delta_soiled_area_mean_and_variance():
    """Independent windward/leeward coefficients, with the sin(tilt)*cos(tilt)
    retention weighting and a single shared noise term (sigma_dep_ret)."""
    f = 0
    density = 40.0
    tilt = np.array([[40.0, 65.0]])
    azimuth = np.array([[90.0, 90.0]])  # both mirrors face East
    dust_conc = np.array([45.0, 55.0])
    wind_dir = np.array([90.0, 270.0])  # col0: from East (windward); col1: from West (leeward)
    wind_speed = np.array([3.5, 5.5])
    omega_ret_windward = 0.03
    omega_ret_leeward = 0.012
    sigma_dep_ret = 0.02

    model = ConstantMeanWindBase(components=["gravitational", "impaction_retention"])
    model.mu_tilde = 0.0  # isolate the retention contribution
    model.sigma_dep = None
    model.omega_ret_windward = omega_ret_windward
    model.omega_ret_leeward = omega_ret_leeward
    model.sigma_dep_ret = sigma_dep_ret
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
    s = np.sin(theta) * np.maximum(0.0, np.cos(theta))
    p_w = s * np.maximum(0.0, np.cos(delta_gamma))
    p_l = s * np.maximum(0.0, -np.cos(delta_gamma))
    expected_mean = alpha[None, :] * wind_speed[None, :] * (p_w * omega_ret_windward + p_l * omega_ret_leeward)
    expected_var = (alpha[None, :] * wind_speed[None, :]) ** 2 * (p_w + p_l) ** 2 * sigma_dep_ret**2

    np.testing.assert_allclose(model.helios.delta_soiled_area[f], expected_mean, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(model.helios.delta_soiled_area_variance[f], expected_var, rtol=RTOL, atol=ATOL)


def test_normal_and_tangential_variance_bases_are_orthogonal():
    """With sigma=1 for each (in isolation), normal_wind's variance + tangential_wind's
    variance == alpha^2 * U^2: wind speed decomposes orthogonally into components
    normal and tangential to the mirror plane."""
    f = 0
    density = 40.0
    tilt = np.array([[20.0, 50.0, 80.0]])
    azimuth = np.array([[45.0, 45.0, 45.0]])
    dust_conc = np.array([30.0, 60.0, 90.0])
    wind_dir = np.array([10.0, 200.0, 300.0])
    wind_speed = np.array([2.0, 4.5, 6.0])

    model_normal = ConstantMeanWindBase(components=["normal_wind"])
    model_normal.omega_windward = 0.0
    model_normal.omega_leeward = 0.0
    model_normal.sigma_dep_gamma = 1.0
    model_normal.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})

    model_tangential = ConstantMeanWindBase(components=["tangential_wind"])
    model_tangential.omega_tangential = 0.0
    model_tangential.sigma_dep_tan = 1.0
    model_tangential.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})

    sim_in = types.SimpleNamespace(
        time={f: None},
        dust=types.SimpleNamespace(TSP={f: density}),
        dust_concentration={f: dust_conc},
        dust_type={f: "TSP"},
        wind_direction={f: wind_dir},
        wind_speed={f: wind_speed},
    )

    model_normal.calculate_delta_soiled_area(sim_in, verbose=False)
    model_tangential.calculate_delta_soiled_area(sim_in, verbose=False)

    alpha = dust_conc / density
    expected_total = (alpha[None, :] * wind_speed[None, :]) ** 2
    total = model_normal.helios.delta_soiled_area_variance[f] + model_tangential.helios.delta_soiled_area_variance[f]
    np.testing.assert_allclose(total, expected_total, rtol=RTOL, atol=ATOL)


def test_fit_mle_recovers_parameters_gravitational_tangential():
    """Smoke test for a non-default component set: fit_mle converges and recovers
    mu_tilde/omega_tangential to the right order of magnitude. Loose tolerances,
    matching the gravitational+normal_wind smoke test above."""
    rng = np.random.default_rng(7)

    f = 0
    N_helios = 4
    N_times = 240
    azimuths = np.array([0.0, 90.0, 180.0, 270.0])
    tilt_values = np.full(N_helios, 45.0)

    true_mu_tilde = 4.0e-4
    true_omega_tangential = 5.0e-5
    density = 40.0
    nominal_reflectance = 0.95
    incidence_angle = 15.0
    inc_ref_factor = np.float64(2.0 / np.cos(np.radians(incidence_angle)))

    dust_conc = rng.uniform(20.0, 80.0, size=N_times)
    wind_dir = rng.uniform(0.0, 360.0, size=N_times)
    wind_speed = rng.uniform(1.0, 6.0, size=N_times)

    tilt = np.tile(tilt_values[:, None], (1, N_times))
    azimuth = np.tile(azimuths[:, None], (1, N_times))

    truth = ConstantMeanWindBase(components=["gravitational", "tangential_wind"])
    truth.mu_tilde = true_mu_tilde
    truth.sigma_dep = None
    truth.omega_tangential = true_omega_tangential
    truth.sigma_dep_tan = None
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
    soiling_factor_true = 1 - cumulative * inc_ref_factor

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

    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanWindBase.__init__(model, components=["gravitational", "tangential_wind"])
    model.helios = _wind_helios_stub(
        tilt={f: tilt},
        azimuth={f: azimuth},
        inc_ref_factor={f: inc_ref_factor},
        nominal_reflectance=nominal_reflectance,
        soiling_factor={},
        soiling_factor_prediction_variance={},
    )

    x0 = np.array([1e-3, 1e-4, 1e-4, 1e-4])  # [mu_tilde, omega_tangential, sigma_dep, sigma_dep_tan]
    x_hat, x_cov = model.fit_mle(sim_in, ref_dat, verbose=False, x0=x0, transform_to_original_scale=True)

    mu_hat, omega_tan_hat, sigma_dep_hat, sigma_tan_hat = x_hat

    assert np.all(np.isfinite(x_hat))
    assert abs(mu_hat - true_mu_tilde) / true_mu_tilde < 0.4
    assert omega_tan_hat > 0
    assert abs(omega_tan_hat - true_omega_tangential) / true_omega_tangential < 0.6


def test_fit_mle_recovers_parameters_gravitational_impaction_retention():
    """Smoke test for the impaction_retention component: fit_mle converges and
    recovers mu_tilde/omega_ret_windward to the right order of magnitude. Loose
    tolerances, matching the gravitational+normal_wind smoke test above."""
    rng = np.random.default_rng(11)

    f = 0
    N_helios = 4
    N_times = 240
    azimuths = np.array([0.0, 90.0, 180.0, 270.0])
    tilt_values = np.full(N_helios, 45.0)

    true_mu_tilde = 4.0e-4
    true_omega_ret_windward = 6.0e-5
    true_omega_ret_leeward = 2.0e-5
    density = 40.0
    nominal_reflectance = 0.95
    incidence_angle = 15.0
    inc_ref_factor = np.float64(2.0 / np.cos(np.radians(incidence_angle)))

    dust_conc = rng.uniform(20.0, 80.0, size=N_times)
    wind_dir = rng.uniform(0.0, 360.0, size=N_times)
    wind_speed = rng.uniform(1.0, 6.0, size=N_times)

    tilt = np.tile(tilt_values[:, None], (1, N_times))
    azimuth = np.tile(azimuths[:, None], (1, N_times))

    truth = ConstantMeanWindBase(components=["gravitational", "impaction_retention"])
    truth.mu_tilde = true_mu_tilde
    truth.sigma_dep = None
    truth.omega_ret_windward = true_omega_ret_windward
    truth.omega_ret_leeward = true_omega_ret_leeward
    truth.sigma_dep_ret = None
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
    soiling_factor_true = 1 - cumulative * inc_ref_factor

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

    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanWindBase.__init__(model, components=["gravitational", "impaction_retention"])
    model.helios = _wind_helios_stub(
        tilt={f: tilt},
        azimuth={f: azimuth},
        inc_ref_factor={f: inc_ref_factor},
        nominal_reflectance=nominal_reflectance,
        soiling_factor={},
        soiling_factor_prediction_variance={},
    )

    x0 = np.array([1e-3, 1e-4, 1e-4, 1e-4, 1e-4])
    x_hat, x_cov = model.fit_mle(sim_in, ref_dat, verbose=False, x0=x0, transform_to_original_scale=True)

    mu_hat, omega_ret_w_hat, omega_ret_l_hat, sigma_dep_hat, sigma_ret_hat = x_hat

    assert np.all(np.isfinite(x_hat))
    assert abs(mu_hat - true_mu_tilde) / true_mu_tilde < 0.4
    assert omega_ret_w_hat > 0
    assert abs(omega_ret_w_hat - true_omega_ret_windward) / true_omega_ret_windward < 0.6


# ---------------------------------------------------------------------------
# 9. Per-component dust channels
#    Each mechanism can be driven by its own dust spec -- a single measure or the
#    difference of two, isolating one size band -- instead of all of them sharing
#    the simulation's dust_type.
# ---------------------------------------------------------------------------


def _dust_with_distribution():
    """A Dust carrying only a size distribution: enough for pm_mass to integrate the
    reference mass of any spec. Log-normal in log10(D), positive over the whole grid."""
    dust = smb.Dust()
    D = np.logspace(-2, 2, 400)  # 0.01 - 100 µm
    dust.D = {0: D}
    dust.pdfM = {0: np.exp(-((np.log10(D) - 0.3) ** 2) / 0.5)}
    return dust


def _mass_below(dust, cutoff):
    D = dust.D[0]
    return np.trapezoid(dust.pdfM[0][D <= cutoff], np.log10(D[D <= cutoff]))


def _channel_sim_in(dust, channels, wind_dir, wind_speed, f=0):
    """simulation_inputs stub carrying per-measure concentration channels. dust_type/
    dust_concentration are deliberately absent: a fully assigned model must not touch
    them."""
    return types.SimpleNamespace(
        time={f: None}, dust=dust, dust_concentration_channels={f: channels}, wind_direction={f: wind_dir}, wind_speed={f: wind_speed}
    )


def test_component_dust_types_naming_and_expression():
    model = ConstantMeanWindBase(
        components=["tangential_wind", "gravitational"], component_dust_types={"gravitational": "PM17", "tangential_wind": "PM10-PM2.5"}
    )
    # Canonical component order, canonical dust names, and a folder-safe label.
    assert model.model_name == "constant-mean_PM17xgravitational_PM10-PM2.5xtangential-wind"
    assert model.component_expression == "PM17*gravitational + (PM10-PM2.5)*tangential_wind"
    # Dust channels change what drives each mechanism, never the parameter vector.
    assert model.parameter_names == ConstantMeanWindBase(components=["gravitational", "tangential_wind"]).parameter_names


def test_component_dust_types_partial_and_default():
    # Unassigned components fall back to the simulation's own dust_type and are left
    # unqualified in both the name and the expression.
    model = ConstantMeanWindBase(components=["gravitational", "normal_wind"], component_dust_types={"gravitational": "PM17"})
    assert model.model_name == "constant-mean_PM17xgravitational_normal-wind"
    assert model.component_expression == "PM17*gravitational + normal_wind"

    # A single spec applies to every component.
    shared = ConstantMeanWindBase(components=["gravitational", "normal_wind"], component_dust_types="PM10")
    assert shared.component_expression == "PM10*gravitational + PM10*normal_wind"

    # The default is unchanged from before per-component channels existed.
    default = ConstantMeanWindBase(components=["gravitational", "normal_wind"])
    assert default.model_name == "constant-mean_gravitational_normal-wind"
    assert default.component_expression == "gravitational + normal_wind"


def test_component_dust_types_rejects_inactive_component_and_bad_spec():
    with pytest.raises(ValueError, match="which this model does not have"):
        ConstantMeanWindBase(components=["gravitational"], component_dust_types={"tangential_wind": "PM10"})
    with pytest.raises(ValueError, match="Unrecognized dust measure"):
        ConstantMeanWindBase(components=["gravitational"], component_dust_types={"gravitational": "humidity"})


def test_per_component_dust_channels_drive_each_mechanism():
    """gravitational on PM17 (mass up to 17 µm), tangential_wind on the 2.5-10 µm band:
    each term must use its own alpha, i.e. its own concentration over its own
    reference mass."""
    f = 0
    mu_tilde = 0.5
    omega_tan = 0.02
    tilt = np.array([[35.0, 55.0]])
    azimuth = np.array([[90.0, 90.0]])
    wind_dir = np.array([120.0, 250.0])
    wind_speed = np.array([3.0, 5.0])
    channels = {"PM2.5": np.array([8.0, 11.0]), "PM10": np.array([25.0, 30.0]), "PM17": np.array([60.0, 90.0])}

    dust = _dust_with_distribution()
    model = ConstantMeanWindBase(
        components=["gravitational", "tangential_wind"], component_dust_types={"gravitational": "PM17", "tangential_wind": "PM10-PM2.5"}
    )
    model.mu_tilde = mu_tilde
    model.sigma_dep = None
    model.omega_tangential = omega_tan
    model.sigma_dep_tan = None
    model.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})

    model.calculate_delta_soiled_area(_channel_sim_in(dust, channels, wind_dir, wind_speed), verbose=False)

    alpha_grav = channels["PM17"] / _mass_below(dust, 17.0)
    alpha_tan = (channels["PM10"] - channels["PM2.5"]) / (_mass_below(dust, 10.0) - _mass_below(dust, 2.5))
    theta = np.radians(tilt)
    delta_gamma = np.radians(azimuth - wind_dir[None, :])
    t = np.sqrt(1 - (np.sin(theta) * np.cos(delta_gamma)) ** 2)
    expected = alpha_grav[None, :] * np.cos(theta) * mu_tilde + alpha_tan[None, :] * wind_speed[None, :] * t * omega_tan

    np.testing.assert_allclose(model.helios.delta_soiled_area[f], expected, rtol=RTOL, atol=ATOL)
    # The two channels must not be interchangeable: PM17 is not the 2.5-10 µm band.
    assert not np.allclose(alpha_grav, alpha_tan)


def test_per_component_dust_channels_drive_the_variance_too():
    f = 0
    sigma_dep = 0.03
    sigma_tan = 0.02
    tilt = np.array([[35.0, 55.0]])
    azimuth = np.array([[90.0, 90.0]])
    wind_dir = np.array([120.0, 250.0])
    wind_speed = np.array([3.0, 5.0])
    channels = {"PM2.5": np.array([8.0, 11.0]), "PM10": np.array([25.0, 30.0]), "PM17": np.array([60.0, 90.0])}

    dust = _dust_with_distribution()
    model = ConstantMeanWindBase(
        components=["gravitational", "tangential_wind"], component_dust_types={"gravitational": "PM17", "tangential_wind": "PM10-PM2.5"}
    )
    model.mu_tilde = 0.5
    model.sigma_dep = sigma_dep
    model.omega_tangential = 0.02
    model.sigma_dep_tan = sigma_tan
    model.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})

    model.calculate_delta_soiled_area(_channel_sim_in(dust, channels, wind_dir, wind_speed), verbose=False)

    alpha_grav = channels["PM17"] / _mass_below(dust, 17.0)
    alpha_tan = (channels["PM10"] - channels["PM2.5"]) / (_mass_below(dust, 10.0) - _mass_below(dust, 2.5))
    theta = np.radians(tilt)
    delta_gamma = np.radians(azimuth - wind_dir[None, :])
    t = np.sqrt(1 - (np.sin(theta) * np.cos(delta_gamma)) ** 2)
    expected = (sigma_dep * alpha_grav[None, :] * np.cos(theta)) ** 2 + (sigma_tan * alpha_tan[None, :] * wind_speed[None, :] * t) ** 2

    np.testing.assert_allclose(model.helios.delta_soiled_area_variance[f], expected, rtol=RTOL, atol=ATOL)


def test_component_dust_channel_missing_from_simulation_raises():
    f = 0
    model = ConstantMeanWindBase(components=["gravitational"], component_dust_types={"gravitational": "PM17-PM10"})
    model.mu_tilde = 0.5
    model.sigma_dep = None
    model.helios = _wind_helios_stub(tilt={f: np.array([[30.0]])})

    sim_in = _channel_sim_in(_dust_with_distribution(), {"PM10": np.array([25.0])}, np.array([0.0]), np.array([3.0]))
    with pytest.raises(ValueError, match="not available for file"):
        model.calculate_delta_soiled_area(sim_in, verbose=False)


def test_shared_dust_spec_reproduces_the_simulation_dust_type_path():
    """Assigning every component the same spec as the simulation's dust_type must give
    exactly the single-channel result -- the two alpha routes agree."""
    f = 0
    tilt = np.array([[20.0, 60.0]])
    azimuth = np.array([[0.0, 0.0]])
    wind_dir = np.array([0.0, 180.0])
    wind_speed = np.array([2.0, 4.0])
    concentration = np.array([40.0, 70.0])

    dust = _dust_with_distribution()
    dust.PM10 = {f: _mass_below(dust, 10.0)}

    def run(component_dust_types, sim_in):
        model = ConstantMeanWindBase(components=["gravitational", "normal_wind"], component_dust_types=component_dust_types)
        model.mu_tilde, model.sigma_dep = 0.4, 0.02
        model.omega_windward, model.omega_leeward, model.sigma_dep_gamma = 0.03, 0.01, 0.015
        model.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})
        model.calculate_delta_soiled_area(sim_in, verbose=False)
        return model.helios.delta_soiled_area[f], model.helios.delta_soiled_area_variance[f]

    via_dust_type = types.SimpleNamespace(
        time={f: None},
        dust=dust,
        dust_concentration={f: concentration},
        dust_type={f: "PM10"},
        wind_direction={f: wind_dir},
        wind_speed={f: wind_speed},
    )
    via_channels = _channel_sim_in(dust, {"PM10": concentration}, wind_dir, wind_speed)

    mean_ref, var_ref = run(None, via_dust_type)
    mean_channels, var_channels = run("PM10", via_channels)

    np.testing.assert_allclose(mean_channels, mean_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(var_channels, var_ref, rtol=RTOL, atol=ATOL)


def test_fit_mle_recovers_parameters_with_per_component_dust_channels():
    """End-to-end smoke test of the fitting path with two different dust channels: the
    likelihood's variance term reads its own alpha per mechanism (see
    _compute_variance_of_measurements), so a mismatch there would bias the estimates.
    Loose tolerances, matching the other fit-recovery smoke tests."""
    rng = np.random.default_rng(13)

    f = 0
    N_helios = 4
    N_times = 240
    azimuths = np.array([0.0, 90.0, 180.0, 270.0])
    tilt_values = np.full(N_helios, 45.0)

    true_mu_tilde = 4.0e-4
    true_omega_tangential = 5.0e-5
    nominal_reflectance = 0.95
    incidence_angle = 15.0
    inc_ref_factor = np.float64(2.0 / np.cos(np.radians(incidence_angle)))

    # Coarse and fine channels vary independently, so the two alphas are not
    # proportional and the fit has to attribute the right variation to each mechanism.
    fine = rng.uniform(5.0, 25.0, size=N_times)
    channels = {"PM2.5": fine, "PM10": fine + rng.uniform(10.0, 40.0, size=N_times), "PM17": rng.uniform(60.0, 140.0, size=N_times)}
    wind_dir = rng.uniform(0.0, 360.0, size=N_times)
    wind_speed = rng.uniform(1.0, 6.0, size=N_times)

    tilt = np.tile(tilt_values[:, None], (1, N_times))
    azimuth = np.tile(azimuths[:, None], (1, N_times))
    component_dust_types = {"gravitational": "PM17", "tangential_wind": "PM10-PM2.5"}

    sim_in = _channel_sim_in(_dust_with_distribution(), channels, wind_dir, wind_speed)
    sim_in.time = {f: np.arange(N_times)}
    sim_in.files = [f]

    truth = ConstantMeanWindBase(components=["gravitational", "tangential_wind"], component_dust_types=component_dust_types)
    truth.mu_tilde = true_mu_tilde
    truth.sigma_dep = None
    truth.omega_tangential = true_omega_tangential
    truth.sigma_dep_tan = None
    truth.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})

    truth.calculate_delta_soiled_area(sim_in, verbose=False)
    soiling_factor_true = 1 - np.cumsum(truth.helios.delta_soiled_area[f], axis=1) * inc_ref_factor

    meas_every = 24
    pred_idx = np.arange(meas_every - 1, N_times, meas_every)
    meas_noise_std = 0.0005
    rho_true_at_meas = nominal_reflectance * soiling_factor_true[:, pred_idx]
    average = (rho_true_at_meas + rng.normal(0.0, meas_noise_std, size=rho_true_at_meas.shape)).T

    ref_dat = types.SimpleNamespace(
        files=[f],
        times={f: pred_idx},
        prediction_indices={f: list(pred_idx)},
        average={f: average},
        sigma_of_the_mean={f: np.full(average.shape, meas_noise_std)},
        rho0={f: np.full(N_helios, nominal_reflectance)},
    )

    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanWindBase.__init__(model, components=["gravitational", "tangential_wind"], component_dust_types=component_dust_types)
    model.helios = _wind_helios_stub(
        tilt={f: tilt},
        azimuth={f: azimuth},
        inc_ref_factor={f: inc_ref_factor},
        nominal_reflectance=nominal_reflectance,
        soiling_factor={},
        soiling_factor_prediction_variance={},
    )

    x0 = np.array([1e-3, 1e-4, 1e-4, 1e-4])  # [mu_tilde, omega_tangential, sigma_dep, sigma_dep_tan]
    x_hat, _x_cov = model.fit_mle(sim_in, ref_dat, verbose=False, x0=x0, transform_to_original_scale=True)
    mu_hat, omega_tan_hat, _sigma_dep_hat, _sigma_tan_hat = x_hat

    assert np.all(np.isfinite(x_hat))
    assert abs(mu_hat - true_mu_tilde) / true_mu_tilde < 0.4
    assert omega_tan_hat > 0
    assert abs(omega_tan_hat - true_omega_tangential) / true_omega_tangential < 0.6
