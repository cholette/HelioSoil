"""
Unit / characterization tests for the horizontal wind-driven impaction model
(heliosoil.horizontal_impaction).

Mirrors the style of tests/test_model_components.py: small synthetic inputs,
independently re-derived reference values, tight tolerances for deterministic
formulas, and a looser end-to-end smoke test for the stochastic MLE fit.
"""

import logging
import types
import numpy as np
import pytest

import heliosoil.base_models as smb
import heliosoil.fitting as smf
from heliosoil.horizontal_impaction import (
    ConstantMeanWindBase,
    ConstantMeanWindDeposition,
    wind_projection_factors,
    wind_tangential_factor,
    wind_retention_factors,
    parse_orientation_names,
    _gravitational_mean_bases,
    _turbulant_wind_mean_bases,
    _COMPONENTS,
)

# variance_basis is derived from noise_basis on the component descriptor rather than
# existing as a free function, so these exercise the same path production code takes.
_gravitational_variance_basis = _COMPONENTS["gravitational"].variance_basis
_turbulant_wind_variance_basis = _COMPONENTS["turbulent_wind"].variance_basis

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
    # 180 deg is a real mirror (yadnarie's OSW_M1_T180), where an unclipped cos(tilt) = -1
    # would make the model predict the mirror CLEANING itself.
    tilt = np.array([[10.0, 45.0, 70.0, 180.0]])
    azimuth = np.array([[0.0, 0.0, 0.0, 0.0]])
    dust_conc = np.array([30.0, 55.0, 90.0, 70.0])
    wind_dir = np.array([0.0, 90.0, 180.0, 45.0])
    wind_speed = np.array([3.0, 4.0, 5.0, 3.5])

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
    expected = alpha[None, :] * np.maximum(0.0, np.cos(np.radians(tilt))) * mu_tilde
    np.testing.assert_allclose(model.helios.delta_soiled_area[f], expected, rtol=RTOL, atol=ATOL)

    # The same numbers out of ConstantMeanBase itself, so the two implementations of the
    # settling term cannot drift apart at past-vertical tilts.
    plain = smb.ConstantMeanBase()
    plain.mu_tilde = mu_tilde
    plain.sigma_dep = None
    plain.helios = _wind_helios_stub(tilt={f: tilt})
    plain.calculate_delta_soiled_area(sim_in, verbose=False)
    np.testing.assert_allclose(model.helios.delta_soiled_area[f], plain.helios.delta_soiled_area[f], rtol=RTOL, atol=ATOL)

    # And the face-down mirror specifically: zero, not negative.
    assert model.helios.delta_soiled_area[f][0, -1] == 0.0


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
    tilt = np.array([[10.0, 45.0, 70.0, 180.0]])
    dust_conc = np.array([30.0, 55.0, 90.0, 70.0])

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
    expected = alpha[None, :] * np.maximum(0.0, np.cos(np.radians(tilt))) * mu_tilde
    np.testing.assert_allclose(model.helios.delta_soiled_area[f], expected, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Past-vertical (face-down) mirrors: the settling factor must be clipped, not signed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tilt_deg", [90.1, 120.0, 180.0, 265.0])
def test_settling_driven_bases_vanish_past_vertical(tilt_deg):
    """A mirror tilted past vertical faces the ground and collects no settled dust.

    With a bare cos(tilt) these bases went NEGATIVE past 90 deg, so the model predicted the
    mirror gaining reflectance without bound -- the failure seen on yadnarie's OSW_M1_T180.
    Both the mean and its variance must vanish together.
    """
    alpha = np.array([1.5, 2.0, 2.5])
    wind_speed = np.array([3.0, 4.0, 5.0])
    tilt = np.full((1, 3), tilt_deg)

    assert _gravitational_mean_bases(alpha, tilt, None, None, wind_speed)[0].tolist() == [[0.0, 0.0, 0.0]]
    assert _gravitational_variance_basis(alpha, tilt, None, None, wind_speed).tolist() == [[0.0, 0.0, 0.0]]
    assert _turbulant_wind_mean_bases(alpha, tilt, None, None, wind_speed)[0].tolist() == [[0.0, 0.0, 0.0]]
    assert _turbulant_wind_variance_basis(alpha, tilt, None, None, wind_speed).tolist() == [[0.0, 0.0, 0.0]]


def test_settling_driven_bases_negligible_at_exactly_vertical():
    """At exactly 90 deg cosd returns +6e-17 rather than 0, so the clip passes that float
    through untouched. It is positive (never the inverted sign this fix is about) and
    negligible, matching how the rest of the suite treats the tilt=90 limit."""
    alpha = np.array([1.5, 2.0, 2.5])
    wind_speed = np.array([3.0, 4.0, 5.0])
    tilt = np.full((1, 3), 90.0)

    mean = _gravitational_mean_bases(alpha, tilt, None, None, wind_speed)[0]
    assert (mean >= 0.0).all()
    np.testing.assert_allclose(mean, 0.0, atol=1e-12)
    np.testing.assert_allclose(_turbulant_wind_mean_bases(alpha, tilt, None, None, wind_speed)[0], 0.0, atol=1e-12)


@pytest.mark.parametrize("tilt_deg", [0.0, 30.0, 89.0])
def test_settling_driven_bases_unchanged_below_vertical(tilt_deg):
    """The clip must not perturb the upward-facing range the model was fitted on."""
    alpha = np.array([1.5, 2.0, 2.5])
    wind_speed = np.array([3.0, 4.0, 5.0])
    tilt = np.full((1, 3), tilt_deg)
    c = np.cos(np.radians(tilt_deg))

    np.testing.assert_allclose(_gravitational_mean_bases(alpha, tilt, None, None, wind_speed)[0], alpha[None, :] * c, rtol=RTOL)
    np.testing.assert_allclose(_gravitational_variance_basis(alpha, tilt, None, None, wind_speed), (alpha[None, :] * c) ** 2, rtol=RTOL)
    np.testing.assert_allclose(
        _turbulant_wind_mean_bases(alpha, tilt, None, None, wind_speed)[0], alpha[None, :] * c * wind_speed[None, :], rtol=RTOL
    )
    np.testing.assert_allclose(
        _turbulant_wind_variance_basis(alpha, tilt, None, None, wind_speed), (alpha[None, :] * c * wind_speed[None, :]) ** 2, rtol=RTOL
    )


def test_settling_driven_variance_shares_the_mean_support():
    """Variance must be zero exactly where the mean is.

    cos^2 is sign-blind, so clipping only the mean would leave a face-down mirror with zero
    predicted deposition but FULL deposition noise -- an inconsistency that feeds the MLE
    likelihood weights and the prediction interval.
    """
    alpha = np.ones(1)
    wind_speed = np.full(1, 4.0)
    tilt = np.linspace(0.0, 180.0, 181).reshape(-1, 1)

    for mean_fn, var_fn in ((_gravitational_mean_bases, _gravitational_variance_basis), (_turbulant_wind_mean_bases, _turbulant_wind_variance_basis)):
        mean = mean_fn(alpha, tilt, None, None, wind_speed)[0]
        var = var_fn(alpha, tilt, None, None, wind_speed)
        assert (mean >= 0.0).all()
        assert (var >= 0.0).all()
        np.testing.assert_array_equal(mean == 0.0, var == 0.0)


def _tilt_sweep_model(components):
    """A three-mirror model at tilt 0 / 90 / 180 sharing one weather series."""
    f = 0
    density = 40.0
    tilt = np.tile(np.array([[0.0], [90.0], [180.0]]), (1, 4))
    azimuth = np.full((3, 4), 90.0)
    dust_conc = np.array([30.0, 40.0, 50.0, 60.0])

    model = ConstantMeanWindBase(components=components)
    model.mu_tilde = 0.5
    model.omega_tangential = 0.01
    model.omega_windward = 0.02
    model.omega_leeward = 0.005
    for name in ("sigma_dep", "sigma_dep_tan", "sigma_dep_gamma"):
        setattr(model, name, None)
    model.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})

    sim_in = types.SimpleNamespace(
        time={f: None},
        dust=types.SimpleNamespace(TSP={f: density}),
        dust_concentration={f: dust_conc},
        dust_type={f: "TSP"},
        wind_direction={f: np.array([90.0, 120.0, 200.0, 300.0])},
        wind_speed={f: np.array([3.0, 4.0, 5.0, 3.5])},
    )
    model.calculate_delta_soiled_area(sim_in, verbose=False)
    return model.helios.delta_soiled_area[f]


def test_face_down_mirror_accumulates_soiling_never_cleans_itself():
    """The end-to-end property the fix exists for: soiling accumulates monotonically at
    every tilt, and a face-down mirror soils LESS than an upward-facing one (yadnarie
    Feb 2025: -2.2 pp at 180 deg against -21.4 pp at 0 deg) rather than gaining reflectance."""
    delta = _tilt_sweep_model(["gravitational", "tangential_wind"])
    assert (delta >= 0.0).all()

    face_up, vertical, face_down = delta.sum(axis=1)
    assert face_down > 0.0  # tangential_wind is the one channel that still reaches it
    assert face_down < face_up
    assert vertical > 0.0


def test_default_components_leave_a_face_down_mirror_flat():
    """Documents the accepted consequence of the clip: with the default
    ["gravitational", "normal_wind"] set, EVERY factor is zero at tilt=180 (cos is clipped,
    sin(180)=0), so the mirror is predicted flat rather than inverted. tangential_wind must
    be in the component set for that mirror to be predicted at all."""
    delta = _tilt_sweep_model(["gravitational", "normal_wind"])
    np.testing.assert_allclose(delta[2, :], 0.0, atol=1e-15)
    assert delta[0, :].sum() > 0.0


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


# ---------------------------------------------------------------------------
# Least-squares fitting (mean_design_matrix / fit_ls)
# ---------------------------------------------------------------------------


def _synthetic_normal_wind_case(seed=7, meas_noise_std=5e-4):
    """A synthetic gravitational + normal_wind dataset and an unfitted model wired to it.

    Same construction as test_fit_mle_recovers_parameters_on_synthetic_data: known parameters
    generate the reflectance exactly, i.i.d. measurement noise is added on top, and the returned
    model carries only the geometry (every parameter still None). meas_noise_std=0 gives the
    noise-free case, where least squares should recover the generating parameters exactly; the
    reported sigma_of_the_mean is floored above zero regardless, since a zero measurement
    variance makes the likelihood the sigma profile uses undefined.

    Returns (model, sim_in, ref_dat, true_mean_parameters).
    """
    rng = np.random.default_rng(seed)
    f = 0
    N_helios, N_times = 4, 240  # 10 days, hourly
    truth_values = {"mu_tilde": 4.0e-4, "omega_windward": 6.0e-5, "omega_leeward": 2.0e-5}
    density, nominal_reflectance, incidence_angle = 40.0, 0.95, 15.0
    inc_ref_factor = np.float64(2.0 / np.cos(np.radians(incidence_angle)))

    tilt = np.tile(np.full(N_helios, 45.0)[:, None], (1, N_times))
    azimuth = np.tile(np.array([0.0, 90.0, 180.0, 270.0])[:, None], (1, N_times))  # N, E, S, W

    sim_in = types.SimpleNamespace(
        time={f: np.arange(N_times)},
        files=[f],
        dust=types.SimpleNamespace(TSP={f: density}),
        dust_concentration={f: rng.uniform(20.0, 80.0, size=N_times)},
        dust_type={f: "TSP"},
        wind_direction={f: rng.uniform(0.0, 360.0, size=N_times)},
        wind_speed={f: rng.uniform(1.0, 6.0, size=N_times)},
    )

    truth = ConstantMeanWindBase()
    for name, value in truth_values.items():
        setattr(truth, name, value)
    truth.sigma_dep = truth.sigma_dep_gamma = None
    truth.helios = _wind_helios_stub(tilt={f: tilt}, azimuth={f: azimuth})
    truth.calculate_delta_soiled_area(sim_in, verbose=False)
    soiling_factor_true = 1 - np.cumsum(truth.helios.delta_soiled_area[f], axis=1) * inc_ref_factor  # clean start

    pred_idx = np.arange(23, N_times, 24)  # daily measurements
    rho_true_at_meas = nominal_reflectance * soiling_factor_true[:, pred_idx]
    average = (rho_true_at_meas + rng.normal(0.0, meas_noise_std, size=rho_true_at_meas.shape)).T

    ref_dat = types.SimpleNamespace(
        files=[f],
        times={f: pred_idx},
        prediction_indices={f: list(pred_idx)},
        average={f: average},
        sigma_of_the_mean={f: np.full(average.shape, max(meas_noise_std, 1e-6))},
        rho0={f: np.full(N_helios, nominal_reflectance)},
    )

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
    return model, sim_in, ref_dat, truth_values


def test_mean_design_matrix_reproduces_the_sum_of_squares():
    """The whole least-squares fit rests on the prediction being affine in the mean parameters,
    so the design matrix must reproduce CommonFittingMethods._sse exactly -- not approximately --
    at arbitrary parameter values, not just near the ones it was built at."""
    model, sim_in, ref_dat, _truth = _synthetic_normal_wind_case()

    X, y = model.mean_design_matrix(sim_in, ref_dat)
    assert X.shape == (ref_dat.average[0].size, 3)  # mu_tilde, omega_windward, omega_leeward

    rng = np.random.default_rng(3)
    for _ in range(4):
        theta = rng.uniform(-1e-3, 1e-3, size=3)
        theta[0] = abs(theta[0])  # mu_tilde is a positive (log-transformed) parameter
        residual = X @ theta - y
        sse_forward = model._sse(np.concatenate([theta, [1e-4, 1e-4]]), sim_in, ref_dat)
        np.testing.assert_allclose(residual @ residual, sse_forward, rtol=1e-9)


def test_mean_design_matrix_leaves_the_model_parameters_untouched():
    """Building the design matrix probes the forward model by overwriting the parameters; it has
    to put them back, or a subsequent fit silently starts from the last probe."""
    model, sim_in, ref_dat, _truth = _synthetic_normal_wind_case()
    model.update_model_parameters(np.array([1e-4, 2e-5, 3e-5, 4e-5, 5e-5]))
    before = [getattr(model, name) for name in model.parameter_names]

    model.mean_design_matrix(sim_in, ref_dat)

    assert [getattr(model, name) for name in model.parameter_names] == before


def test_mean_design_matrix_increment_residuals_difference_consecutive_measurements():
    """residuals="increment" fits the change in reflectance between measurements, so it has one
    row per interval rather than per measurement, and its rows are the level rows differenced."""
    model, sim_in, ref_dat, _truth = _synthetic_normal_wind_case()
    n_times, n_helios = ref_dat.average[0].shape

    _X_level, y_level = model.mean_design_matrix(sim_in, ref_dat, residuals="level")
    X_inc, y_inc = model.mean_design_matrix(sim_in, ref_dat, residuals="increment")

    assert X_inc.shape == ((n_times - 1) * n_helios, 3)
    expected = np.diff(y_level.reshape(n_times, n_helios), axis=0).ravel()
    np.testing.assert_allclose(y_inc, expected, rtol=1e-10, atol=1e-15)


def test_mean_design_matrix_rejects_an_unknown_residual_kind():
    model, sim_in, ref_dat, _truth = _synthetic_normal_wind_case()
    with pytest.raises(ValueError, match="residuals"):
        model.mean_design_matrix(sim_in, ref_dat, residuals="daily")


def test_fit_ls_is_exact_without_measurement_noise():
    """With noise-free measurements the least-squares problem is consistent, so the linear solve
    must return the generating parameters -- no starting point, no tolerance to tune."""
    model, sim_in, ref_dat, truth = _synthetic_normal_wind_case(meas_noise_std=0.0)

    x_hat, _x_cov = model.fit_ls(sim_in, ref_dat, verbose=False, transform_to_original_scale=True)

    np.testing.assert_allclose(x_hat, [truth["mu_tilde"], truth["omega_windward"], truth["omega_leeward"]], rtol=1e-6)


def test_fit_ls_recovers_parameters_on_synthetic_data():
    """Counterpart of test_fit_mle_recovers_parameters_on_synthetic_data: the same data, fitted
    by least squares instead of a joint MLE. The estimate covers the mean parameters only, and
    the fit leaves the model holding it."""
    model, sim_in, ref_dat, truth = _synthetic_normal_wind_case()

    x_hat, x_cov = model.fit_ls(sim_in, ref_dat, verbose=False, transform_to_original_scale=True)
    mu_hat, omega_w_hat, omega_l_hat = x_hat

    assert np.all(np.isfinite(x_hat))
    assert x_cov.shape == (3, 3)
    assert model.mean_parameter_names == ["mu_tilde", "omega_windward", "omega_leeward"]
    assert abs(mu_hat - truth["mu_tilde"]) / truth["mu_tilde"] < 0.2
    assert abs(omega_w_hat - truth["omega_windward"]) / truth["omega_windward"] < 0.4
    assert abs(omega_l_hat - truth["omega_leeward"]) / truth["omega_leeward"] < 0.6

    np.testing.assert_allclose([getattr(model, n) for n in model.mean_parameter_names], x_hat, rtol=1e-12)


def test_fit_ls_leaves_no_noise_model_and_no_prediction_variance():
    """Least squares says nothing about the sigmas, so the fit must clear them rather than leave
    a stale value behind -- that is what makes predict_soiling_factor produce a mean prediction
    with no prediction interval for the figures to draw."""
    model, sim_in, ref_dat, _truth = _synthetic_normal_wind_case()
    model.update_model_parameters(np.array([1e-4, 2e-5, 3e-5, 4e-5, 5e-5]))  # a stale MLE-style fit

    model.fit_ls(sim_in, ref_dat, verbose=False)

    assert [getattr(model, name) for name in model._sigma_param_names] == [None, None]
    model.predict_soiling_factor(sim_in, reflectance_data=ref_dat, verbose=False)
    assert model.helios.soiling_factor_prediction_variance == {}


def test_fit_ls_fitted_scale_matches_the_original_scale_estimate():
    """fit_ls follows fit_mle's convention: transform_to_original_scale=False returns the
    estimate on the model's own (partially log) scale, which transform_scale maps back."""
    model, sim_in, ref_dat, _truth = _synthetic_normal_wind_case()

    y_hat, y_cov = model.fit_ls(sim_in, ref_dat, verbose=False, transform_to_original_scale=False)
    x_hat, _x_cov = model.fit_ls(sim_in, ref_dat, verbose=False, transform_to_original_scale=True)

    np.testing.assert_allclose(model.transform_scale(y_hat), x_hat, rtol=1e-6)
    assert y_cov.shape == (3, 3)
    # The log-transformed entry (mu_tilde) differs from its original; the linear omegas are
    # carried through untouched.
    np.testing.assert_allclose(y_hat[1:], x_hat[1:], rtol=1e-12)


def test_transform_scale_rejects_a_vector_that_is_neither_mean_only_nor_complete():
    """transform_scale now takes either length, so a wrong one must fail loudly instead of
    silently broadcasting against a mismatched log mask."""
    model, _sim_in, _ref_dat, _truth = _synthetic_normal_wind_case()

    with pytest.raises(ValueError, match="Cannot transform a vector of 4 parameter"):
        model.transform_scale(np.ones(4), direction="forward")


def test_fit_ls_handles_a_component_the_data_cannot_see():
    """A mechanism whose geometry basis is identically zero (impaction_retention with every
    mirror horizontal) makes the design matrix singular. The fit must still return finite
    estimates for the parameters the data does identify rather than dying in the linear solve."""
    model, sim_in, ref_dat, _truth = _synthetic_normal_wind_case()
    f = 0
    helios = model.helios
    helios.tilt[f] = np.zeros_like(helios.tilt[f])  # flat: sin(tilt)*cos(tilt) == 0
    # Re-initialising the base rebuilds helios from scratch, so the (now flat) stub goes back
    # afterwards rather than before.
    ConstantMeanWindBase.__init__(model, components=["gravitational", "impaction_retention"])
    model.helios = helios

    x_hat, _x_cov = model.fit_ls(sim_in, ref_dat, verbose=False, transform_to_original_scale=True)

    assert np.all(np.isfinite(x_hat))
    np.testing.assert_allclose(x_hat[1:3], 0.0, atol=1e-30)  # the unidentifiable omegas stay at 0
    assert x_hat[0] > 0  # mu_tilde is still identified by the cos(tilt) term


def _constant_mean_model(model, sim_in, ref_dat):
    """A ConstantMeanDeposition wired to the same synthetic case, bypassing the Excel-reading
    __init__ exactly as the ConstantMeanWindDeposition models above do."""
    plain = smf.ConstantMeanDeposition.__new__(smf.ConstantMeanDeposition)
    smb.ConstantMeanBase.__init__(plain)
    plain.helios = _wind_helios_stub(
        tilt=dict(model.helios.tilt),
        inc_ref_factor=dict(model.helios.inc_ref_factor),
        nominal_reflectance=model.helios.nominal_reflectance,
        soiling_factor={},
        soiling_factor_prediction_variance={},
    )
    return plain


def test_constant_mean_least_squares_matches_the_gravitational_only_wind_model():
    """ConstantMeanDeposition is affine in mu_tilde, so it takes the same one-shot linear fit
    as the wind family. The two are the same model (see
    test_gravitational_only_reproduces_constant_mean_base_and_skips_wind_requirement), so
    fitting the same data must give the same answer -- to solver precision, not loosely."""
    model, sim_in, ref_dat, _truth = _synthetic_normal_wind_case()
    helios = model.helios
    ConstantMeanWindBase.__init__(model, components=["gravitational"])
    model.helios = helios

    plain = _constant_mean_model(model, sim_in, ref_dat)

    wind_hat, wind_cov = model.fit_ls(sim_in, ref_dat, verbose=False, transform_to_original_scale=True)
    plain_hat, plain_cov = plain.fit_ls(sim_in, ref_dat, verbose=False, transform_to_original_scale=True)

    assert plain.mean_parameter_names == ["mu_tilde"] == model.mean_parameter_names
    np.testing.assert_allclose(plain_hat, wind_hat, rtol=1e-12)
    np.testing.assert_allclose(plain_cov, wind_cov, rtol=1e-10)


def test_constant_mean_least_squares_leaves_no_noise_model():
    """Same contract as the wind model: no sigma is fitted, a stale one is cleared, and the
    prediction therefore carries no variance for the figures to shade."""
    model, sim_in, ref_dat, _truth = _synthetic_normal_wind_case()
    plain = _constant_mean_model(model, sim_in, ref_dat)
    plain.update_model_parameters(np.array([1e-4, 5e-5]))  # a stale MLE-style fit

    x_hat, _cov = plain.fit_ls(sim_in, ref_dat, verbose=False, transform_to_original_scale=True)

    assert plain.sigma_dep is None
    assert plain.mu_tilde == x_hat[0] > 0
    plain.predict_soiling_factor(sim_in, reflectance_data=ref_dat, verbose=False)
    assert plain.helios.soiling_factor_prediction_variance == {}


def test_constant_mean_least_squares_beats_the_mle_on_its_own_objective():
    """The point of offering least squares: on the sum of squares it minimizes, it must do at
    least as well as the likelihood fit, which optimizes something else."""
    model, sim_in, ref_dat, _truth = _synthetic_normal_wind_case()
    ls_model = _constant_mean_model(model, sim_in, ref_dat)
    mle_model = _constant_mean_model(model, sim_in, ref_dat)

    ls_hat, _cov = ls_model.fit_ls(sim_in, ref_dat, verbose=False, transform_to_original_scale=True)
    mle_hat, _mle_cov = mle_model.fit_mle(sim_in, ref_dat, verbose=False, x0=np.array([1e-3, 1e-4]), transform_to_original_scale=True)

    sse_ls = ls_model._sse(np.array([ls_hat[0], 1e-4]), sim_in, ref_dat)
    sse_mle = mle_model._sse(np.array([mle_hat[0], 1e-4]), sim_in, ref_dat)

    assert sse_ls <= sse_mle


def test_constant_mean_transform_scale_takes_either_length():
    """fit_ls returns mu_tilde alone, so transform_scale must accept a length-1 vector while
    leaving the two-parameter behaviour fit_mle relies on untouched."""
    model, sim_in, ref_dat, _truth = _synthetic_normal_wind_case()
    plain = _constant_mean_model(model, sim_in, ref_dat)

    np.testing.assert_allclose(plain.transform_scale([np.e], direction="forward"), [1.0], rtol=1e-12)
    np.testing.assert_allclose(plain.transform_scale([np.e, np.e**2], direction="forward"), [1.0, 2.0], rtol=1e-12)
    with pytest.raises(ValueError, match="Cannot transform a vector of 3 parameter"):
        plain.transform_scale([1.0, 2.0, 3.0], direction="forward")


# ---------------------------------------------------------------------------
# Shared least-squares machinery (heliosoil.fitting), across all three model classes
# ---------------------------------------------------------------------------


def _bare_wind_model(components=None):
    """A ConstantMeanWindDeposition carrying only its parameter layout -- enough for the
    scale-transform hooks, which touch no data."""
    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanWindBase.__init__(model, components=components)
    return model


@pytest.mark.parametrize(
    "make_model, point",
    [
        (lambda: smf.ConstantMeanDeposition.__new__(smf.ConstantMeanDeposition), np.array([3.0e-5])),
        (lambda: smf.SemiPhysical.__new__(smf.SemiPhysical), np.array([250.0])),
        (_bare_wind_model, np.array([3.0e-5, 7.0e-7, -2.0e-7])),  # a negative omega is legal
    ],
    ids=["constant_mean", "semi_physical", "constant_mean_wind"],
)
def test_fitted_scale_jacobian_matches_the_forward_transform(make_model, point):
    """The delta method carries a least-squares covariance onto the fitted scale using this
    derivative. Each class states it analytically next to its own transform_scale, so a hook
    that disagreed would silently corrupt every reported confidence interval -- nothing else
    would fail. Checked against a central difference of the transform itself."""
    model = make_model()
    analytic = np.atleast_1d(model._fitted_scale_jacobian(point))

    numeric = np.empty(len(point), dtype=float)
    for i in range(len(point)):
        step = 1e-6 * abs(point[i])
        up, down = point.astype(float).copy(), point.astype(float).copy()
        up[i] += step
        down[i] -= step
        numeric[i] = (model.transform_scale(up, direction="forward")[i] - model.transform_scale(down, direction="forward")[i]) / (2 * step)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5)


def test_least_squares_reports_an_estimate_pinned_to_its_search_bound(caplog):
    """A bounded scalar fit that stops on its bracket has not found an optimum -- the objective
    was still improving. That is a real diagnostic about identifiability, so it must be said
    out loud rather than returned as if it were a converged estimate."""
    model = smf.SemiPhysical.__new__(smf.SemiPhysical)
    lower, upper = model._LS_SCALAR_BOUNDS

    with caplog.at_level(logging.WARNING, logger="heliosoil"):
        # A bounded search converges only to its own xatol, so it stops just short of the
        # bound -- 999.99998 for yadnarie's semi-physical fit. That must still be flagged.
        model._warn_if_at_scalar_bound("hrz0", upper - 2.1e-5)
        assert "search bound" in caplog.text

        caplog.clear()
        model._warn_if_at_scalar_bound("hrz0", lower + 1e-6)
        assert "search bound" in caplog.text

        caplog.clear()
        model._warn_if_at_scalar_bound("hrz0", 0.5 * upper)
        assert caplog.text == ""


def test_semi_physical_transform_scale_takes_either_length():
    """fit_ls returns hrz0 alone, so the log-log transform must accept a length-1 vector while
    leaving the two-parameter behaviour fit_mle relies on untouched."""
    model = smf.SemiPhysical.__new__(smf.SemiPhysical)

    full = model.transform_scale([np.exp(np.e), np.e], direction="forward")
    mean_only = model.transform_scale([np.exp(np.e)], direction="forward")

    np.testing.assert_allclose(full, [1.0, 1.0], rtol=1e-12)
    np.testing.assert_allclose(mean_only, full[:1], rtol=1e-12)
    with pytest.raises(ValueError, match="Cannot transform a vector of 3 parameter"):
        model.transform_scale([1.0, 2.0, 3.0], direction="forward")


def test_mean_parameter_names_cover_every_model_class():
    """The pipeline reads a least-squares run's reported parameters off this property, so each
    class must declare its mean/noise split rather than inherit an empty default."""
    assert smf.ConstantMeanDeposition.__new__(smf.ConstantMeanDeposition).mean_parameter_names == ["mu_tilde"]
    assert smf.SemiPhysical.__new__(smf.SemiPhysical).mean_parameter_names == ["hrz0"]
    assert _bare_wind_model(["gravitational", "tangential_wind"]).mean_parameter_names == ["mu_tilde", "omega_tangential"]
