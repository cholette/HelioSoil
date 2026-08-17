"""
Tests that the deposition-variance window matches the mean window.

A reflectance difference between measurements at simulation indices k_{i-1} and k_i
accumulates the deposition of intervals k_{i-1}+1 ... k_i INCLUSIVE: compute_soiling_factor
forms the mean with an inclusive cumulative sum, so soiling_factor[k] already contains
delta_soiled_area[k]. The variance of that difference must cover exactly the same intervals.
Taking the cumulative sum at k_i - 1 instead drops the last interval of every window, which
understates the deposition variance by roughly one interval in (k_i - k_{i-1}) and makes the
likelihood internally inconsistent -- its mean and its variance disagreeing about which
intervals a difference spans.

The window is pinned two ways, both independent of the implementation:

1. ``test_*_spike_*`` -- a dust concentration nonzero in exactly one interval must move the
   mean and the variance of the SAME difference. This ties the two windows to each other
   without hard-coding either, so it keeps holding if the accumulation convention changes.
2. ``test_*_matches_explicit_window_sum`` -- the variance against an explicit per-difference
   loop over ``range(k_{i-1}+1, k_i+1)``, which states the intended window directly.

Both the plain constant-mean model (heliosoil.fitting) and the wind model
(heliosoil.horizontal_impaction) are covered, since each carries its own copy of the
variance assembly.

The last two sections cover the mechanism-independent pieces the multi-mirror covariance
is later assembled from: ``_difference_windows`` (the same window, now returned explicitly
rather than implied by an index arithmetic), ``_reflectance_loss_factor`` (per mirror,
because the drift correction gives each its own nominal reflectance) and
``_differenced_measurement_covariance`` (R = Delta Sigma_r Delta', tridiagonal because
consecutive differences share an endpoint measurement).
"""

import types

import numpy as np
import pytest

from heliosoil.base_models import ConstantMeanBase
from heliosoil.fitting import ConstantMeanDeposition
from heliosoil.horizontal_impaction import ConstantMeanWindDeposition
from heliosoil.utilities import gravitational_settling_factor, cosd, sind, _nominal_reflectance_anchor

RTOL = 1e-12
ATOL = 0.0

F = 0
T_GRID = 21
PREDICTION_INDICES = [0, 4, 9, 17, 20]  # uneven spacing: dk = 4, 5, 8, 3
N_DIFF = len(PREDICTION_INDICES) - 1

MU_TILDE = 0.015
SIGMA_DEP = 8e-4
OMEGA_WINDWARD = 6.0e-07
OMEGA_LEEWARD = 9.0e-07
SIGMA_DEP_GAMMA = 5e-06
DENSITY = 40.0
NOMINAL_REFLECTANCE = 0.95
INC_REF_FACTOR = 2.0 / np.cos(np.radians(15.0))


def _blank_helios_dicts(model):
    for name in (
        "delta_soiled_area",
        "delta_soiled_area_variance",
        "soiling_factor",
        "soiling_factor_prediction_variance",
    ):
        setattr(model.helios, name, {})


def _constant_mean_model(tilt):
    """A ConstantMeanDeposition with site data stubbed instead of read from Excel."""
    model = ConstantMeanDeposition.__new__(ConstantMeanDeposition)
    ConstantMeanBase.__init__(model)
    model.mu_tilde, model.sigma_dep = MU_TILDE, SIGMA_DEP
    model.helios.nominal_reflectance = NOMINAL_REFLECTANCE
    model.helios.tilt = {F: tilt}
    model.helios.inc_ref_factor = {F: np.array(INC_REF_FACTOR)}
    _blank_helios_dicts(model)
    return model


def _wind_model(tilt, azimuth):
    """A gravitational + normal_wind model, likewise stubbed."""
    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanBase.__init__(model)
    model.components = None  # replaced immediately below by the real resolution
    from heliosoil.horizontal_impaction import ConstantMeanWindBase

    ConstantMeanWindBase.__init__(model, components=["gravitational", "normal_wind"])
    model.verbose = False
    model.mu_tilde, model.sigma_dep = MU_TILDE, SIGMA_DEP
    model.omega_windward, model.omega_leeward = OMEGA_WINDWARD, OMEGA_LEEWARD
    model.sigma_dep_gamma = SIGMA_DEP_GAMMA
    model.helios.nominal_reflectance = NOMINAL_REFLECTANCE
    model.helios.tilt = {F: tilt}
    model.helios.azimuth = {F: azimuth}
    model.helios.inc_ref_factor = {F: np.array(INC_REF_FACTOR)}
    _blank_helios_dicts(model)
    return model


def _sim_in(concentration, wind_speed=None, wind_direction=None):
    sim = types.SimpleNamespace(
        files=["experiment_0"],
        time={F: np.arange(T_GRID, dtype=float)},
        dust=types.SimpleNamespace(PM10={F: DENSITY}),
        dust_concentration={F: np.asarray(concentration, dtype=float)},
        dust_type={F: "PM10"},
    )
    if wind_speed is not None:
        sim.wind_speed = {F: np.asarray(wind_speed, dtype=float)}
        sim.wind_direction = {F: np.asarray(wind_direction, dtype=float)}
    return sim


def _ref_dat(n_mirrors, measurement_sigma=0.0):
    shape = (len(PREDICTION_INDICES), n_mirrors)
    return types.SimpleNamespace(
        files=["experiment_0"],
        times={F: np.asarray(PREDICTION_INDICES, dtype=float)},
        prediction_indices={F: PREDICTION_INDICES},
        rho0={F: np.full(n_mirrors, 0.93)},
        sigma_of_the_mean={F: np.full(shape, measurement_sigma)},
        average={F: np.zeros(shape)},
    )


def _mean_difference_windows(model, sim_in, ref_dat):
    """Per-difference sum of delta_soiled_area, read off the model's own forward pass.

    This is the window the MEAN uses, derived from compute_soiling_factor's convention
    rather than restated, so the tests below compare the variance against the code's
    actual behaviour.
    """
    model.predict_soiling_factor(sim_in, reflectance_data=ref_dat, verbose=False)
    delta = model.helios.delta_soiled_area[F]
    pi = PREDICTION_INDICES
    return np.array(
        [delta[:, pi[i] + 1 : pi[i + 1] + 1].sum(axis=1) for i in range(N_DIFF)]
    )


# ---------------------------------------------------------------------------
# 1. Spike tests -- the mean and the variance must respond to the same interval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spike_index", range(1, T_GRID))
def test_constant_mean_spike_moves_mean_and_variance_together(spike_index):
    """Dust in exactly one interval must move the mean and variance of one same difference."""
    tilt = np.full((1, T_GRID), 30.0)
    model, ref_dat = _constant_mean_model(tilt), _ref_dat(1)

    concentration = np.zeros(T_GRID)
    concentration[spike_index] = 50.0
    sim_in = _sim_in(concentration)

    mean_by_difference = _mean_difference_windows(model, sim_in, ref_dat)[:, 0]
    variance_by_difference = model._compute_variance_of_measurements(
        SIGMA_DEP, sim_in, reflectance_data=ref_dat
    )[F][:, 0]

    moved_mean = set(np.flatnonzero(np.abs(mean_by_difference) > 0.0))
    moved_variance = set(np.flatnonzero(variance_by_difference > 0.0))
    assert moved_mean == moved_variance, (
        f"interval {spike_index} enters the mean of differences {sorted(moved_mean)} "
        f"but the variance of differences {sorted(moved_variance)}"
    )


@pytest.mark.parametrize("spike_index", range(1, T_GRID))
def test_wind_model_spike_moves_mean_and_variance_together(spike_index):
    """Same invariant for the wind model, whose variance assembly is a separate copy."""
    tilt = np.full((2, T_GRID), 45.0)
    azimuth = np.tile(np.array([0.0, 180.0])[:, None], (1, T_GRID))
    model, ref_dat = _wind_model(tilt, azimuth), _ref_dat(2)

    concentration = np.zeros(T_GRID)
    concentration[spike_index] = 50.0
    sim_in = _sim_in(
        concentration,
        wind_speed=np.full(T_GRID, 4.0),
        wind_direction=np.full(T_GRID, 20.0),
    )

    mean_by_difference = _mean_difference_windows(model, sim_in, ref_dat)
    variance_by_difference = model._compute_variance_of_measurements(
        None, sim_in, reflectance_data=ref_dat
    )[F]

    for mirror in range(tilt.shape[0]):
        moved_mean = set(np.flatnonzero(np.abs(mean_by_difference[:, mirror]) > 0.0))
        moved_variance = set(np.flatnonzero(variance_by_difference[:, mirror] > 0.0))
        assert moved_mean == moved_variance, (
            f"mirror {mirror}: interval {spike_index} enters the mean of differences "
            f"{sorted(moved_mean)} but the variance of differences {sorted(moved_variance)}"
        )


# ---------------------------------------------------------------------------
# 2. The variance against an explicit window sum
# ---------------------------------------------------------------------------


def test_constant_mean_variance_matches_explicit_window_sum():
    """Reference built by looping the intended window, independent of the implementation."""
    rng = np.random.default_rng(0)
    tilt = rng.uniform(0.0, 60.0, size=(3, T_GRID))
    concentration = np.linspace(20.0, 90.0, T_GRID)
    measurement_sigma = 1.5e-3

    model = _constant_mean_model(tilt)
    sim_in, ref_dat = _sim_in(concentration), _ref_dat(3, measurement_sigma)
    model.predict_soiling_factor(sim_in, reflectance_data=ref_dat, verbose=False)

    got = model._compute_variance_of_measurements(
        SIGMA_DEP, sim_in, reflectance_data=ref_dat
    )[F]

    alpha = concentration / DENSITY
    basis = (alpha[None, :] * gravitational_settling_factor(tilt)) ** 2
    b = NOMINAL_REFLECTANCE * INC_REF_FACTOR
    pi = PREDICTION_INDICES
    expected = np.empty((N_DIFF, tilt.shape[0]))
    for i in range(N_DIFF):
        window = range(pi[i] + 1, pi[i + 1] + 1)  # inclusive of the endpoint interval
        expected[i] = SIGMA_DEP**2 * b**2 * basis[:, list(window)].sum(axis=1)
    expected += 2.0 * measurement_sigma**2  # sigma_{k_i}^2 + sigma_{k_{i-1}}^2

    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


def test_wind_model_variance_matches_explicit_window_sum():
    """Same reference for the wind model, summing both active mechanisms."""
    rng = np.random.default_rng(1)
    tilt = rng.uniform(0.0, 60.0, size=(2, T_GRID))
    azimuth = np.tile(np.array([0.0, 135.0])[:, None], (1, T_GRID))
    concentration = np.linspace(20.0, 90.0, T_GRID)
    wind_speed = np.linspace(1.0, 7.0, T_GRID)
    wind_direction = np.linspace(0.0, 350.0, T_GRID)
    measurement_sigma = 1.0e-3

    model = _wind_model(tilt, azimuth)
    sim_in = _sim_in(concentration, wind_speed, wind_direction)
    ref_dat = _ref_dat(2, measurement_sigma)
    model.predict_soiling_factor(sim_in, reflectance_data=ref_dat, verbose=False)

    got = model._compute_variance_of_measurements(None, sim_in, reflectance_data=ref_dat)[F]

    alpha = concentration / DENSITY
    grav = (alpha[None, :] * gravitational_settling_factor(tilt)) ** 2
    delta_gamma = azimuth - wind_direction[None, :]
    projection = sind(tilt) * np.abs(cosd(delta_gamma))  # p_windward + p_leeward
    wind = (alpha[None, :] * wind_speed[None, :] * projection) ** 2

    b = NOMINAL_REFLECTANCE * INC_REF_FACTOR
    pi = PREDICTION_INDICES
    expected = np.empty((N_DIFF, tilt.shape[0]))
    for i in range(N_DIFF):
        window = list(range(pi[i] + 1, pi[i + 1] + 1))
        expected[i] = b**2 * (
            SIGMA_DEP**2 * grav[:, window].sum(axis=1)
            + SIGMA_DEP_GAMMA**2 * wind[:, window].sum(axis=1)
        )
    expected += 2.0 * measurement_sigma**2

    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 3. The window covers the whole record, with no gaps and no double counting
# ---------------------------------------------------------------------------


def test_windows_partition_the_measured_span():
    """Summing the per-difference variances recovers one sweep of the whole span.

    The difference windows are disjoint and together cover intervals k_0+1 ... k_N, so with
    the measurement noise switched off the total is the deposition variance accumulated once
    over that span. A window short by one interval per difference fails this by construction.
    """
    tilt = np.full((1, T_GRID), 25.0)
    concentration = np.linspace(30.0, 80.0, T_GRID)
    model = _constant_mean_model(tilt)
    sim_in, ref_dat = _sim_in(concentration), _ref_dat(1, 0.0)
    model.predict_soiling_factor(sim_in, reflectance_data=ref_dat, verbose=False)

    total = model._compute_variance_of_measurements(
        SIGMA_DEP, sim_in, reflectance_data=ref_dat
    )[F][:, 0].sum()

    alpha = concentration / DENSITY
    basis = (alpha * gravitational_settling_factor(tilt)[0]) ** 2
    b = NOMINAL_REFLECTANCE * INC_REF_FACTOR
    span = slice(PREDICTION_INDICES[0] + 1, PREDICTION_INDICES[-1] + 1)
    expected = SIGMA_DEP**2 * b**2 * basis[span].sum()

    np.testing.assert_allclose(total, expected, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 4. _difference_windows and _reflectance_loss_factor
# ---------------------------------------------------------------------------


def test_difference_windows_are_the_intervals_the_mean_accumulates():
    """The returned slices are exactly k_{i-1}+1 ... k_i, disjoint and gap-free."""
    model = _constant_mean_model(np.full((1, T_GRID), 20.0))
    windows = model._difference_windows(F, _ref_dat(1))

    assert len(windows) == N_DIFF
    pi = PREDICTION_INDICES
    covered = []
    for i, window in enumerate(windows):
        indices = list(range(*window.indices(T_GRID)))
        assert indices == list(range(pi[i] + 1, pi[i + 1] + 1))
        covered.extend(indices)

    assert len(covered) == len(set(covered)), "windows overlap"
    assert covered == list(range(pi[0] + 1, pi[-1] + 1)), "windows leave a gap"


def test_difference_windows_agree_with_the_variance_they_weight():
    """Summing a basis over the returned windows reproduces the variance's own windowing.

    Ties the explicit helper to the arithmetic in _compute_variance_of_measurements, so the
    two cannot drift apart when the covariance assembly starts using the helper instead.
    """
    rng = np.random.default_rng(3)
    tilt = rng.uniform(0.0, 60.0, size=(2, T_GRID))
    concentration = np.linspace(25.0, 85.0, T_GRID)

    model = _constant_mean_model(tilt)
    sim_in, ref_dat = _sim_in(concentration), _ref_dat(2, 0.0)
    model.predict_soiling_factor(sim_in, reflectance_data=ref_dat, verbose=False)

    got = model._compute_variance_of_measurements(SIGMA_DEP, sim_in, reflectance_data=ref_dat)[F]

    alpha = concentration / DENSITY
    basis = (alpha[None, :] * gravitational_settling_factor(tilt)) ** 2
    b = model._reflectance_loss_factor(F, ref_dat)
    windows = model._difference_windows(F, ref_dat)
    expected = np.stack([SIGMA_DEP**2 * b**2 * basis[:, w].sum(axis=1) for w in windows])

    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


def test_reflectance_loss_factor_is_per_mirror_and_constant_without_drift():
    """Without a reference mirror every entry is the same fixed constant."""
    model = _constant_mean_model(np.full((4, T_GRID), 30.0))
    b = model._reflectance_loss_factor(F, _ref_dat(4))

    assert b.shape == (4,)
    np.testing.assert_allclose(b, NOMINAL_REFLECTANCE * INC_REF_FACTOR, rtol=RTOL)


def test_reflectance_loss_factor_follows_the_drift_corrected_anchor():
    """With a reference mirror each mirror gets its own nominal reflectance.

    b must use the nominal reflectance at the moment that mirror's rho0 was recorded --
    the same _nominal_reflectance_anchor the rest of the fitting code uses -- or the
    covariance would be scaled by a different b than the mean is.
    """
    n_mirrors = 3
    model = _constant_mean_model(np.full((n_mirrors, T_GRID), 30.0))
    ref_dat = _ref_dat(n_mirrors)

    nominal = np.tile(np.linspace(0.95, 0.90, len(PREDICTION_INDICES))[:, None], (1, n_mirrors))
    nominal *= np.array([1.00, 1.01, 0.99])[None, :]
    ref_dat.nominal_reflectance = {F: nominal}
    ref_dat.rho0_index = {F: np.array([0, 2, 1])}

    b = model._reflectance_loss_factor(F, ref_dat)
    anchor = _nominal_reflectance_anchor(ref_dat, F, NOMINAL_REFLECTANCE)

    assert b.shape == (n_mirrors,)
    np.testing.assert_allclose(b, anchor * INC_REF_FACTOR, rtol=RTOL)
    assert len(set(np.round(b, 12))) == n_mirrors, "drift should make the mirrors differ"


def test_reflectance_loss_factor_rejects_a_varying_incidence_angle():
    model = _constant_mean_model(np.full((2, T_GRID), 30.0))
    model.helios.inc_ref_factor = {F: np.array([2.0, 2.1])}
    with pytest.raises(ValueError, match="fixed reflectometer"):
        model._reflectance_loss_factor(F, _ref_dat(2))


# ---------------------------------------------------------------------------
# 5. _differenced_measurement_covariance
# ---------------------------------------------------------------------------


def _difference_operator(n_meas):
    """The (n_meas-1, n_meas) first-difference operator, built explicitly."""
    delta = np.zeros((n_meas - 1, n_meas))
    for i in range(n_meas - 1):
        delta[i, i] = -1.0
        delta[i, i + 1] = 1.0
    return delta


def test_measurement_covariance_matches_delta_sigma_delta():
    """R = Delta Sigma_r Delta', against the operator written out in full."""
    rng = np.random.default_rng(11)
    n_mirrors = 3
    sigma = rng.uniform(5e-4, 2e-3, size=(len(PREDICTION_INDICES), n_mirrors))

    model = _constant_mean_model(np.full((n_mirrors, T_GRID), 30.0))
    ref_dat = _ref_dat(n_mirrors)
    ref_dat.sigma_of_the_mean = {F: sigma}

    got = model._differenced_measurement_covariance(F, ref_dat, endpoint_correction=True)
    delta = _difference_operator(len(PREDICTION_INDICES))

    assert got.shape == (n_mirrors, N_DIFF, N_DIFF)
    for p in range(n_mirrors):
        expected = delta @ np.diag(sigma[:, p] ** 2) @ delta.transpose()
        np.testing.assert_allclose(got[p], expected, rtol=RTOL, atol=ATOL)


def test_measurement_covariance_is_tridiagonal_and_symmetric():
    rng = np.random.default_rng(12)
    n_mirrors = 2
    sigma = rng.uniform(5e-4, 2e-3, size=(len(PREDICTION_INDICES), n_mirrors))
    model = _constant_mean_model(np.full((n_mirrors, T_GRID), 30.0))
    ref_dat = _ref_dat(n_mirrors)
    ref_dat.sigma_of_the_mean = {F: sigma}

    cov = model._differenced_measurement_covariance(F, ref_dat)
    for p in range(n_mirrors):
        np.testing.assert_allclose(cov[p], cov[p].transpose(), rtol=RTOL, atol=ATOL)
        for i in range(N_DIFF):
            for j in range(N_DIFF):
                if abs(i - j) >= 2:
                    assert cov[p, i, j] == 0.0, f"entry ({i},{j}) should be zero"
        np.linalg.cholesky(cov[p])  # raises if not positive definite


def test_measurement_covariance_without_the_endpoint_correction_is_todays_behaviour():
    """endpoint_correction=False must reproduce exactly what the current likelihood uses."""
    rng = np.random.default_rng(13)
    n_mirrors = 3
    sigma = rng.uniform(5e-4, 2e-3, size=(len(PREDICTION_INDICES), n_mirrors))
    model = _constant_mean_model(np.full((n_mirrors, T_GRID), 30.0))
    ref_dat = _ref_dat(n_mirrors)
    ref_dat.sigma_of_the_mean = {F: sigma}

    cov = model._differenced_measurement_covariance(F, ref_dat, endpoint_correction=False)
    s2 = sigma**2
    for p in range(n_mirrors):
        np.testing.assert_allclose(np.diag(cov[p]), s2[1:, p] + s2[:-1, p], rtol=RTOL, atol=ATOL)
        assert np.count_nonzero(cov[p] - np.diag(np.diag(cov[p]))) == 0


def test_measurement_covariance_off_diagonal_is_the_shared_endpoint():
    """The off-diagonal is -sigma^2 of the measurement the two differences share."""
    n_mirrors = 1
    sigma = np.array([[1e-3], [2e-3], [3e-3], [4e-3], [5e-3]])
    model = _constant_mean_model(np.full((n_mirrors, T_GRID), 30.0))
    ref_dat = _ref_dat(n_mirrors)
    ref_dat.sigma_of_the_mean = {F: sigma}

    cov = model._differenced_measurement_covariance(F, ref_dat)[0]
    for i in range(N_DIFF - 1):
        shared_measurement = i + 1  # differences i and i+1 meet at measurement i+1
        assert cov[i, i + 1] == -(sigma[shared_measurement, 0] ** 2)


def test_measurement_covariance_handles_a_single_difference():
    """Two measurements give one difference and no off-diagonal to correct."""
    model = _constant_mean_model(np.full((1, T_GRID), 30.0))
    ref_dat = _ref_dat(1)
    ref_dat.prediction_indices = {F: [0, 5]}
    ref_dat.sigma_of_the_mean = {F: np.array([[1e-3], [2e-3]])}

    cov = model._differenced_measurement_covariance(F, ref_dat)
    assert cov.shape == (1, 1, 1)
    np.testing.assert_allclose(cov[0, 0, 0], 1e-6 + 4e-6, rtol=RTOL)
