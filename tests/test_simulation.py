"""
Tests for simulating reflectance datasets from the model.

``simulate_reflectance_data`` draws deposition from the generative process, accumulates
it into a reflectance path, samples that path and adds measurement noise. It does not
sample from ``_experiment_covariance``, and that is the point of the main test here: if
the simulator drew from the covariance the likelihood assembles, any error in the
assembly would appear in both and cancel, so a recovery study could not detect it.
Simulating generatively makes the assembly falsifiable, and the empirical covariance of
many simulated datasets is then an independent check on it.

That check also reaches the general per-mirror-tilt assembly path, which the Kronecker
comparison in test_variance_components.py does not: it only covers the common-tilt case.

1. ``test_covariance_of_simulated_data_*`` -- the anti-circularity check.
2. ``test_simulated_*`` -- shapes, reproducibility, and the missing-data path.
"""

import types
import numpy as np
import pytest

from heliosoil.fitting import ConstantMeanDeposition
from heliosoil.base_models import ConstantMeanBase, ReflectanceMeasurements


RTOL = 1e-10
ATOL = 0.0

F = 0
NOMINAL_REFLECTANCE = 0.95
MU_TILDE = 0.002
SIGMA_DEP = 6e-4
DENSITY = 40.0
MEASUREMENT_SIGMA = 1.2e-3
NUMBER_OF_MEASUREMENTS = 4.0
INCIDENCE_ANGLE = 15.0

T_GRID = 13
MEASUREMENT_INDICES = [0, 3, 7, 12]
N_DIFF = len(MEASUREMENT_INDICES) - 1
N_MIRRORS = 3


def _model(seed=0):
    """A ConstantMeanDeposition with distinct per-mirror tilts (the general path)."""
    rng = np.random.default_rng(seed)
    model = ConstantMeanDeposition.__new__(ConstantMeanDeposition)
    ConstantMeanBase.__init__(model)
    model.set_variance_model("components")
    model.mu_tilde, model.sigma_dep = MU_TILDE, SIGMA_DEP
    model.helios.nominal_reflectance = NOMINAL_REFLECTANCE
    model.helios.tilt = {F: rng.uniform(0.0, 60.0, size=(N_MIRRORS, T_GRID))}
    model.helios.incidence_angle = {F: INCIDENCE_ANGLE}
    model.helios.inc_ref_factor = {F: np.array(2.0 / np.cos(np.radians(INCIDENCE_ANGLE)))}
    for name in ("delta_soiled_area", "delta_soiled_area_variance",
                 "soiling_factor", "soiling_factor_prediction_variance"):
        setattr(model.helios, name, {})
    return model


def _sim_in():
    return types.SimpleNamespace(
        files=["experiment_0"],
        time={F: np.arange(T_GRID, dtype=float)},
        dust=types.SimpleNamespace(PM10={F: DENSITY}),
        dust_concentration={F: np.linspace(25.0, 85.0, T_GRID)},
        dust_type={F: "PM10"},
    )


def _simulate(model, sim_in, kappa, rng, missing_fraction=0.0,
              measurement_sigma=MEASUREMENT_SIGMA, n_measurements=NUMBER_OF_MEASUREMENTS):
    return model.simulate_reflectance_data(
        sim_in,
        [MU_TILDE, SIGMA_DEP, kappa],
        MEASUREMENT_INDICES,
        measurement_sigma,
        number_of_measurements=n_measurements,
        rng=rng,
        missing_fraction=missing_fraction,
    )


# ---------------------------------------------------------------------------
# 1. The covariance of simulated data matches the assembled covariance
# ---------------------------------------------------------------------------


def test_covariance_of_simulated_data_matches_the_assembly():
    """Sample covariance of the reflectance differences converges to _experiment_covariance.

    The simulator never touches _experiment_covariance, so agreement is evidence the
    assembly is right rather than a tautology. Mirror-major ordering throughout, matching
    the likelihood.
    """
    model, sim_in = _model(), _sim_in()
    kappa = 0.4
    # Chosen so the cross-mirror covariance is resolved several standard errors clear of
    # the tolerance; the assertion at the end of this test checks that it is.
    n_draws = 20000
    rng = np.random.default_rng(20240804)

    differences = np.empty((n_draws, N_MIRRORS * N_DIFF))
    for draw in range(n_draws):
        data = _simulate(model, sim_in, kappa, rng)
        differences[draw] = np.diff(data.average[F], axis=0).transpose().ravel()

    expected = model._experiment_covariance(
        F, SIGMA_DEP, kappa, sim_in, _simulate(model, sim_in, kappa, rng)
    )
    empirical = np.cov(differences, rowvar=False)

    # The standard error of a covariance entry is at most sqrt(2 S_ii S_jj / n), so
    # scale the tolerance by the largest variance on the diagonal.
    tolerance = 6 * np.sqrt(2.0) * np.max(np.diag(expected)) / np.sqrt(n_draws)
    np.testing.assert_allclose(empirical, expected, atol=tolerance, rtol=0.0)

    # Guard against the tolerance being so loose that the test cannot fail: the
    # cross-mirror blocks must be resolved well clear of it.
    cross_mirror = expected[0:N_DIFF, N_DIFF : 2 * N_DIFF]
    assert np.max(np.abs(cross_mirror)) > 5 * tolerance


def test_covariance_of_simulated_data_is_diagonal_without_a_common_component():
    """At kappa = 0 the simulated mirrors are uncorrelated, as the assembly says."""
    model, sim_in = _model(), _sim_in()
    n_draws = 4000
    rng = np.random.default_rng(11)

    differences = np.empty((n_draws, N_MIRRORS * N_DIFF))
    for draw in range(n_draws):
        data = _simulate(model, sim_in, 0.0, rng)
        differences[draw] = np.diff(data.average[F], axis=0).transpose().ravel()

    expected = model._experiment_covariance(
        F, SIGMA_DEP, 0.0, sim_in, _simulate(model, sim_in, 0.0, rng)
    )
    empirical = np.cov(differences, rowvar=False)
    tolerance = 6 * np.sqrt(2.0) * np.max(np.diag(expected)) / np.sqrt(n_draws)

    np.testing.assert_allclose(empirical, expected, atol=tolerance, rtol=0.0)

    cross_mirror = empirical[0:N_DIFF, N_DIFF : 2 * N_DIFF]
    assert np.max(np.abs(cross_mirror)) < tolerance

    # The tolerance must be tight enough for the absence of correlation to mean
    # something: a moderate kappa would put the cross-mirror block well above it.
    with_common = model._experiment_covariance(
        F, SIGMA_DEP, 0.4, sim_in, _simulate(model, sim_in, 0.0, rng)
    )
    assert np.max(np.abs(with_common[0:N_DIFF, N_DIFF : 2 * N_DIFF])) > 2 * tolerance


def test_covariance_of_simulated_data_shows_the_shared_endpoint_correlation():
    """Consecutive differences are negatively correlated through their shared measurement.

    Run in a measurement-noise-dominated regime on purpose. In the low-noise regime of
    the test above, the endpoint term is -sigma_of_the_mean^2, which sits below the Monte
    Carlo tolerance, so that test cannot see it and dropping the off-diagonal escapes
    detection. Here the measurement noise is large enough for the term to be resolved.
    """
    model, sim_in = _model(), _sim_in()
    kappa, n_draws = 0.3, 8000
    loud_sigma = 8e-3
    rng = np.random.default_rng(31)

    differences = np.empty((n_draws, N_MIRRORS * N_DIFF))
    for draw in range(n_draws):
        data = _simulate(
            model, sim_in, kappa, rng, measurement_sigma=loud_sigma, n_measurements=1.0
        )
        differences[draw] = np.diff(data.average[F], axis=0).transpose().ravel()

    expected = model._experiment_covariance(
        F, SIGMA_DEP, kappa, sim_in,
        _simulate(model, sim_in, kappa, rng, measurement_sigma=loud_sigma, n_measurements=1.0),
    )
    empirical = np.cov(differences, rowvar=False)
    tolerance = 6 * np.sqrt(2.0) * np.max(np.diag(expected)) / np.sqrt(n_draws)

    np.testing.assert_allclose(empirical, expected, atol=tolerance, rtol=0.0)

    # Within one mirror, consecutive differences share the measurement at their common
    # endpoint, giving a covariance of -sigma_of_the_mean^2.
    within_mirror = np.array([empirical[i, i + 1] for i in range(N_DIFF - 1)])
    np.testing.assert_allclose(within_mirror, -(loud_sigma**2), atol=tolerance)
    assert loud_sigma**2 > 4 * tolerance  # resolved well clear of the noise floor


# ---------------------------------------------------------------------------
# 2. The simulated object
# ---------------------------------------------------------------------------


def test_simulated_data_is_a_usable_reflectance_object():
    model, sim_in = _model(), _sim_in()
    data = _simulate(model, sim_in, 0.3, np.random.default_rng(2))

    assert isinstance(data, ReflectanceMeasurements)
    assert data.average[F].shape == (len(MEASUREMENT_INDICES), N_MIRRORS)
    assert list(data.prediction_indices[F]) == MEASUREMENT_INDICES
    assert data.tilts[F].shape == (N_MIRRORS, T_GRID)
    assert data.reflectometer_incidence_angle[F] == INCIDENCE_ANGLE
    np.testing.assert_allclose(
        data.sigma_of_the_mean[F],
        MEASUREMENT_SIGMA / np.sqrt(NUMBER_OF_MEASUREMENTS),
        rtol=RTOL,
    )
    # Reflectance declines from a clean start and stays physical.
    assert np.all(data.average[F] < 1.0)
    assert data.average[F][-1].mean() < data.average[F][0].mean()


def test_simulated_data_can_be_fitted():
    """The likelihood accepts the simulated object and returns a finite value."""
    model, sim_in = _model(), _sim_in()
    data = _simulate(model, sim_in, 0.3, np.random.default_rng(5))

    value = model._negative_log_likelihood([MU_TILDE, SIGMA_DEP, 0.3], sim_in, data)
    assert np.isfinite(value)


def test_simulated_data_is_reproducible():
    model, sim_in = _model(), _sim_in()
    first = _simulate(model, sim_in, 0.3, np.random.default_rng(4)).average[F]
    same = _simulate(model, sim_in, 0.3, np.random.default_rng(4)).average[F]
    other = _simulate(model, sim_in, 0.3, np.random.default_rng(5)).average[F]

    np.testing.assert_allclose(first, same, rtol=RTOL, atol=ATOL)
    assert not np.allclose(first, other)


# ---------------------------------------------------------------------------
# 3. Missing data
# ---------------------------------------------------------------------------


def test_missing_measurements_appear_at_the_requested_rate():
    model, sim_in = _model(), _sim_in()
    rng = np.random.default_rng(9)

    missing, n_repeats = 0.0, 200
    for _ in range(n_repeats):
        data = _simulate(model, sim_in, 0.3, rng, missing_fraction=0.25)
        missing += np.isnan(data.average[F]).mean()

    n_values = n_repeats * data.average[F].size
    np.testing.assert_allclose(
        missing / n_repeats, 0.25, atol=5 * np.sqrt(0.25 * 0.75 / n_values)
    )


def test_missing_measurements_are_marginalised_by_the_likelihood():
    """Gaps drop out of the likelihood rather than poisoning it.

    Sampled until a dataset has gaps but no mirror lost entirely: an all-NaN column
    would make rho0 a NaN via nanmax, which is a separate concern from marginalisation.
    """
    model, sim_in = _model(), _sim_in()
    rng = np.random.default_rng(9)

    for _ in range(50):
        data = _simulate(model, sim_in, 0.3, rng, missing_fraction=0.2)
        gaps = np.isnan(data.average[F])
        if gaps.any() and not gaps.all(axis=0).any():
            break
    else:
        raise AssertionError("no suitable dataset with partial gaps was generated")

    complete = model._negative_log_likelihood([MU_TILDE, SIGMA_DEP, 0.3], sim_in, data)
    assert np.isfinite(complete)


def test_missing_fraction_is_validated():
    model, sim_in = _model(), _sim_in()
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="missing_fraction"):
            _simulate(model, sim_in, 0.3, np.random.default_rng(1), missing_fraction=bad)
