"""
Tests for the random deposition draw shared by both soiling models.

Mirrors at one site do not receive independent deposition: a dusty or gusty interval
loads all of them at once. ``SoilingBase.random_delta_soiled_area`` therefore splits the
noise into a component common to every mirror during an interval and a mirror-specific
component, with ``kappa`` the common fraction of the variance.

The draw is generative -- ``eta`` and ``xi`` are sampled and propagated -- rather than
taken from an assembled covariance. That is deliberate: it lets the simulated data serve
as an independent check on the fitting covariance, which would be impossible if both came
from the same matrix.

1. ``test_structure_*`` -- what ``kappa`` does to the correlation between mirrors.
2. ``test_reduces_*`` -- ``kappa = 0`` is the independent case.
3. ``test_reproducible_*`` / ``test_rejects_*`` -- seeding and error handling.
"""

import types
import numpy as np
import pytest

from heliosoil.base_models import ConstantMeanBase, PhysicalBase


RTOL = 1e-10
ATOL = 0.0

F = 0
MU_TILDE = 0.003
SIGMA_DEP = 5e-4
DENSITY = 40.0
N_MIRRORS = 4
T_GRID = 12


def _model(sigma_dep=SIGMA_DEP, seed=0):
    """A ConstantMeanBase with tilts and dust set directly."""
    rng = np.random.default_rng(seed)
    model = ConstantMeanBase()
    model.mu_tilde, model.sigma_dep = MU_TILDE, sigma_dep
    model.helios.tilt = {F: rng.uniform(0.0, 60.0, size=(N_MIRRORS, T_GRID))}
    model.helios.delta_soiled_area = {}
    model.helios.delta_soiled_area_variance = {}
    return model


def _sim_in():
    return types.SimpleNamespace(
        files=["experiment_0"],
        time={F: np.arange(T_GRID, dtype=float)},
        dust=types.SimpleNamespace(PM10={F: DENSITY}),
        dust_concentration={F: np.linspace(20.0, 90.0, T_GRID)},
        dust_type={F: "PM10"},
    )


def _draws(model, sim_in, kappa, n_draws, seed=11):
    """n_draws simulated deposition arrays, shape (n_draws, n_mirrors, n_times)."""
    rng = np.random.default_rng(seed)
    return np.array(
        [
            model.random_delta_soiled_area(sim_in, kappa=kappa, rng=rng, verbose=False)[F]
            for _ in range(n_draws)
        ]
    )


# ---------------------------------------------------------------------------
# 1. Correlation structure
# ---------------------------------------------------------------------------


def test_structure_matches_the_variance_split():
    """Per-mirror variance is sigma_dep^2 alpha^2 cos^2 regardless of kappa.

    Splitting the variance must not change the total: kappa moves variance between the
    common and mirror-specific channels, it does not add any.
    """
    model, sim_in = _model(), _sim_in()
    n_draws = 40000

    model.calculate_delta_soiled_area(sim_in, verbose=False)
    expected_variance = model.helios.delta_soiled_area_variance[F]
    expected_mean = model.helios.delta_soiled_area[F]

    # Monte Carlo standard errors: var * sqrt(2/n) for a variance, sd/sqrt(n) for a mean.
    variance_tolerance = 8 * np.sqrt(2 / n_draws)
    mean_tolerance = 8 * np.sqrt(expected_variance.max() / n_draws)

    for kappa in (0.0, 0.3, 1.0):
        draws = _draws(model, sim_in, kappa, n_draws)
        np.testing.assert_allclose(draws.var(axis=0), expected_variance, rtol=variance_tolerance)
        np.testing.assert_allclose(draws.mean(axis=0), expected_mean, atol=mean_tolerance)


def test_structure_correlates_mirrors_in_proportion_to_kappa():
    """Corr(mirror p, mirror q) within an interval is kappa; across intervals it is zero."""
    model, sim_in = _model(), _sim_in()
    n_draws = 40000

    for kappa in (0.0, 0.35, 0.8):
        draws = _draws(model, sim_in, kappa, n_draws)
        centred = draws - draws.mean(axis=0)
        scaled = centred / centred.std(axis=0)

        # same interval, different mirrors
        same_interval = np.mean(scaled[:, 0, :] * scaled[:, 1, :], axis=0)
        # different intervals, different mirrors
        across = np.mean(scaled[:, 0, :-1] * scaled[:, 1, 1:], axis=0)

        error = 4 / np.sqrt(n_draws)
        np.testing.assert_allclose(same_interval, kappa, atol=error)
        np.testing.assert_allclose(across, 0.0, atol=error)


def test_structure_removes_contrast_noise_at_kappa_one():
    """At kappa = 1 the deposition is perfectly common, so mirror contrasts vanish."""
    model, sim_in = _model(), _sim_in()
    draws = _draws(model, sim_in, kappa=1.0, n_draws=64)

    model.calculate_delta_soiled_area(sim_in, verbose=False)
    scale = np.sqrt(model.helios.delta_soiled_area_variance[F])
    mean = model.helios.delta_soiled_area[F]

    # (delta - mean)/scale is the same standard normal for every mirror.
    standardised = (draws - mean) / scale
    np.testing.assert_allclose(standardised[:, 0, :], standardised[:, 3, :], rtol=1e-9)


# ---------------------------------------------------------------------------
# 2. Reduction to the independent case
# ---------------------------------------------------------------------------


def test_reduces_to_independent_draws_at_kappa_zero():
    """kappa = 0 leaves the mirrors uncorrelated."""
    model, sim_in = _model(), _sim_in()
    n_draws = 40000
    draws = _draws(model, sim_in, kappa=0.0, n_draws=n_draws)

    centred = draws - draws.mean(axis=0)
    scaled = centred / centred.std(axis=0)
    for p in range(1, N_MIRRORS):
        correlation = np.mean(scaled[:, 0, :] * scaled[:, p, :], axis=0)
        np.testing.assert_allclose(correlation, 0.0, atol=4 / np.sqrt(n_draws))


def test_kappa_defaults_to_the_models_variance_split():
    """Omitting kappa uses common_variance_fraction, which is zero by default."""
    model, sim_in = _model(), _sim_in()
    assert model.common_variance_fraction == 0.0

    without = model.random_delta_soiled_area(sim_in, rng=np.random.default_rng(3), verbose=False)
    explicit = model.random_delta_soiled_area(
        sim_in, kappa=0.0, rng=np.random.default_rng(3), verbose=False
    )
    np.testing.assert_allclose(without[F], explicit[F], rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 3. Seeding and error handling
# ---------------------------------------------------------------------------


def test_reproducible_given_a_generator():
    model, sim_in = _model(), _sim_in()
    kwargs = dict(kappa=0.4, verbose=False)

    first = model.random_delta_soiled_area(sim_in, rng=np.random.default_rng(7), **kwargs)[F]
    same = model.random_delta_soiled_area(sim_in, rng=np.random.default_rng(7), **kwargs)[F]
    other = model.random_delta_soiled_area(sim_in, rng=np.random.default_rng(8), **kwargs)[F]

    np.testing.assert_allclose(first, same, rtol=RTOL, atol=ATOL)
    assert not np.allclose(first, other)


def test_rejects_out_of_range_kappa():
    model, sim_in = _model(), _sim_in()
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            model.random_delta_soiled_area(sim_in, kappa=bad, verbose=False)


def test_rejects_missing_sigma_dep():
    """Without a noise scale there is nothing to draw."""
    model, sim_in = _model(sigma_dep=None), _sim_in()
    with pytest.raises(ValueError, match="sigma_dep"):
        model.random_delta_soiled_area(sim_in, verbose=False)


def test_physical_model_reports_an_unprepared_state():
    """PhysicalBase needs deposition_flux and adhesion_removal run first.

    ConstantMeanBase computes the variance inside calculate_delta_soiled_area, but
    PhysicalBase needs the extinction weighting and particle distribution first. Reaching
    the draw without them should say so rather than fail deep in an array expression.
    """
    model = PhysicalBase()
    model.sigma_dep = SIGMA_DEP
    model.helios.tilt = {F: np.zeros((2, T_GRID))}
    model.helios.delta_soiled_area = {}
    model.helios.delta_soiled_area_variance = {}

    with pytest.raises((ValueError, AttributeError, KeyError)) as info:
        model.random_delta_soiled_area(_sim_in(), verbose=False)

    # The useful outcome is a diagnosable failure, not a silent empty result.
    assert str(info.value) != ""
