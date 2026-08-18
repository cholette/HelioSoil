"""
Tests for the generative side: drawing deposition, and simulating a fittable dataset.

``random_delta_soiled_area`` draws, per mechanism, one standard normal per interval shared
across mirrors and one independent per (mirror, interval), scaled by that mechanism's own
loading:

    delta[p, j] = mean[p, j]
                + sum_c g_c[p, j] * sigma_c * ( sqrt(kappa_c) * eta_c[j]
                                                + sqrt(1 - kappa_c) * xi_c[p, j] )

The centrepiece here is ``test_empirical_covariance_matches_the_analytic_one``: the
covariance of many simulated reflectance-difference datasets, compared against
``_experiment_covariance``. The two are written independently -- one draws random variables,
the other sums second moments -- so their agreement is evidence that both are right, not a
tautology. Everything else in this file exists to make that comparison trustworthy: that the
draw has the right mean, that the shared and independent parts land where they should, and
that the seed controls the result.
"""

import numpy as np
import pytest

from test_variance_components import (
    F,
    MEASUREMENT_SIGMA,
    N_DIFF,
    PREDICTION_INDICES,
    T_GRID,
    _parameters,
    _ref_dat,
    _sim_in,
    _weather,
    _wind_model,
)

RTOL = 1e-11

TILTS = [10.0, 55.0, 100.0]
AZIMUTHS = [0.0, 130.0, 250.0]


def _case(kappa=0.4, variance_model="shared_kappa", tilts=TILTS, azimuths=AZIMUTHS):
    concentration, speed, direction = _weather()
    tilt = np.array([[t] * T_GRID for t in tilts])
    azimuth = np.array([[a] * T_GRID for a in azimuths])
    model = _wind_model(tilt, azimuth)
    model.set_variance_model(variance_model, kappa=kappa, endpoint_correction=True)
    if variance_model == "per_mechanism":
        model.kappa_dep = model.kappa_dep_gamma = kappa
    return model, _sim_in(concentration, speed, direction), _ref_dat(len(tilts))


# ---------------------------------------------------------------------------
# The draw itself
# ---------------------------------------------------------------------------


def test_the_draw_is_centred_on_calculate_delta_soiled_area():
    """Averaging many draws must return the deterministic mean, mechanism noise being
    zero-mean by construction."""
    model, sim_in, _ = _case()
    model.calculate_delta_soiled_area(sim_in, verbose=False)
    mean = np.array(model.helios.delta_soiled_area[F], dtype=float)

    rng = np.random.default_rng(0)
    draws = np.array([model.random_delta_soiled_area(sim_in, rng=rng, verbose=False)[F] for _ in range(4000)])
    standard_error = draws.std(axis=0) / np.sqrt(draws.shape[0])
    # Every entry within four standard errors of the deterministic mean.
    assert np.all(np.abs(draws.mean(axis=0) - mean) <= 4.0 * standard_error + 1e-18)


def test_the_draw_has_the_variance_the_diagonal_likelihood_assumes():
    """Per (mirror, interval), the variance must be sum_c sigma_c**2 g_c**2 -- exactly what
    calculate_delta_soiled_area records and _compute_variance_of_measurements accumulates."""
    model, sim_in, _ = _case()
    model.calculate_delta_soiled_area(sim_in, verbose=False)
    expected = np.array(model.helios.delta_soiled_area_variance[F], dtype=float)

    rng = np.random.default_rng(1)
    draws = np.array([model.random_delta_soiled_area(sim_in, rng=rng, verbose=False)[F] for _ in range(6000)])
    observed = draws.var(axis=0)
    # A variance estimated from n draws has relative standard error sqrt(2/n) ~ 1.8%.
    np.testing.assert_allclose(observed[expected > 0.0], expected[expected > 0.0], rtol=0.12)


@pytest.mark.parametrize("kappa,correlated", [(0.0, False), (1.0, True)])
def test_kappa_controls_whether_mirrors_move_together(kappa, correlated):
    """kappa = 0 makes mirrors independent; kappa = 1 makes a mechanism's noise wholly
    shared, so two mirrors it loads move in lockstep."""
    # Both mirrors at the same tilt and azimuth, so a mechanism loads them identically.
    model, sim_in, _ = _case(kappa=kappa, tilts=[40.0, 40.0], azimuths=[70.0, 70.0])
    rng = np.random.default_rng(2)
    draws = np.array([model.random_delta_soiled_area(sim_in, rng=rng, verbose=False)[F] for _ in range(4000)])

    interval = T_GRID // 2
    a, b = draws[:, 0, interval], draws[:, 1, interval]
    correlation = np.corrcoef(a, b)[0, 1]
    if correlated:
        assert correlation > 0.95
    else:
        assert abs(correlation) < 0.06


def test_a_mechanism_draws_no_noise_for_a_mirror_it_does_not_load():
    """A face-down mirror feels no gravitational settling, so none of that mechanism's
    noise -- common part included -- may reach it."""
    model, sim_in, _ = _case(kappa=1.0, tilts=[0.0, 120.0], azimuths=[10.0, 200.0])
    model.sigma_dep_gamma = None  # gravitational only
    model.calculate_delta_soiled_area(sim_in, verbose=False)
    mean = np.array(model.helios.delta_soiled_area[F], dtype=float)

    rng = np.random.default_rng(3)
    for _ in range(50):
        drawn = model.random_delta_soiled_area(sim_in, rng=rng, verbose=False)[F]
        np.testing.assert_array_equal(drawn[1, :], mean[1, :])  # exactly the mean, no noise
    assert not np.allclose(drawn[0, :], mean[0, :])  # the horizontal one does get noise


def test_the_generator_makes_the_draw_reproducible():
    model, sim_in, _ = _case()
    a = model.random_delta_soiled_area(sim_in, rng=np.random.default_rng(7), verbose=False)[F]
    b = model.random_delta_soiled_area(sim_in, rng=np.random.default_rng(7), verbose=False)[F]
    c = model.random_delta_soiled_area(sim_in, rng=np.random.default_rng(8), verbose=False)[F]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_drawing_without_any_magnitude_set_is_an_error():
    """A least-squares fit leaves every sigma unset; simulating from it would silently
    return the deterministic mean."""
    model, sim_in, _ = _case()
    model.sigma_dep = model.sigma_dep_gamma = None
    with pytest.raises(ValueError, match="carries a noise magnitude"):
        model.random_delta_soiled_area(sim_in, rng=np.random.default_rng(0), verbose=False)


# ---------------------------------------------------------------------------
# The centrepiece: generative vs analytic
# ---------------------------------------------------------------------------


def _difference_residual_samples(model, sim_in, ref_dat, draws, seed):
    """Many draws of the mirror-major stacked reflectance differences, minus their mean.

    Built the way simulate_reflectance_data builds a dataset -- accumulate the deposition
    into a path, sample it, add measurement noise, difference -- but returning the residual
    directly, so nothing depends on the ReflectanceMeasurements plumbing.
    """
    rng = np.random.default_rng(seed)
    b = model._reflectance_loss_factor(F, ref_dat)
    n_mirrors = len(b)
    samples = np.empty((draws, n_mirrors * N_DIFF))

    sigma_of_the_mean = np.asarray(ref_dat.sigma_of_the_mean[F])
    for draw in range(draws):
        deposited = model.random_delta_soiled_area(sim_in, rng=rng, verbose=False)[F]
        path = -b[:, None] * np.cumsum(deposited, axis=1)  # (n_mirrors, n_times)
        sampled = path[:, PREDICTION_INDICES].transpose()  # (n_measurements, n_mirrors)
        measured = sampled + sigma_of_the_mean * rng.standard_normal(sampled.shape)
        samples[draw] = np.diff(measured, axis=0).transpose().ravel()
    return samples - samples.mean(axis=0)


@pytest.mark.parametrize("kappa", [0.0, 0.5])
def test_empirical_covariance_matches_the_analytic_one(kappa):
    """The covariance of simulated datasets must equal _experiment_covariance.

    This is the check that ties the two halves of the model together. The generative path
    draws eta and xi and accumulates them; the analytic path sums second moments over
    mechanisms and windows. Neither is derived from the other, so agreement is evidence.

    Tolerance: a covariance entry estimated from n draws has standard error of order
    sqrt((C_ii C_jj + C_ij^2) / n), so entries are compared against that scale rather than
    against a fixed relative tolerance -- a near-zero entry has no meaningful relative error.
    """
    model, sim_in, ref_dat = _case(kappa=kappa)
    draws = 20000
    samples = _difference_residual_samples(model, sim_in, ref_dat, draws, seed=11)
    empirical = samples.transpose() @ samples / (draws - 1)

    analytic = model._experiment_covariance(F, sim_in, ref_dat, endpoint_correction=True)

    diagonal = np.diag(analytic)
    standard_error = np.sqrt((np.outer(diagonal, diagonal) + analytic**2) / draws)
    deviation = np.abs(empirical - analytic) / standard_error

    # No entry off by more than 5 standard errors, and the bulk well inside that.
    assert deviation.max() < 5.0, f"worst entry {deviation.max():.2f} standard errors out"
    assert np.median(deviation) < 1.0


def test_the_analytic_covariance_is_not_trivially_diagonal_in_the_test_case():
    """Guards the test above: if every off-diagonal were zero, it would pass without
    testing the mirror coupling at all."""
    model, sim_in, ref_dat = _case(kappa=0.5)
    analytic = model._experiment_covariance(F, sim_in, ref_dat, endpoint_correction=True)
    cross_mirror = analytic[:N_DIFF, N_DIFF : 2 * N_DIFF]
    assert np.abs(cross_mirror).max() > 0.05 * np.abs(np.diag(analytic)).max()


# ---------------------------------------------------------------------------
# simulate_reflectance_data
# ---------------------------------------------------------------------------


def test_simulate_reflectance_data_produces_a_dataset_the_likelihood_accepts():
    model, sim_in, _ = _case()
    params = _parameters(model) + [0.4]
    simulated = model.simulate_reflectance_data(
        sim_in,
        params,
        measurement_indices=PREDICTION_INDICES,
        measurement_sigma=MEASUREMENT_SIGMA,
        number_of_measurements=9.0,
        rng=np.random.default_rng(4),
    )

    assert simulated.average[F].shape == (len(PREDICTION_INDICES), len(TILTS))
    np.testing.assert_array_equal(simulated.prediction_indices[F], PREDICTION_INDICES)
    # sigma_of_the_mean is the single-reading sigma divided by sqrt(number_of_measurements).
    np.testing.assert_allclose(simulated.sigma_of_the_mean[F], MEASUREMENT_SIGMA / 3.0, rtol=RTOL)

    value = model._negative_log_likelihood(params, sim_in, simulated)
    assert np.isfinite(value)


def test_simulated_reflectance_falls_monotonically_when_the_noise_is_switched_off():
    """With every magnitude at zero the path is the deterministic mean, which can only
    fall: the mean deposition per interval is nonnegative."""
    model, sim_in, _ = _case()
    params = _parameters(model)
    params[3] = params[4] = 0.0  # sigma_dep, sigma_dep_gamma
    simulated = model.simulate_reflectance_data(
        sim_in,
        params + [0.4],
        measurement_indices=PREDICTION_INDICES,
        measurement_sigma=0.0,
        rng=np.random.default_rng(5),
    )
    average = simulated.average[F]
    assert np.all(np.diff(average, axis=0) <= 0.0)
    assert np.all(average[0, :] <= model.helios.nominal_reflectance)


def test_a_noisy_path_still_falls_overall_but_need_not_fall_every_step():
    """The deposition noise is Gaussian and untruncated, so a single interval's drawn
    deposition can be negative and the path can tick up. Only the trend is guaranteed.
    Worth pinning: it is the reason the model can predict reflectance recovery it does not
    physically mean, and the reason a monotonicity check would be the wrong assertion."""
    model, sim_in, _ = _case()
    simulated = model.simulate_reflectance_data(
        sim_in,
        _parameters(model) + [0.4],
        measurement_indices=PREDICTION_INDICES,
        measurement_sigma=0.0,
        rng=np.random.default_rng(5),
    )
    average = simulated.average[F]
    assert np.all(average[-1, :] < average[0, :])  # net soiling over the campaign
    assert np.any(np.diff(average, axis=0) > 0.0)  # but not step by step


def test_missing_fraction_blanks_measurements():
    model, sim_in, _ = _case()
    simulated = model.simulate_reflectance_data(
        sim_in,
        _parameters(model) + [0.4],
        measurement_indices=PREDICTION_INDICES,
        measurement_sigma=MEASUREMENT_SIGMA,
        missing_fraction=0.3,
        rng=np.random.default_rng(6),
    )
    blanked = np.isnan(simulated.average[F])
    assert blanked.any()
    assert not blanked.all()


@pytest.mark.parametrize("missing_fraction", [-0.1, 1.0, 1.5])
def test_missing_fraction_must_be_a_proper_fraction(missing_fraction):
    model, sim_in, _ = _case()
    with pytest.raises(ValueError, match=r"missing_fraction must be in \[0, 1\)"):
        model.simulate_reflectance_data(
            sim_in,
            _parameters(model) + [0.4],
            measurement_indices=PREDICTION_INDICES,
            measurement_sigma=MEASUREMENT_SIGMA,
            missing_fraction=missing_fraction,
        )


def test_simulate_reflectance_data_is_reproducible_from_its_generator():
    model, sim_in, _ = _case()
    params = _parameters(model) + [0.4]

    def simulate(seed):
        return model.simulate_reflectance_data(
            sim_in,
            params,
            measurement_indices=PREDICTION_INDICES,
            measurement_sigma=MEASUREMENT_SIGMA,
            rng=np.random.default_rng(seed),
        ).average[F]

    np.testing.assert_array_equal(simulate(12), simulate(12))
    assert not np.allclose(simulate(12), simulate(13))
