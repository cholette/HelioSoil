"""
Tests for the across-mirror measurement covariance (_covariance_of_measurements) and the
likelihood built on it.

The deposition fluctuation is ONE draw per timestep for the whole site, so the reflectance
increments of two mirrors measured over the same interval are correlated. The likelihood
used to be written on the diagonal of that covariance, i.e. as if the mirrors were
independent, which counts N_helios independent observations where the data carry roughly
one.

Two things are pinned here:
  - the change is conservative on the diagonal -- the marginal per-mirror variance is
    exactly what the old cumsum-difference form produced (test_diagonal_*), so anything
    reading a single mirror's prediction interval is unaffected;
  - the off-diagonal is real and has the structure the model implies (rank 1 for a
    tilt-only model), and the likelihood built on it beats the diagonal one at recovering
    a genuinely shared sigma (test_shared_noise_*).
"""

import types

import numpy as np
import pytest

import heliosoil.fitting as smf
from heliosoil.horizontal_impaction import ConstantMeanWindBase, ConstantMeanWindDeposition, _COMPONENTS, COMPONENT_KEYS
from heliosoil.utilities import gravitational_settling_factor

RTOL = 1e-10


def _helios_stub(**arrays):
    helios = types.SimpleNamespace(delta_soiled_area={}, delta_soiled_area_variance={})
    for name, value in arrays.items():
        setattr(helios, name, value)
    return helios


def _fixture(n_helios=4, n_times=240, meas_every=24, seed=3, tilt_values=None):
    """A small multi-mirror experiment: tilts spread over the upward-facing range, one
    weather series, measurements every `meas_every` timesteps."""
    rng = np.random.default_rng(seed)
    f = 0
    if tilt_values is None:
        tilt_values = np.linspace(0.0, 75.0, n_helios)
    tilt = np.tile(np.asarray(tilt_values, dtype=float)[:, None], (1, n_times))
    azimuth = np.tile(np.linspace(0.0, 270.0, n_helios)[:, None], (1, n_times))

    density = 40.0
    nominal_reflectance = 0.95
    inc_ref_factor = np.float64(2.0 / np.cos(np.radians(15.0)))

    sim_in = types.SimpleNamespace(
        time={f: np.arange(n_times)},
        files=[f],
        dust=types.SimpleNamespace(TSP={f: density}),
        dust_concentration={f: rng.uniform(20.0, 80.0, size=n_times)},
        dust_type={f: "TSP"},
        wind_direction={f: rng.uniform(0.0, 360.0, size=n_times)},
        wind_speed={f: rng.uniform(1.0, 6.0, size=n_times)},
    )

    pred_idx = np.arange(meas_every - 1, n_times, meas_every)
    n_meas = len(pred_idx)
    meas_noise_std = 5e-4
    ref_dat = types.SimpleNamespace(
        files=[f],
        times={f: pred_idx},
        prediction_indices={f: list(pred_idx)},
        average={f: np.full((n_meas, n_helios), nominal_reflectance)},
        sigma_of_the_mean={f: np.full((n_meas, n_helios), meas_noise_std)},
        rho0={f: np.full(n_helios, nominal_reflectance)},
    )

    helios = _helios_stub(
        tilt={f: tilt},
        azimuth={f: azimuth},
        inc_ref_factor={f: inc_ref_factor},
        nominal_reflectance=nominal_reflectance,
        soiling_factor={},
        soiling_factor_prediction_variance={},
    )
    return f, sim_in, ref_dat, helios, tilt, azimuth


def _legacy_diagonal(model, sigmas, sim_in, ref_dat, f, variance_bases):
    """The variance the pre-covariance code computed, re-derived here rather than imported.

        s2total = b^2 * sum_k sigma_k^2 * (cumsum(basis_k)[i2] - cumsum(basis_k)[i1])
                  + meas_sig[:-1]^2 + meas_sig[1:]^2

    `variance_bases` maps sigma parameter name -> (N_helios, N_times) SQUARED basis.
    """
    pif = ref_dat.prediction_indices[f]
    b = model.helios.nominal_reflectance * model.helios.inc_ref_factor[f]
    meas_sig = ref_dat.sigma_of_the_mean[f]
    ind1 = pif[0:-1]
    ind2 = [x - 1 for x in pif[1::]]

    total = np.zeros((len(ind1), model.helios.tilt[f].shape[0]))
    for name, basis in variance_bases.items():
        sigma = sigmas[name]
        if sigma is None:
            continue
        c2t = np.cumsum(basis, axis=1).transpose()
        total = total + sigma**2 * (c2t[ind2, :] - c2t[ind1, :])
    return b**2 * total + meas_sig[0:-1, :] ** 2 + meas_sig[1::, :] ** 2


# ---------------------------------------------------------------------------
# 1. Conservative reduction: the diagonal is exactly what it always was
# ---------------------------------------------------------------------------


def test_diagonal_matches_legacy_formula_constant_mean():
    """The marginal per-mirror variance of a tilt-only model is unchanged."""
    f, sim_in, ref_dat, helios, tilt, _ = _fixture()
    model = smf.ConstantMeanDeposition.__new__(smf.ConstantMeanDeposition)
    model.helios = helios
    model.mu_tilde, model.sigma_dep = 4.0e-4, 2.0e-4

    alpha = sim_in.dust_concentration[f] / sim_in.dust.TSP[f]
    legacy = _legacy_diagonal(
        model, {"sigma_dep": model.sigma_dep}, sim_in, ref_dat, f, {"sigma_dep": (alpha[None, :] * gravitational_settling_factor(tilt)) ** 2}
    )

    cov = model._covariance_of_measurements(sim_in, ref_dat, {"sigma_dep": model.sigma_dep})
    np.testing.assert_allclose(np.diagonal(cov[f], axis1=1, axis2=2), legacy, rtol=RTOL)

    # and the public marginal-variance accessor agrees with it too
    marginal = model._compute_variance_of_measurements(model.sigma_dep, sim_in, reflectance_data=ref_dat)
    np.testing.assert_allclose(marginal[f], legacy, rtol=RTOL)


@pytest.mark.parametrize(
    "components", [["gravitational"], ["gravitational", "normal_wind"], ["gravitational", "impaction_retention"], list(COMPONENT_KEYS)]
)
def test_diagonal_matches_legacy_formula_wind_models(components):
    """Same reduction for every wind-mechanism combination, whose covariance sums one Gram
    matrix per active mechanism."""
    f, sim_in, ref_dat, helios, tilt, azimuth = _fixture()
    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanWindBase.__init__(model, components=components)
    model.helios = helios

    sigmas = {name: 1.5e-4 * (i + 1) for i, name in enumerate(model._sigma_param_names)}
    for name, value in sigmas.items():
        setattr(model, name, value)

    alpha = sim_in.dust_concentration[f] / sim_in.dust.TSP[f]
    variance_bases = {
        _COMPONENTS[key].sigma_param_name: _COMPONENTS[key].variance_basis(alpha, tilt, azimuth, sim_in.wind_direction[f], sim_in.wind_speed[f])
        for key in model._component_keys
    }
    legacy = _legacy_diagonal(model, sigmas, sim_in, ref_dat, f, variance_bases)

    cov = model._covariance_of_measurements(sim_in, ref_dat, model._current_sigmas())
    np.testing.assert_allclose(np.diagonal(cov[f], axis1=1, axis2=2), legacy, rtol=RTOL)


def test_inactive_sigma_contributes_nothing():
    """A None sigma (an un-estimated mechanism, as after a least-squares fit) drops out,
    leaving only measurement noise -- matching how the old form skipped it."""
    f, sim_in, ref_dat, helios, _, _ = _fixture()
    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanWindBase.__init__(model, components=["gravitational", "normal_wind"])
    model.helios = helios
    model.sigma_dep = None
    model.sigma_dep_gamma = None

    cov = model._covariance_of_measurements(sim_in, ref_dat, model._current_sigmas())[f]
    meas_sig = ref_dat.sigma_of_the_mean[f]
    expected = meas_sig[0:-1, :] ** 2 + meas_sig[1::, :] ** 2
    np.testing.assert_allclose(np.diagonal(cov, axis1=1, axis2=2), expected, rtol=RTOL)
    off = cov - np.stack([np.diag(np.diag(c)) for c in cov])
    np.testing.assert_allclose(off, 0.0, atol=0.0)


# ---------------------------------------------------------------------------
# 2. noise_basis is the square root of variance_basis, by construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", COMPONENT_KEYS)
def test_variance_basis_is_noise_basis_squared(key):
    """variance_basis is derived from noise_basis, so the two can never disagree -- the
    property the covariance relies on when it forms B B^T from the unsquared basis."""
    f, sim_in, _, _, tilt, azimuth = _fixture()
    component = _COMPONENTS[key]
    alpha = sim_in.dust_concentration[f] / sim_in.dust.TSP[f]
    args = (alpha, tilt, azimuth, sim_in.wind_direction[f], sim_in.wind_speed[f])
    np.testing.assert_allclose(component.variance_basis(*args), component.noise_basis(*args) ** 2, rtol=RTOL)


# ---------------------------------------------------------------------------
# 3. The off-diagonal is real, and has the structure the model implies
# ---------------------------------------------------------------------------


def test_tilt_only_deposition_covariance_is_rank_one():
    """For a tilt-only model the noise basis factorizes as cos(tilt) x alpha, so the
    deposition part of Sigma is exactly rank 1: every pair of mirrors perfectly correlated.
    This is the structure the diagonal form was discarding."""
    f, sim_in, ref_dat, helios, _, _ = _fixture()
    model = smf.ConstantMeanDeposition.__new__(smf.ConstantMeanDeposition)
    model.helios = helios
    model.mu_tilde, model.sigma_dep = 4.0e-4, 2.0e-4

    with_noise = model._covariance_of_measurements(sim_in, ref_dat, {"sigma_dep": model.sigma_dep})[f]
    without = model._covariance_of_measurements(sim_in, ref_dat, {"sigma_dep": None})[f]
    deposition = with_noise - without  # strip the measurement diagonal

    for dep in deposition:
        eigenvalues = np.sort(np.linalg.eigvalsh(dep))[::-1]
        assert eigenvalues[0] > 0
        # rank 1: everything past the leading eigenvalue is numerical dust
        assert eigenvalues[1] / eigenvalues[0] < 1e-12
        corr = dep / np.sqrt(np.outer(np.diag(dep), np.diag(dep)))
        np.testing.assert_allclose(corr, 1.0, rtol=1e-9)


def test_covariance_is_symmetric_positive_definite():
    f, sim_in, ref_dat, helios, tilt, azimuth = _fixture()
    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanWindBase.__init__(model, components=["gravitational", "normal_wind"])
    model.helios = helios
    model.sigma_dep, model.sigma_dep_gamma = 2.0e-4, 1.0e-4

    for cov in model._covariance_of_measurements(sim_in, ref_dat, model._current_sigmas())[f]:
        np.testing.assert_allclose(cov, cov.T, rtol=RTOL)
        np.linalg.cholesky(cov)  # raises if not positive definite


def test_gaussian_loglike_reduces_to_the_diagonal_form():
    """With no deposition term the likelihood must reproduce the old
    independent-observations sum term for term."""
    rng = np.random.default_rng(7)
    n_int, n_helios = 6, 4
    variances = rng.uniform(1e-8, 1e-6, size=(n_int, n_helios))
    residuals = rng.normal(0.0, 1e-3, size=(n_int, n_helios))
    deposition = np.zeros((n_int, n_helios, n_helios))

    legacy = np.sum(-0.5 * np.log(variances) - residuals**2 / (2 * variances))
    assert smf.CommonFittingMethods._gaussian_loglike(residuals, deposition, variances) == pytest.approx(legacy, rel=1e-12)


def test_gaussian_loglike_matches_a_direct_dense_evaluation():
    """The whitened eigendecomposition must agree with a textbook dense evaluation of the
    Gaussian log-density on a well-conditioned case, where the dense form is trustworthy."""
    rng = np.random.default_rng(17)
    n_helios = 5
    root = rng.normal(size=(n_helios, n_helios))
    deposition = np.stack([root @ root.T * 1e-7])
    variances = np.full((1, n_helios), 3e-7)
    residuals = rng.normal(0.0, 1e-3, size=(1, n_helios))

    sigma = deposition[0] + np.diag(variances[0])
    sign, logdet = np.linalg.slogdet(sigma)
    assert sign > 0
    dense = -0.5 * (logdet + residuals[0] @ np.linalg.solve(sigma, residuals[0]))

    assert smf.CommonFittingMethods._gaussian_loglike(residuals, deposition, variances) == pytest.approx(dense, rel=1e-10)


def test_gaussian_loglike_survives_an_extreme_deposition_scale():
    """The regime that broke a Cholesky of the assembled Sigma: a rank-1 deposition term ~1e18
    against a ~1e-7 measurement diagonal, which the optimizer transits while searching sigma.
    It must score finitely so the search can step back out, not hit an infeasible wall."""
    n_helios = 7
    c = np.cos(np.radians(np.linspace(0.0, 75.0, n_helios)))
    deposition = np.stack([1e18 * np.outer(c, c)])
    variances = np.full((1, n_helios), 1e-7)
    residuals = np.full((1, n_helios), 1e-3)

    value = smf.CommonFittingMethods._gaussian_loglike(residuals, deposition, variances)
    assert np.isfinite(value)


def test_correlated_covariance_changes_the_likelihood():
    """Guard against the fix being a no-op: correlated mirrors must score differently from
    independent ones at the same marginal variances."""
    rng = np.random.default_rng(8)
    n_helios = 4
    total = 4e-7
    residuals = rng.normal(0.0, 1e-3, size=(1, n_helios))

    independent = (np.zeros((1, n_helios, n_helios)), np.full((1, n_helios), total))
    shared = (np.stack([0.9 * total * np.ones((n_helios, n_helios))]), np.full((1, n_helios), 0.1 * total))

    assert smf.CommonFittingMethods._gaussian_loglike(residuals, *independent) != pytest.approx(
        smf.CommonFittingMethods._gaussian_loglike(residuals, *shared), rel=1e-6
    )


def test_zero_measurement_variance_raises_a_diagnostic_error():
    """Measurement error is what keeps the covariance invertible once mirrors are correlated,
    so a zero must surface as an explanatory error, not a bare LinAlgError from deep inside."""
    residuals = np.zeros((1, 2))
    deposition = np.zeros((1, 2, 2))
    variances = np.array([[1e-6, 0.0]])
    with pytest.raises(np.linalg.LinAlgError, match="zero measurement variance"):
        smf.CommonFittingMethods._gaussian_loglike(residuals, deposition, variances)


# ---------------------------------------------------------------------------
# 4. The point of the exercise: shared noise is recovered, not diluted
# ---------------------------------------------------------------------------


def _simulate_shared_noise(rng, n_helios, n_times, tilt_values, true_mu, true_sigma, meas_noise_std, meas_every):
    """Reflectance measurements generated with ONE deposition draw per timestep shared by
    every mirror -- the process the model claims and the likelihood must now match."""
    f = 0
    tilt = np.tile(np.asarray(tilt_values, dtype=float)[:, None], (1, n_times))
    density = 40.0
    nominal_reflectance = 0.95
    inc_ref_factor = np.float64(2.0 / np.cos(np.radians(15.0)))
    dust_conc = rng.uniform(20.0, 80.0, size=n_times)
    alpha = dust_conc / density

    sim_in = types.SimpleNamespace(
        time={f: np.arange(n_times)}, files=[f], dust=types.SimpleNamespace(TSP={f: density}), dust_concentration={f: dust_conc}, dust_type={f: "TSP"}
    )

    eps = rng.normal(0.0, true_sigma, size=n_times)  # one draw per TIMESTEP, shared by all mirrors
    basis = alpha[None, :] * gravitational_settling_factor(tilt)
    delta_area = basis * (true_mu + eps[None, :])
    soiling_factor = 1 - np.cumsum(delta_area, axis=1) * inc_ref_factor

    pred_idx = np.arange(meas_every - 1, n_times, meas_every)
    rho = nominal_reflectance * soiling_factor[:, pred_idx]
    average = (rho + rng.normal(0.0, meas_noise_std, size=rho.shape)).T

    ref_dat = types.SimpleNamespace(
        files=[f],
        times={f: pred_idx},
        prediction_indices={f: list(pred_idx)},
        average={f: average},
        sigma_of_the_mean={f: np.full(average.shape, meas_noise_std)},
        rho0={f: np.full(n_helios, nominal_reflectance)},
    )
    helios = _helios_stub(
        tilt={f: tilt},
        inc_ref_factor={f: inc_ref_factor},
        nominal_reflectance=nominal_reflectance,
        soiling_factor={},
        soiling_factor_prediction_variance={},
    )
    return sim_in, ref_dat, helios


# ---------------------------------------------------------------------------
# 3b. The simulator draws the same process the likelihood assumes
# ---------------------------------------------------------------------------


def test_simulated_noise_is_shared_across_mirrors_tilt_only():
    """random_delta_soiled_area must draw ONE deposition fluctuation per timestep, shared by
    every mirror. For a tilt-only model each mirror's share is its own cos(tilt), so the
    simulated noise is perfectly correlated across mirrors. Drawing per (mirror, timestep) --
    what this used to do -- gives correlation ~0 and makes a field average look sqrt(N) quieter
    than it is."""
    f, sim_in, _ref, helios, _tilt, _az = _fixture(n_helios=5, n_times=40)
    model = smf.ConstantMeanDeposition.__new__(smf.ConstantMeanDeposition)
    model.helios = helios
    model.mu_tilde, model.sigma_dep = 4.0e-4, 2.0e-4

    rng = np.random.default_rng(5)
    draws = np.stack([model.random_delta_soiled_area(sim_in, verbose=False, rng=rng)[f] for _ in range(4000)])

    model.calculate_delta_soiled_area(sim_in, verbose=False)
    expected_sd = np.sqrt(model.helios.delta_soiled_area_variance[f])

    # marginal sd per (mirror, timestep) is unchanged by sharing the draw
    np.testing.assert_allclose(draws.std(axis=0), expected_sd, rtol=0.08)
    # mean is the deterministic prediction
    np.testing.assert_allclose(draws.mean(axis=0), model.helios.delta_soiled_area[f], rtol=0.05, atol=1e-12)

    # and at any one timestep the mirrors move together
    for t in (0, 7, 19, 39):
        corr = np.corrcoef(draws[:, :, t], rowvar=False)
        np.testing.assert_allclose(corr, 1.0, atol=1e-8)


def test_simulated_covariance_matches_the_likelihood_covariance():
    """The simulator and the likelihood must describe the same process: the empirical
    across-mirror covariance of one timestep's simulated deposition has to match the Gram
    matrix the covariance is built from, mechanism by mechanism."""
    f, sim_in, _ref, helios, tilt, azimuth = _fixture(n_helios=4, n_times=30)
    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanWindBase.__init__(model, components=["gravitational", "normal_wind"])
    model.helios = helios
    model.mu_tilde, model.omega_windward, model.omega_leeward = 4.0e-4, 6.0e-5, 2.0e-5
    model.sigma_dep, model.sigma_dep_gamma = 2.0e-4, 1.0e-4

    rng = np.random.default_rng(9)
    draws = np.stack([model.random_delta_soiled_area(sim_in, verbose=False, rng=rng)[f] for _ in range(20000)])

    bases = model._noise_bases(sim_in, f)
    sigmas = model._current_sigmas()
    for t in (0, 11, 29):
        expected = sum(sigmas[name] ** 2 * np.outer(basis[:, t], basis[:, t]) for name, basis in bases.items())
        empirical = np.cov(draws[:, :, t], rowvar=False)
        scale = np.abs(expected).max()
        np.testing.assert_allclose(empirical / scale, expected / scale, atol=0.05)


def _shared_noise_case():
    rng = np.random.default_rng(20250805)
    n_helios, n_times = 6, 1440
    true_mu, true_sigma = 4.0e-4, 1.5e-4
    tilt_values = np.linspace(0.0, 75.0, n_helios)
    sim_in, ref_dat, helios = _simulate_shared_noise(rng, n_helios, n_times, tilt_values, true_mu, true_sigma, meas_noise_std=2e-4, meas_every=24)
    model = smf.ConstantMeanDeposition.__new__(smf.ConstantMeanDeposition)
    model.helios = helios
    return model, sim_in, ref_dat, n_helios, true_mu, true_sigma


def test_shared_noise_parameters_are_recovered():
    """Data generated with one shared deposition draw per timestep are fitted back by the
    covariance likelihood."""
    model, sim_in, ref_dat, _, true_mu, true_sigma = _shared_noise_case()

    x_hat, _ = model.fit_mle(sim_in, ref_dat, verbose=False, x0=np.array([3.0e-4, 1.0e-4]), transform_to_original_scale=True)
    mu_hat, sigma_hat = x_hat

    assert np.all(np.isfinite(x_hat))
    assert abs(mu_hat - true_mu) / true_mu < 0.35
    assert abs(sigma_hat - true_sigma) / true_sigma < 0.5


def test_diagonal_likelihood_overstates_the_information():
    """The regression this whole change exists for -- stated as precision, not bias.

    The old diagonal likelihood's sigma_dep POINT ESTIMATE is very nearly unbiased: with
    perfectly correlated mirrors, averaging N_helios copies of the same normalized residual
    still estimates the shared variance. Do not "fix" a bias here; there isn't one.

    What it gets wrong is how much it believes it knows. Fisher information about mu_tilde
    saturates at 1/(sigma^2 b^2 A_i) per interval under the true covariance -- no N_helios in
    it, because the mirrors repeat one draw -- while the diagonal form keeps accumulating
    N_helios-fold. The ratio therefore approaches N_helios as deposition noise comes to
    dominate measurement noise, and the reported standard errors are sqrt(N_helios) too small.
    """
    model, sim_in, ref_dat, n_helios, true_mu, true_sigma = _shared_noise_case()
    f = 0

    def covariance_nll(mu):
        return model._negative_log_likelihood([mu, true_sigma], sim_in, ref_dat)

    def diagonal_nll(mu):
        model.update_model_parameters([mu, true_sigma])
        model.predict_soiling_factor(sim_in, reflectance_data=ref_dat, verbose=False)
        s2 = model._compute_variance_of_measurements(true_sigma, sim_in, reflectance_data=ref_dat)[f]
        pred = model.helios.nominal_reflectance * model.helios.soiling_factor[f][:, ref_dat.prediction_indices[f]].transpose()
        resid = np.diff(ref_dat.average[f], axis=0) - np.diff(pred, axis=0)
        return -np.sum(-0.5 * np.log(s2) - resid**2 / (2 * s2))

    # Both objectives are exactly quadratic in mu_tilde (the prediction is linear in it and
    # neither covariance depends on it), so a central second difference is exact.
    def information(nll, step):
        return (nll(true_mu + step) - 2 * nll(true_mu) + nll(true_mu - step)) / step**2

    step = 0.1 * true_mu
    ratio = information(diagonal_nll, step) / information(covariance_nll, step)

    # deposition variance dominates measurement variance here, so the ratio should sit just
    # under its N_helios ceiling rather than anywhere near 1
    assert 0.7 * n_helios < ratio < 1.05 * n_helios
