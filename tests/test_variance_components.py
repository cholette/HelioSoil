"""
Tests for the multi-mirror difference covariance and the multivariate likelihood.

The mirrors at a site are measured at the same times and soil under the same weather, so
their deposition noise is not independent. Each mechanism's noise is split into a part
common to every mirror during an interval and a mirror-specific part, in proportion
``kappa_c`` to ``1 - kappa_c``. ``_experiment_covariance`` assembles the resulting
covariance of all reflectance differences in an experiment; entry ``((p, i), (q, i'))`` is

    delta_ii' * sum_c sigma_c**2 * [kappa_c + (1 - kappa_c) * delta_pq]
                      * sum_{j in W_i} a_c_j_p * a_c_j_q
    + delta_pq * R_ii'_p

with ``a_c_j_p = b_p * g_c_j_p`` the response of mirror ``p`` to mechanism ``c``'s noise in
interval ``j``, ``W_i`` the intervals spanned by difference ``i``, and ``R`` the differenced
reflectometer covariance. See the soiling model notes, section 8.3.

Six things are checked, in this order:

1. **Nesting.** At ``kappa = 0`` and with no endpoint correction the multivariate
   likelihood must reproduce the original diagonal one. This is the safety net for every
   existing fit: the new code has to contain the old one as a special case.
2. **Single mirror.** With one mirror there is nothing for the common part to be common
   *with*, so the likelihood cannot depend on any kappa. This is the identifiability
   statement of notes eq. (94) in its sharpest form.
3. **Closed form.** For one mechanism and two mirrors of identical geometry, every block of
   the covariance is checked against the formula above written out by hand from the dust
   concentration, the tilt and the measurement sigmas -- not against the model's own
   loadings, which would be circular.
4. **Positive definiteness**, across kappa and across mechanism subsets. This should hold by
   construction (each mechanism contributes the covariance of an actual random vector, and R
   is strictly positive definite), so a failure here means the assembly is wrong.
5. **Attribution.** Two mechanisms whose loadings are orthogonal across mirrors -- a
   horizontal mirror feels only gravitational settling, a face-down one only normal wind --
   must not leak into each other: neither the covariance blocks nor the kappas.
6. **Marginalisation.** A missing measurement drops the differences that depend on it, which
   is exact for a normal, and must equal the likelihood of the reduced data computed
   independently.
"""

import types

import numpy as np
import pytest

from heliosoil.base_models import ConstantMeanBase
from heliosoil.fitting import ConstantMeanDeposition
from heliosoil.horizontal_impaction import ConstantMeanWindBase, ConstantMeanWindDeposition
from heliosoil.utilities import gravitational_settling_factor

RTOL = 1e-11

F = 0
T_GRID = 25
PREDICTION_INDICES = [0, 4, 9, 17, 24]  # uneven spacing: dk = 4, 5, 8, 7
N_DIFF = len(PREDICTION_INDICES) - 1

MU_TILDE = 0.015
SIGMA_DEP = 8e-4
OMEGA_WINDWARD = 6.0e-07
OMEGA_LEEWARD = 9.0e-07
SIGMA_DEP_GAMMA = 5e-06
DENSITY = 40.0
NOMINAL_REFLECTANCE = 0.95
INC_REF_FACTOR = 2.0 / np.cos(np.radians(15.0))
MEASUREMENT_SIGMA = 1.5e-3

B = NOMINAL_REFLECTANCE * INC_REF_FACTOR


# ---------------------------------------------------------------------------
# Stubs. The site data a model normally reads from Excel is set directly, so these
# tests exercise the covariance arithmetic and nothing else.
# ---------------------------------------------------------------------------


def _blank_helios_dicts(model):
    for name in ("delta_soiled_area", "delta_soiled_area_variance", "soiling_factor", "soiling_factor_prediction_variance"):
        setattr(model.helios, name, {})


def _constant_mean_model(tilt):
    model = ConstantMeanDeposition.__new__(ConstantMeanDeposition)
    ConstantMeanBase.__init__(model)
    model.mu_tilde, model.sigma_dep = MU_TILDE, SIGMA_DEP
    model.helios.nominal_reflectance = NOMINAL_REFLECTANCE
    model.helios.tilt = {F: np.asarray(tilt, dtype=float)}
    model.helios.inc_ref_factor = {F: np.array(INC_REF_FACTOR)}
    _blank_helios_dicts(model)
    return model


def _wind_model(tilt, azimuth, components=("gravitational", "normal_wind")):
    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanBase.__init__(model)
    ConstantMeanWindBase.__init__(model, components=list(components))
    model.verbose = False
    model.mu_tilde, model.sigma_dep = MU_TILDE, SIGMA_DEP
    model.omega_windward, model.omega_leeward = OMEGA_WINDWARD, OMEGA_LEEWARD
    model.sigma_dep_gamma = SIGMA_DEP_GAMMA
    model.helios.nominal_reflectance = NOMINAL_REFLECTANCE
    model.helios.tilt = {F: np.asarray(tilt, dtype=float)}
    model.helios.azimuth = {F: np.asarray(azimuth, dtype=float)}
    model.helios.inc_ref_factor = {F: np.array(INC_REF_FACTOR)}
    _blank_helios_dicts(model)
    return model


def _parameters(model):
    """The model's parameter vector, read off the attributes the stubs set."""
    names = getattr(model, "_param_names", None) or ["mu_tilde", "sigma_dep"]
    return [getattr(model, name) for name in names]


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


def _ref_dat(n_mirrors, measurement_sigma=MEASUREMENT_SIGMA, seed=0):
    """Reflectance that decreases over time, so the residuals are not all zero."""
    rng = np.random.default_rng(seed)
    shape = (len(PREDICTION_INDICES), n_mirrors)
    average = 0.93 - np.cumsum(rng.uniform(0.0, 2e-3, shape), axis=0)
    sigma = np.full(shape, float(measurement_sigma))
    return types.SimpleNamespace(
        files=["experiment_0"],
        times={F: np.asarray(PREDICTION_INDICES, dtype=float)},
        prediction_indices={F: PREDICTION_INDICES},
        rho0={F: average[0, :].copy()},
        sigma_of_the_mean={F: sigma},
        average={F: average},
    )


def _weather(seed=1):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(20.0, 60.0, T_GRID),  # dust concentration
        rng.uniform(1.0, 9.0, T_GRID),  # wind speed
        rng.uniform(0.0, 360.0, T_GRID),  # wind direction
    )


def _windows():
    """The intervals each difference spans, restated rather than read from the model."""
    pi = PREDICTION_INDICES
    return [range(pi[i] + 1, pi[i + 1] + 1) for i in range(N_DIFF)]


# ---------------------------------------------------------------------------
# 1. Nesting -- the multivariate likelihood contains the diagonal one
# ---------------------------------------------------------------------------


def test_constant_mean_components_reduce_to_the_diagonal_likelihood_at_kappa_zero():
    """Independent mirrors and independent differences is exactly the original likelihood."""
    concentration, _, _ = _weather()
    model, ref_dat = _constant_mean_model(np.full((3, T_GRID), 30.0)), _ref_dat(3)
    sim_in = _sim_in(concentration)

    diagonal = model._negative_log_likelihood_diagonal(_parameters(model), sim_in, ref_dat)
    components = model._negative_log_likelihood_components(_parameters(model), sim_in, ref_dat, endpoint_correction=False)

    assert components == pytest.approx(diagonal, rel=RTOL)


def test_wind_components_reduce_to_the_diagonal_likelihood_at_kappa_zero():
    """Same nesting with several mechanisms, where the channels no longer share a sigma."""
    concentration, speed, direction = _weather()
    tilt = np.array([[0.0] * T_GRID, [45.0] * T_GRID, [80.0] * T_GRID])
    azimuth = np.array([[0.0] * T_GRID, [90.0] * T_GRID, [200.0] * T_GRID])
    model, ref_dat = _wind_model(tilt, azimuth), _ref_dat(3)
    sim_in = _sim_in(concentration, speed, direction)

    diagonal = model._negative_log_likelihood_diagonal(_parameters(model), sim_in, ref_dat)
    components = model._negative_log_likelihood_components(_parameters(model), sim_in, ref_dat, endpoint_correction=False)

    assert components == pytest.approx(diagonal, rel=RTOL)


def test_the_default_variance_model_dispatches_to_the_diagonal_likelihood():
    """An untouched model must fit exactly as it did before the covariance existed."""
    concentration, speed, direction = _weather()
    tilt = np.array([[10.0] * T_GRID, [70.0] * T_GRID])
    azimuth = np.array([[20.0] * T_GRID, [150.0] * T_GRID])
    model, ref_dat = _wind_model(tilt, azimuth), _ref_dat(2)
    sim_in = _sim_in(concentration, speed, direction)

    assert model.variance_model == "independent"
    assert model.endpoint_correction is False
    dispatched = model._negative_log_likelihood(_parameters(model), sim_in, ref_dat)
    diagonal = model._negative_log_likelihood_diagonal(_parameters(model), sim_in, ref_dat)
    assert dispatched == diagonal  # the same code path, not merely the same number


@pytest.mark.parametrize(
    "variance_model,expected",
    [("independent", False), ("shared_kappa", True), ("per_mechanism", True)],
)
def test_endpoint_correction_follows_the_variance_model_unless_set(variance_model, expected):
    model = _constant_mean_model(np.full((2, T_GRID), 30.0))
    model.set_variance_model(variance_model)
    assert model.endpoint_correction is expected

    model.set_variance_model(variance_model, endpoint_correction=not expected)
    assert model.endpoint_correction is (not expected)


# ---------------------------------------------------------------------------
# 2. One mirror -- nothing for the common component to be common with
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_single_mirror_likelihood_is_invariant_to_kappa(kappa):
    """With one mirror the two variance components are not separately identified.

    kappa splits a fixed total between a common and a mirror-specific part; with a single
    mirror both enter the variance the same way, so the likelihood must not move at all.
    """
    concentration, speed, direction = _weather()
    tilt = np.array([[35.0] * T_GRID])
    azimuth = np.array([[110.0] * T_GRID])
    model, ref_dat = _wind_model(tilt, azimuth), _ref_dat(1)
    sim_in = _sim_in(concentration, speed, direction)

    model.set_variance_model("shared_kappa", kappa=0.0, endpoint_correction=True)
    reference = model._negative_log_likelihood(_parameters(model), sim_in, ref_dat)

    model.set_variance_model("shared_kappa", kappa=kappa, endpoint_correction=True)
    assert model._negative_log_likelihood(_parameters(model), sim_in, ref_dat) == pytest.approx(reference, rel=RTOL)


# ---------------------------------------------------------------------------
# 3. Closed form -- every block checked against the formula written out by hand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.0, 0.3, 1.0])
def test_covariance_matches_the_closed_form_for_one_mechanism(kappa):
    """Two mirrors of identical geometry, one mechanism, checked entry by entry.

    Everything on the right-hand side is rebuilt from the dust concentration, the tilt and
    the measurement sigmas, so this does not lean on the model's own loading code.
    """
    tilt_degrees = 30.0
    concentration, _, _ = _weather()
    model, ref_dat = _constant_mean_model(np.full((2, T_GRID), tilt_degrees)), _ref_dat(2)
    sim_in = _sim_in(concentration)
    model.set_variance_model("shared_kappa", kappa=kappa)

    # a_j = b * alpha_j * cos+(tilt), identical for both mirrors here.
    a = B * (concentration / DENSITY) * gravitational_settling_factor(tilt_degrees)
    deposition = np.array([SIGMA_DEP**2 * np.sum(a[list(window)] ** 2) for window in _windows()])

    # R = Delta Sigma_r Delta': tridiagonal, since consecutive differences share a
    # measurement. All sigmas equal here, so 2 s^2 on the diagonal and -s^2 next to it.
    s2 = MEASUREMENT_SIGMA**2
    r = np.zeros((N_DIFF, N_DIFF))
    for i in range(N_DIFF):
        r[i, i] = 2.0 * s2
        if i + 1 < N_DIFF:
            r[i, i + 1] = r[i + 1, i] = -s2

    cov = model._experiment_covariance(F, sim_in, ref_dat, endpoint_correction=True)

    same_mirror = np.diag(deposition) + r
    other_mirror = np.diag(kappa * deposition)
    for p in range(2):
        for q in range(2):
            block = cov[p * N_DIFF : (p + 1) * N_DIFF, q * N_DIFF : (q + 1) * N_DIFF]
            expected = same_mirror if p == q else other_mirror
            np.testing.assert_allclose(block, expected, rtol=RTOL, atol=0.0, err_msg=f"block ({p}, {q})")


def test_covariance_is_symmetric():
    concentration, speed, direction = _weather()
    tilt = np.array([[5.0] * T_GRID, [50.0] * T_GRID, [95.0] * T_GRID])
    azimuth = np.array([[0.0] * T_GRID, [120.0] * T_GRID, [240.0] * T_GRID])
    model, ref_dat = _wind_model(tilt, azimuth), _ref_dat(3)
    model.set_variance_model("shared_kappa", kappa=0.42)

    cov = model._experiment_covariance(F, _sim_in(concentration, speed, direction), ref_dat, endpoint_correction=True)
    np.testing.assert_allclose(cov, cov.transpose(), rtol=0.0, atol=0.0)


def test_entries_are_ordered_mirror_major():
    """Difference i of mirror p sits at index p * n_differences + i.

    Pinned through the measurement noise, which is the one input that can be made
    obviously different between mirrors.
    """
    concentration, _, _ = _weather()
    model, ref_dat = _constant_mean_model(np.full((2, T_GRID), 20.0)), _ref_dat(2)
    ref_dat.sigma_of_the_mean[F] = np.column_stack(
        [np.full(len(PREDICTION_INDICES), 1e-2), np.full(len(PREDICTION_INDICES), 1e-6)]
    )

    cov = model._experiment_covariance(F, _sim_in(concentration), ref_dat, endpoint_correction=False)
    first_block = np.diag(cov)[:N_DIFF]
    second_block = np.diag(cov)[N_DIFF:]

    # The noisy mirror's block is dominated by measurement noise (2e-4), the quiet one's is
    # not. If the flattening were difference-major these would interleave instead.
    assert np.all(first_block > 1e-4)
    assert np.all(second_block < 1e-4)


# ---------------------------------------------------------------------------
# 4. Positive definiteness -- structural, so a failure means a wrong assembly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("endpoint_correction", [False, True])
def test_covariance_is_positive_definite_across_kappa(kappa, endpoint_correction):
    concentration, speed, direction = _weather()
    tilt = np.array([[0.0] * T_GRID, [30.0] * T_GRID, [60.0] * T_GRID, [90.0] * T_GRID])
    azimuth = np.array([[0.0] * T_GRID, [90.0] * T_GRID, [180.0] * T_GRID, [270.0] * T_GRID])
    model, ref_dat = _wind_model(tilt, azimuth), _ref_dat(4)
    model.set_variance_model("shared_kappa", kappa=kappa)

    cov = model._experiment_covariance(F, _sim_in(concentration, speed, direction), ref_dat, endpoint_correction=endpoint_correction)
    assert np.linalg.eigvalsh(cov).min() > 0.0


@pytest.mark.parametrize(
    "components",
    [
        ("gravitational",),
        ("gravitational", "normal_wind"),
        ("gravitational", "turbulent_wind", "normal_wind"),
        ("gravitational", "normal_wind", "tangential_wind", "impaction_retention"),
    ],
)
def test_covariance_is_positive_definite_for_every_mechanism_subset(components):
    concentration, speed, direction = _weather()
    tilt = np.array([[0.0] * T_GRID, [45.0] * T_GRID, [110.0] * T_GRID])
    azimuth = np.array([[0.0] * T_GRID, [100.0] * T_GRID, [220.0] * T_GRID])
    model = _wind_model(tilt, azimuth, components=components)
    for name in model._sigma_param_names:
        setattr(model, name, 3e-5)
    for name in model._mean_param_names:
        if getattr(model, name, None) is None:
            setattr(model, name, 5e-7)
    model.set_variance_model("shared_kappa", kappa=0.6)

    cov = model._experiment_covariance(F, _sim_in(concentration, speed, direction), _ref_dat(3), endpoint_correction=True)
    assert np.linalg.eigvalsh(cov).min() > 0.0


def test_a_channel_with_no_magnitude_contributes_nothing():
    """A least-squares fit leaves the sigmas unset; the covariance is then measurement noise."""
    concentration, speed, direction = _weather()
    tilt = np.array([[15.0] * T_GRID, [75.0] * T_GRID])
    azimuth = np.array([[40.0] * T_GRID, [190.0] * T_GRID])
    model, ref_dat = _wind_model(tilt, azimuth), _ref_dat(2)
    model.sigma_dep = model.sigma_dep_gamma = None
    model.set_variance_model("shared_kappa", kappa=0.8)

    cov = model._experiment_covariance(F, _sim_in(concentration, speed, direction), ref_dat, endpoint_correction=True)
    measurement = model._differenced_measurement_covariance(F, ref_dat, endpoint_correction=True)
    for p in range(2):
        block = cov[p * N_DIFF : (p + 1) * N_DIFF, p * N_DIFF : (p + 1) * N_DIFF]
        np.testing.assert_allclose(block, measurement[p], rtol=RTOL, atol=0.0)
    np.testing.assert_allclose(cov[:N_DIFF, N_DIFF:], 0.0, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# 5. Attribution -- mechanisms that load different mirrors must not leak
# ---------------------------------------------------------------------------
#
# Gravitational settling scales with cos+(tilt), which is exactly zero past vertical; the
# normal-wind projection scales with sin(tilt), which is exactly zero at horizontal. So a
# horizontal mirror feels only gravitational deposition and a face-down one only wind. This
# is the Yadnarie field's structure in miniature.

ORTHOGONAL_TILTS = [0.0, 0.0, 120.0, 120.0]
ORTHOGONAL_AZIMUTHS = [10.0, 10.0, 200.0, 200.0]


def _orthogonal_model(kappa_dep, kappa_dep_gamma):
    tilt = np.array([[t] * T_GRID for t in ORTHOGONAL_TILTS])
    azimuth = np.array([[a] * T_GRID for a in ORTHOGONAL_AZIMUTHS])
    model = _wind_model(tilt, azimuth)
    model.set_variance_model("per_mechanism")
    model.kappa_dep, model.kappa_dep_gamma = kappa_dep, kappa_dep_gamma
    return model


def test_the_orthogonal_construction_really_is_orthogonal():
    """The premise of the attribution tests, checked rather than assumed."""
    concentration, speed, direction = _weather()
    model = _orthogonal_model(0.5, 0.5)
    channels = {c.name: c.loading for c in model.noise_channels(F, _sim_in(concentration, speed, direction))}

    np.testing.assert_array_equal(channels["sigma_dep"][2:, :], 0.0)  # face-down: no settling
    np.testing.assert_array_equal(channels["sigma_dep_gamma"][:2, :], 0.0)  # horizontal: no wind
    assert channels["sigma_dep"][:2, :].max() > 0.0
    assert channels["sigma_dep_gamma"][2:, :].max() > 0.0


def test_mechanisms_do_not_couple_mirrors_they_do_not_both_load():
    """Mirror 0 (gravitational only) and mirror 2 (wind only) share no noise at any kappa."""
    concentration, speed, direction = _weather()
    model = _orthogonal_model(0.9, 0.9)
    cov = model._experiment_covariance(F, _sim_in(concentration, speed, direction), _ref_dat(4), endpoint_correction=True)

    for p, q in [(0, 2), (0, 3), (1, 2), (1, 3)]:
        block = cov[p * N_DIFF : (p + 1) * N_DIFF, q * N_DIFF : (q + 1) * N_DIFF]
        np.testing.assert_allclose(block, 0.0, rtol=0.0, atol=0.0, err_msg=f"block ({p}, {q}) should be empty")


def test_one_mechanisms_kappa_does_not_move_the_others_block():
    """kappa_dep_gamma must move the wind pair's coupling and leave the gravitational pair alone."""
    concentration, speed, direction = _weather()
    sim_in, ref_dat = _sim_in(concentration, speed, direction), _ref_dat(4)

    low = _orthogonal_model(0.5, 0.2)._experiment_covariance(F, sim_in, ref_dat, endpoint_correction=True)
    high = _orthogonal_model(0.5, 0.8)._experiment_covariance(F, sim_in, ref_dat, endpoint_correction=True)

    def block(cov, p, q):
        return cov[p * N_DIFF : (p + 1) * N_DIFF, q * N_DIFF : (q + 1) * N_DIFF]

    # The gravitational pair (0, 1) is untouched...
    np.testing.assert_allclose(block(low, 0, 1), block(high, 0, 1), rtol=0.0, atol=0.0)
    # ...and so is every diagonal block, since kappa only splits a fixed total.
    for p in range(4):
        np.testing.assert_allclose(block(low, p, p), block(high, p, p), rtol=0.0, atol=0.0)
    # The wind pair (2, 3) scales with kappa_dep_gamma.
    np.testing.assert_allclose(block(high, 2, 3), 4.0 * block(low, 2, 3), rtol=RTOL, atol=0.0)
    assert np.abs(block(low, 2, 3)).max() > 0.0


def test_a_mechanism_only_one_mirror_loads_leaves_the_likelihood_kappa_free():
    """notes eq. (94): kappa_c needs at least two mirrors that mechanism actually loads."""
    concentration, speed, direction = _weather()
    tilt = np.array([[0.0] * T_GRID, [0.0] * T_GRID, [120.0] * T_GRID])  # one wind-loaded mirror
    azimuth = np.array([[10.0] * T_GRID, [10.0] * T_GRID, [200.0] * T_GRID])
    sim_in, ref_dat = _sim_in(concentration, speed, direction), _ref_dat(3)

    values = []
    for kappa_gamma in (0.0, 0.4, 1.0):
        model = _wind_model(tilt, azimuth)
        model.set_variance_model("per_mechanism", endpoint_correction=True)
        model.kappa_dep, model.kappa_dep_gamma = 0.3, kappa_gamma
        values.append(model._negative_log_likelihood(_parameters(model), sim_in, ref_dat))

    assert values[1] == pytest.approx(values[0], rel=RTOL)
    assert values[2] == pytest.approx(values[0], rel=RTOL)


# ---------------------------------------------------------------------------
# 6. Marginalisation -- a missing measurement is dropped, not imputed
# ---------------------------------------------------------------------------


def test_a_missing_measurement_is_marginalised_exactly():
    """Blanking one reading must give the likelihood of the data that remain.

    The right-hand side is built with slogdet and solve rather than the Cholesky the
    implementation uses, so this is an independent evaluation of the same normal density.
    """
    concentration, speed, direction = _weather()
    tilt = np.array([[10.0] * T_GRID, [55.0] * T_GRID, [100.0] * T_GRID])
    azimuth = np.array([[0.0] * T_GRID, [130.0] * T_GRID, [250.0] * T_GRID])
    sim_in, ref_dat = _sim_in(concentration, speed, direction), _ref_dat(3)
    ref_dat.average[F][2, 1] = np.nan  # mirror 1 was not read at the third visit

    model = _wind_model(tilt, azimuth)
    model.set_variance_model("shared_kappa", kappa=0.35, endpoint_correction=True)
    got = model._negative_log_likelihood(_parameters(model), sim_in, ref_dat)

    # Rebuild the same density from the retained rows only.
    model.update_model_parameters(_parameters(model))
    model.predict_soiling_factor(sim_in, reflectance_data=ref_dat, verbose=False)
    residual = model._difference_residuals(F, ref_dat).transpose().ravel()
    cov = model._experiment_covariance(F, sim_in, ref_dat, endpoint_correction=True)

    keep = np.isfinite(residual)
    assert keep.sum() == residual.size - 2  # one blank reading kills the two differences on it
    residual, cov = residual[keep], cov[keep, :][:, keep]
    _, log_det = np.linalg.slogdet(cov)
    expected = 0.5 * (residual.size * np.log(2 * np.pi) + log_det + residual @ np.linalg.solve(cov, residual))

    assert got == pytest.approx(expected, rel=RTOL)


def test_marginalisation_ignores_the_dropped_mirrors_own_noise_only():
    """Dropping mirror 1's differences must not disturb the others' contributions."""
    concentration, speed, direction = _weather()
    tilt = np.array([[10.0] * T_GRID, [55.0] * T_GRID])
    azimuth = np.array([[0.0] * T_GRID, [130.0] * T_GRID])
    sim_in = _sim_in(concentration, speed, direction)

    complete = _ref_dat(2)
    blanked = _ref_dat(2)
    blanked.average[F][:, 1] = np.nan  # mirror 1 never read at all

    model = _wind_model(tilt, azimuth)
    model.set_variance_model("shared_kappa", kappa=0.0, endpoint_correction=True)
    only_mirror_zero = model._negative_log_likelihood(_parameters(model), sim_in, blanked)

    single = _ref_dat(1)
    single.average[F] = complete.average[F][:, :1]
    single.rho0[F] = complete.rho0[F][:1]
    single_model = _wind_model(tilt[:1, :], azimuth[:1, :])
    single_model.set_variance_model("shared_kappa", kappa=0.0, endpoint_correction=True)
    alone = single_model._negative_log_likelihood(_parameters(single_model), sim_in, single)

    assert only_mirror_zero == pytest.approx(alone, rel=RTOL)


# ---------------------------------------------------------------------------
# Guards on the variance-model settings themselves
# ---------------------------------------------------------------------------


def test_set_variance_model_rejects_an_unknown_model():
    model = _constant_mean_model(np.full((1, T_GRID), 10.0))
    with pytest.raises(ValueError, match="variance_model must be one of"):
        model.set_variance_model("components")


@pytest.mark.parametrize("kappa", [-1e-9, 1.0 + 1e-9, 2.0])
def test_set_variance_model_rejects_kappa_outside_the_unit_interval(kappa):
    model = _constant_mean_model(np.full((1, T_GRID), 10.0))
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        model.set_variance_model("shared_kappa", kappa=kappa)


def test_kappa_names_follow_the_sigma_names():
    tilt = np.array([[0.0] * T_GRID])
    azimuth = np.array([[0.0] * T_GRID])
    model = _wind_model(tilt, azimuth, components=("gravitational", "normal_wind", "tangential_wind"))
    assert model._sigma_param_names == ["sigma_dep", "sigma_dep_gamma", "sigma_dep_tan"]
    assert model._kappa_param_names == ("kappa_dep", "kappa_dep_gamma", "kappa_dep_tan")
    # kappa is deliberately NOT in the parameter vector yet: it is model state that
    # update_model_parameters does not touch.
    assert not set(model._kappa_param_names) & set(model._param_names)


def test_per_mechanism_requires_every_kappa_to_be_set():
    concentration, speed, direction = _weather()
    tilt = np.array([[0.0] * T_GRID, [60.0] * T_GRID])
    azimuth = np.array([[0.0] * T_GRID, [180.0] * T_GRID])
    model = _wind_model(tilt, azimuth)
    model.variance_model = "per_mechanism"  # bypass the seeding set_variance_model does
    model.kappa_dep, model.kappa_dep_gamma = 0.5, None

    with pytest.raises(ValueError, match="kappa_dep_gamma"):
        model._experiment_covariance(F, _sim_in(concentration, speed, direction), _ref_dat(2))


def test_per_mechanism_seeds_from_the_shared_fraction():
    tilt = np.array([[0.0] * T_GRID, [60.0] * T_GRID])
    azimuth = np.array([[0.0] * T_GRID, [180.0] * T_GRID])
    model = _wind_model(tilt, azimuth)
    model.set_variance_model("per_mechanism", kappa=0.7)
    assert model.kappa_dep == 0.7
    assert model.kappa_dep_gamma == 0.7

    # An already-set fraction is left alone.
    model.kappa_dep_gamma = 0.2
    model.set_variance_model("per_mechanism", kappa=0.7)
    assert model.kappa_dep_gamma == 0.2
