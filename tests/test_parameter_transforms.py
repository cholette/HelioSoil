"""
Tests for the parameter vector, its transforms, and the variance-component reporting.

Every parameter is optimised on an unconstrained "fitted" scale and stored on its natural
one. Which map applies is declared per parameter by ``transform_codes`` -- log for a
magnitude, log-log for ``hrz0``, logit for a common fraction, identity for a wind
coefficient that may legitimately fit negative -- and one ``transform_scale`` in
``CommonFittingMethods`` serves every model family. These tests pin:

1. **The maps themselves**, against their closed forms, for all three families. A round
   trip is not enough: it would pass if two transforms were swapped.
2. **The derivatives**, against numerical differentiation. ``natural_scale_jacobian`` and
   ``_fitted_scale_jacobian`` are reciprocals evaluated at matching points, and both are
   used by the delta method, so an error there silently corrupts every interval.
3. **The Hessian transform**, by round trip and against the explicit diagonal congruence.
4. **The common fractions on the parameter vector**: which ones appear under which
   variance model, that ``_param_names`` (what the forward model accepts) does NOT gain
   them, and that "shared_kappa" and "per_mechanism" agree when the fractions are equal.
5. **The identifiability guard**, which needs two mirrors that a mechanism actually loads.
6. **The reporting**: sigma, the scaled magnitude tau = sigma * s, the variance share, and
   kappa, each with a delta-method interval that is None -- not NaN -- where the curvature
   is unusable, which is what a parameter on its bound produces.
"""

import types

import numpy as np
import pytest
from scipy.special import expit, logit

from heliosoil.base_models import ConstantMeanBase
from heliosoil.fitting import (
    ConstantMeanDeposition,
    SemiPhysical,
    TRANSFORM_IDENTITY,
    TRANSFORM_LOG,
    TRANSFORM_LOGIT,
    TRANSFORM_LOGLOG,
    kappa_param_name,
)
from heliosoil.horizontal_impaction import ConstantMeanWindBase, ConstantMeanWindDeposition

from test_variance_components import (
    F,
    T_GRID,
    _ref_dat,
    _sim_in,
    _weather,
    _wind_model,
)

RTOL = 1e-11


def _semi_physical():
    """Transform behaviour needs no site data: the names and codes are class attributes."""
    return SemiPhysical.__new__(SemiPhysical)


def _constant_mean():
    return ConstantMeanDeposition.__new__(ConstantMeanDeposition)


# ---------------------------------------------------------------------------
# 1. The maps, against their closed forms
# ---------------------------------------------------------------------------


def test_semi_physical_carries_hrz0_as_log_log():
    model = _semi_physical()
    assert model.transform_codes == (TRANSFORM_LOGLOG, TRANSFORM_LOG)
    x = [np.exp(np.e), 3e-4]
    y = model.transform_scale(x, direction="forward")
    np.testing.assert_allclose(y, [np.log(np.log(x[0])), np.log(x[1])], rtol=RTOL)
    np.testing.assert_allclose(model.transform_scale(y), x, rtol=RTOL)


def test_constant_mean_carries_both_parameters_as_logs():
    model = _constant_mean()
    assert model.transform_codes == (TRANSFORM_LOG, TRANSFORM_LOG)
    x = [0.014, 7e-4]
    y = model.transform_scale(x, direction="forward")
    np.testing.assert_allclose(y, np.log(x), rtol=RTOL)
    np.testing.assert_allclose(model.transform_scale(y), x, rtol=RTOL)


def test_wind_model_keeps_its_omegas_linear():
    """A negative omega is meaningful (scouring), so those entries must not be logged."""
    model = _wind_model(np.zeros((1, T_GRID)), np.zeros((1, T_GRID)))
    assert model.transform_codes == (TRANSFORM_LOG, TRANSFORM_IDENTITY, TRANSFORM_IDENTITY, TRANSFORM_LOG, TRANSFORM_LOG)
    x = [0.02, -3e-7, 5e-7, 9e-4, 4e-6]
    y = model.transform_scale(x, direction="forward")
    np.testing.assert_allclose(y, [np.log(x[0]), x[1], x[2], np.log(x[3]), np.log(x[4])], rtol=RTOL)
    np.testing.assert_allclose(model.transform_scale(y), x, rtol=RTOL)


def test_a_mean_only_vector_transforms_with_the_leading_codes():
    """fit_ls returns the means alone, and they come first in the vector."""
    model = _wind_model(np.zeros((1, T_GRID)), np.zeros((1, T_GRID)))
    means = [0.02, -3e-7, 5e-7]
    np.testing.assert_allclose(model.transform_scale(means, direction="forward"), [np.log(means[0]), means[1], means[2]], rtol=RTOL)
    np.testing.assert_allclose(_semi_physical().transform_scale([np.exp(np.e)], direction="forward"), [1.0], rtol=RTOL)
    np.testing.assert_allclose(_constant_mean().transform_scale([np.e], direction="forward"), [1.0], rtol=RTOL)


@pytest.mark.parametrize("length", [0, 4, 9])
def test_transform_scale_rejects_a_vector_of_the_wrong_length(length):
    model = _wind_model(np.zeros((1, T_GRID)), np.zeros((1, T_GRID)))
    with pytest.raises(ValueError, match=f"Cannot transform a vector of {length} parameter"):
        model.transform_scale(np.ones(length), direction="forward")


def test_transform_scale_rejects_a_bare_scalar():
    """A scalar used to fall through and return None, which is worse than failing."""
    with pytest.raises(ValueError, match="takes a parameter vector"):
        _constant_mean().transform_scale(0.01, direction="forward")


def test_transform_scale_rejects_an_unknown_direction():
    with pytest.raises(ValueError, match="direction not recognized"):
        _constant_mean().transform_scale([0.01, 1e-3], direction="sideways")


# ---------------------------------------------------------------------------
# 2. The derivatives, against numerical differentiation
# ---------------------------------------------------------------------------


def _numerical_natural_derivative(model, y):
    """d(natural)/d(fitted), by central difference through the public transform."""
    y = np.asarray(y, dtype=float)
    out = np.zeros(len(y))
    for i in range(len(y)):
        step = 1e-6 * max(abs(y[i]), 1.0)
        up, down = y.copy(), y.copy()
        up[i] += step
        down[i] -= step
        out[i] = (model.transform_scale(up)[i] - model.transform_scale(down)[i]) / (2.0 * step)
    return out


@pytest.mark.parametrize(
    "make_model,natural",
    [
        (_semi_physical, [np.exp(np.e), 3e-4]),
        (_constant_mean, [0.014, 7e-4]),
    ],
)
def test_natural_scale_jacobian_matches_numerical_differentiation(make_model, natural):
    model = make_model()
    y = model.transform_scale(natural, direction="forward")
    np.testing.assert_allclose(model.natural_scale_jacobian(y), _numerical_natural_derivative(model, y), rtol=1e-6)


def test_natural_scale_jacobian_matches_numerical_differentiation_with_kappa():
    model = _wind_model(np.zeros((2, T_GRID)), np.zeros((2, T_GRID)))
    model.set_variance_model("per_mechanism", kappa=0.3)
    y = model.transform_scale([0.02, -3e-7, 5e-7, 9e-4, 4e-6, 0.3, 0.7], direction="forward")
    np.testing.assert_allclose(model.natural_scale_jacobian(y), _numerical_natural_derivative(model, y), rtol=1e-6)


@pytest.mark.parametrize(
    "make_model,natural,expected",
    [
        # d log(log(hrz0)) / d hrz0 = 1 / (hrz0 log hrz0)
        (_semi_physical, [np.exp(np.e)], lambda x: 1.0 / (x * np.log(x))),
        # d log(mu_tilde) / d mu_tilde = 1 / mu_tilde
        (_constant_mean, [0.014], lambda x: 1.0 / x),
    ],
)
def test_fitted_scale_jacobian_matches_the_closed_form(make_model, natural, expected):
    model = make_model()
    np.testing.assert_allclose(model._fitted_scale_jacobian(natural), expected(np.asarray(natural)), rtol=RTOL)


def test_fitted_scale_jacobian_is_the_reciprocal_of_the_natural_one():
    model = _wind_model(np.zeros((1, T_GRID)), np.zeros((1, T_GRID)))
    means = [0.02, -3e-7, 5e-7]
    y = model.transform_scale(means, direction="forward")
    np.testing.assert_allclose(model._fitted_scale_jacobian(means) * model.natural_scale_jacobian(y), 1.0, rtol=RTOL)


# ---------------------------------------------------------------------------
# 3. The Hessian transform
# ---------------------------------------------------------------------------


def test_hessian_transform_is_the_diagonal_congruence():
    model = _wind_model(np.zeros((1, T_GRID)), np.zeros((1, T_GRID)))
    x = [0.02, -3e-7, 5e-7, 9e-4, 4e-6]
    y = np.asarray(model.transform_scale(x, direction="forward"))
    rng = np.random.default_rng(3)
    a = rng.normal(size=(5, 5))
    hessian = a @ a.transpose()

    _, transformed = model.transform_scale(y, likelihood_hessian=hessian)
    jacobian = model.natural_scale_jacobian(y)
    np.testing.assert_allclose(transformed, hessian / np.outer(jacobian, jacobian), rtol=RTOL)


def test_hessian_transform_round_trips():
    model = _constant_mean()
    x = [0.014, 7e-4]
    y = np.asarray(model.transform_scale(x, direction="forward"))
    rng = np.random.default_rng(5)
    a = rng.normal(size=(2, 2))
    hessian = a @ a.transpose()

    x_back, natural = model.transform_scale(y, likelihood_hessian=hessian)
    _, fitted = model.transform_scale(x_back, likelihood_hessian=natural, direction="forward")
    np.testing.assert_allclose(fitted, hessian, rtol=1e-9)


def test_the_covariance_delta_method_agrees_with_the_hessian_route():
    """``J cov J`` and ``inv(transform(inv(cov)))`` are the same matrix.

    ``fit_mle`` uses the first. This pins that the switch was algebra, not a change of
    answer -- for a covariance the inverse route can actually be taken.
    """
    model = _constant_mean()
    y = np.asarray(model.transform_scale([0.014, 7e-4], direction="forward"))
    rng = np.random.default_rng(11)
    a = rng.normal(size=(2, 2))
    y_cov = a @ a.transpose() + np.eye(2)

    jacobian = model.natural_scale_jacobian(y)
    direct = (jacobian[:, None] * y_cov) * jacobian[None, :]

    _, hessian = model.transform_scale(y, likelihood_hessian=np.linalg.inv(y_cov))
    np.testing.assert_allclose(direct, np.linalg.inv(hessian), rtol=1e-9)


def test_the_covariance_delta_method_survives_a_variance_parameter_on_its_bound():
    """The two routes stop agreeing where one of them stops existing.

    sigma >= 0 makes the constraint active whenever the observed scatter is no larger than
    the measurement noise already predicts -- a routine outcome, not a pathology. The
    curvature is then singular, the round trip through a Hessian raises, and the direct
    delta method still returns a finite (rank-deficient) covariance.
    """
    model = _constant_mean()
    y = np.asarray(model.transform_scale([0.014, 7e-4], direction="forward"))
    y_cov = np.array([[0.02, 0.0], [0.0, 0.0]])  # sigma_dep pinned: no curvature left

    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(y_cov)

    jacobian = model.natural_scale_jacobian(y)
    x_hat_cov = (jacobian[:, None] * y_cov) * jacobian[None, :]
    assert np.all(np.isfinite(x_hat_cov))
    assert x_hat_cov[0, 0] == pytest.approx(0.02 * 0.014**2, rel=RTOL)
    assert x_hat_cov[1, 1] == 0.0


# ---------------------------------------------------------------------------
# 4. The common fractions on the parameter vector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variance_model,expected_tail",
    [
        ("independent", []),
        ("shared_kappa", ["common_variance_fraction"]),
        ("per_mechanism", ["kappa_dep", "kappa_dep_gamma"]),
    ],
)
def test_the_variance_model_decides_which_fractions_are_fitted(variance_model, expected_tail):
    model = _wind_model(np.zeros((2, T_GRID)), np.zeros((2, T_GRID)))
    model.set_variance_model(variance_model)

    core = ["mu_tilde", "omega_windward", "omega_leeward", "sigma_dep", "sigma_dep_gamma"]
    assert model.parameter_names == core + expected_tail
    assert model.transform_codes[len(core) :] == (TRANSFORM_LOGIT,) * len(expected_tail)
    # The forward model's own parameter list must NOT gain them: a common fraction changes
    # the likelihood, not the prediction, so it is not a valid override for
    # calculate_delta_soiled_area.
    assert list(model._param_names) == core


def test_kappa_names_are_derived_from_the_magnitude_names():
    assert kappa_param_name("sigma_dep") == "kappa_dep"
    assert kappa_param_name("sigma_dep_gamma") == "kappa_dep_gamma"
    with pytest.raises(ValueError, match="naming convention"):
        kappa_param_name("noise_level")


def test_kappa_round_trips_through_the_logit():
    model = _wind_model(np.zeros((2, T_GRID)), np.zeros((2, T_GRID)))
    model.set_variance_model("per_mechanism", kappa=0.5)
    x = [0.02, -3e-7, 5e-7, 9e-4, 4e-6, 0.05, 0.95]
    y = model.transform_scale(x, direction="forward")
    np.testing.assert_allclose(y[5:], [logit(0.05), logit(0.95)], rtol=RTOL)
    np.testing.assert_allclose(model.transform_scale(y), x, rtol=1e-10)


@pytest.mark.parametrize("fitted", [-50.0, -5.0, 0.0, 5.0, 50.0])
def test_the_logit_keeps_kappa_strictly_inside_the_unit_interval(fitted):
    """Which is why the covariance never sees the singular kappa = 1 case."""
    model = _wind_model(np.zeros((2, T_GRID)), np.zeros((2, T_GRID)))
    model.set_variance_model("shared_kappa")
    kappa = model.transform_scale([0.02, -3e-7, 5e-7, 9e-4, 4e-6, fitted])[5]
    assert 0.0 <= kappa <= 1.0
    assert kappa == expit(fitted)


def test_update_model_parameters_writes_the_fractions_it_fits():
    model = _wind_model(np.zeros((2, T_GRID)), np.zeros((2, T_GRID)))

    model.set_variance_model("shared_kappa")
    model.update_model_parameters([0.02, -3e-7, 5e-7, 9e-4, 4e-6, 0.25])
    assert model.common_variance_fraction == 0.25

    model.set_variance_model("per_mechanism")
    model.update_model_parameters([0.02, -3e-7, 5e-7, 9e-4, 4e-6, 0.1, 0.8])
    assert (model.kappa_dep, model.kappa_dep_gamma) == (0.1, 0.8)
    # The mean and magnitude entries are still written, unshifted.
    assert (model.mu_tilde, model.sigma_dep, model.sigma_dep_gamma) == (0.02, 9e-4, 4e-6)


def test_a_mean_and_magnitude_vector_is_still_accepted_when_fractions_are_fitted():
    """fit_ls and the 1-D warm start pass short vectors; those must not be mistaken for
    a full one whose tail is fractions."""
    model = _wind_model(np.zeros((2, T_GRID)), np.zeros((2, T_GRID)))
    model.set_variance_model("shared_kappa", kappa=0.6)
    model.update_model_parameters([0.02, -3e-7, 5e-7, 9e-4, 4e-6])
    assert model.common_variance_fraction == 0.6  # untouched
    assert model.sigma_dep_gamma == 4e-6


def test_update_model_parameters_rejects_a_fraction_outside_the_unit_interval():
    model = _wind_model(np.zeros((2, T_GRID)), np.zeros((2, T_GRID)))
    model.set_variance_model("shared_kappa")
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        model.update_model_parameters([0.02, -3e-7, 5e-7, 9e-4, 4e-6, 1.5])


def test_shared_and_per_mechanism_agree_when_the_fractions_are_equal():
    concentration, speed, direction = _weather()
    tilt = np.array([[t] * T_GRID for t in [10.0, 40.0, 70.0]])
    azimuth = np.array([[a] * T_GRID for a in [0.0, 120.0, 240.0]])
    sim_in, ref_dat = _sim_in(concentration, speed, direction), _ref_dat(3)
    core = [0.015, 6e-7, 9e-7, 8e-4, 5e-6]

    shared = _wind_model(tilt, azimuth)
    shared.set_variance_model("shared_kappa")
    a = shared._negative_log_likelihood(core + [0.3], sim_in, ref_dat)

    per = _wind_model(tilt, azimuth)
    per.set_variance_model("per_mechanism")
    b = per._negative_log_likelihood(core + [0.3, 0.3], sim_in, ref_dat)

    assert a == pytest.approx(b, rel=RTOL)


# ---------------------------------------------------------------------------
# 5. The identifiability guard
# ---------------------------------------------------------------------------


def _three_mirror_case(tilts):
    concentration, speed, direction = _weather()
    tilt = np.array([[t] * T_GRID for t in tilts])
    azimuth = np.array([[a] * T_GRID for a in [10.0, 130.0, 250.0][: len(tilts)]])
    return _wind_model(tilt, azimuth), _sim_in(concentration, speed, direction), _ref_dat(len(tilts))


def test_loaded_mirror_counts_ignores_mirrors_a_mechanism_does_not_touch():
    """Face-down mirrors feel no settling; horizontal ones feel no normal wind."""
    model, sim_in, ref_dat = _three_mirror_case([0.0, 45.0, 120.0])
    counts = model.loaded_mirror_counts(sim_in, ref_dat)
    assert counts["sigma_dep"] == 2  # tilts 0 and 45; 120 is past vertical
    assert counts["sigma_dep_gamma"] == 2  # tilts 45 and 120; sin(0) = 0


def test_independent_needs_no_mirrors_at_all():
    model, sim_in, ref_dat = _three_mirror_case([0.0])
    model.set_variance_model("independent")
    model._check_variance_components_identifiable(sim_in, ref_dat)  # must not raise


def test_shared_kappa_needs_one_mechanism_with_two_loaded_mirrors():
    model, sim_in, ref_dat = _three_mirror_case([30.0])
    model.set_variance_model("shared_kappa")
    with pytest.raises(ValueError, match="at least two mirrors loaded by the same mechanism"):
        model._check_variance_components_identifiable(sim_in, ref_dat)

    model, sim_in, ref_dat = _three_mirror_case([0.0, 30.0])
    model.set_variance_model("shared_kappa")
    model._check_variance_components_identifiable(sim_in, ref_dat)  # gravitational loads both


def test_per_mechanism_needs_two_loaded_mirrors_for_every_mechanism():
    # Both horizontal: gravitational sees two, normal wind sees none.
    model, sim_in, ref_dat = _three_mirror_case([0.0, 0.0])
    model.set_variance_model("per_mechanism")
    with pytest.raises(ValueError, match="sigma_dep_gamma"):
        model._check_variance_components_identifiable(sim_in, ref_dat)

    model, sim_in, ref_dat = _three_mirror_case([30.0, 60.0])
    model.set_variance_model("per_mechanism")
    model._check_variance_components_identifiable(sim_in, ref_dat)  # both load both


def test_the_guard_reports_the_counts_it_saw():
    """The message has to say what was wrong, not just that something was."""
    model, sim_in, ref_dat = _three_mirror_case([0.0, 0.0])
    model.set_variance_model("per_mechanism")
    with pytest.raises(ValueError, match="sigma_dep: 2, sigma_dep_gamma: 0"):
        model._check_variance_components_identifiable(sim_in, ref_dat)


# ---------------------------------------------------------------------------
# 6. Reporting
# ---------------------------------------------------------------------------


def _summary_case(kappa=0.4):
    concentration, speed, direction = _weather()
    tilt = np.array([[t] * T_GRID for t in [0.0, 20.0, 45.0, 70.0]])
    azimuth = np.array([[a] * T_GRID for a in [10.0, 80.0, 150.0, 220.0]])
    model = _wind_model(tilt, azimuth)
    model.set_variance_model("shared_kappa")
    core = [0.015, 6e-7, 9e-7, 8e-4, 5e-6]
    model.update_model_parameters(core + [kappa])
    y = np.asarray(model.transform_scale(core + [kappa], direction="forward"))
    cov = np.diag([0.01, 1e-14, 1e-14, 0.04, 0.09, 0.25])
    return model, y, cov, _sim_in(concentration, speed, direction), _ref_dat(4)


def test_summary_reports_sigma_on_the_natural_scale_with_a_log_interval():
    model, y, cov, sim_in, ref_dat = _summary_case()
    entry = model.variance_component_summary(y, cov, sim_in, ref_dat)["sigma_dep"]
    estimate, (lower, upper) = entry["sigma"]
    assert estimate == pytest.approx(8e-4, rel=RTOL)
    assert lower == pytest.approx(np.exp(y[3] - 1.96 * 0.2), rel=RTOL)
    assert upper == pytest.approx(np.exp(y[3] + 1.96 * 0.2), rel=RTOL)
    assert lower < estimate < upper  # positive, and the interval brackets the estimate


def test_tau_is_sigma_times_the_rms_loading():
    model, y, cov, sim_in, ref_dat = _summary_case()
    scales = model.noise_channel_scales(sim_in, ref_dat)
    summary = model.variance_component_summary(y, cov, sim_in, ref_dat)
    for name in ("sigma_dep", "sigma_dep_gamma"):
        sigma = summary[name]["sigma"][0]
        assert summary[name]["tau"][0] == pytest.approx(sigma * scales[name], rel=RTOL)
    # tau is a constant multiple of sigma, so its interval is the same relative width.
    for name in ("sigma_dep", "sigma_dep_gamma"):
        sigma_lo, sigma_hi = summary[name]["sigma"][1]
        tau_lo, tau_hi = summary[name]["tau"][1]
        assert tau_hi / tau_lo == pytest.approx(sigma_hi / sigma_lo, rel=1e-9)


def test_the_variance_shares_sum_to_one_and_stay_inside_the_unit_interval():
    model, y, cov, sim_in, ref_dat = _summary_case()
    summary = model.variance_component_summary(y, cov, sim_in, ref_dat)
    shares = [entry["share"][0] for entry in summary.values()]
    assert sum(shares) == pytest.approx(1.0, rel=RTOL)
    for share, (lower, upper) in ((entry["share"][0], entry["share"][1]) for entry in summary.values()):
        assert 0.0 < lower < share < upper < 1.0


def test_the_share_is_the_ratio_of_squared_scaled_magnitudes():
    model, y, cov, sim_in, ref_dat = _summary_case()
    scales = model.noise_channel_scales(sim_in, ref_dat)
    tau2 = {name: (getattr(model, name) * scales[name]) ** 2 for name in model._sigma_param_names}
    total = sum(tau2.values())
    summary = model.variance_component_summary(y, cov, sim_in, ref_dat)
    for name in tau2:
        assert summary[name]["share"][0] == pytest.approx(tau2[name] / total, rel=1e-9)


@pytest.mark.parametrize("kappa", [0.05, 0.4, 0.9])
def test_kappa_is_reported_with_an_interval_inside_the_unit_interval(kappa):
    model, y, cov, sim_in, ref_dat = _summary_case(kappa=kappa)
    entry = model.variance_component_summary(y, cov, sim_in, ref_dat)["sigma_dep"]
    estimate, (lower, upper) = entry["kappa"]
    assert estimate == pytest.approx(kappa, rel=1e-9)
    assert 0.0 < lower < estimate < upper < 1.0


def test_an_unusable_curvature_gives_no_interval_rather_than_a_nan_one():
    """A parameter on its bound leaves a curvature the Wald interval cannot use. Reporting
    None says so; reporting NaN reads as a number."""
    model, y, cov, sim_in, ref_dat = _summary_case()
    cov = cov.copy()
    cov[3, 3] = np.nan
    entry = model.variance_component_summary(y, cov, sim_in, ref_dat)["sigma_dep"]
    assert entry["sigma"][1] is None
    assert entry["tau"][1] is None
    assert np.isfinite(entry["sigma"][0])  # the estimate is still reported


def test_the_summary_omits_tau_and_the_share_without_a_design():
    """sigma and kappa need only the fit; tau and the share need the loadings too."""
    model, y, cov, _, _ = _summary_case()
    entry = model.variance_component_summary(y, cov)["sigma_dep"]
    assert set(entry) == {"sigma", "kappa"}


def test_the_summary_skips_a_mechanism_with_no_magnitude():
    """A least-squares fit leaves every magnitude unset."""
    model, y, cov, sim_in, ref_dat = _summary_case()
    model.sigma_dep_gamma = None
    summary = model.variance_component_summary(y, cov, sim_in, ref_dat)
    assert set(summary) == {"sigma_dep"}
    assert summary["sigma_dep"]["share"] == (1.0, None)  # alone, so the logit is not finite
