"""
Tests for the parameter plumbing shared by the fitting classes.

Both models optimise on an unconstrained scale -- ``log(log(hrz0))`` or
``log(mu_tilde)``, ``log(sigma_dep)``, and for the variance-components model
``logit(kappa)`` -- so that the natural-scale constraints ``hrz0 > 1``,
``sigma_dep > 0`` and ``0 < kappa < 1`` hold without bounds. ``transform_scale`` moves
parameters and Hessians between the two scales, and the parameter vector grows from two
entries to three when the components model is selected.

The Jacobian is checked against central finite differences rather than against a
rewritten derivative, so an error in the analytic form cannot be mirrored in the
reference.
"""

import types
import numpy as np

from heliosoil.fitting import SemiPhysical, ConstantMeanDeposition
from heliosoil.base_models import ConstantMeanBase, PhysicalBase


RTOL = 1e-10
ATOL = 0.0


def _bare(cls, base):
    """A model instance without the Excel-backed __init__."""
    model = cls.__new__(cls)
    base.__init__(model)
    return model


def _models():
    return [
        _bare(SemiPhysical, PhysicalBase),
        _bare(ConstantMeanDeposition, ConstantMeanBase),
    ]


# ---------------------------------------------------------------------------
# 1. Defaults and configuration
# ---------------------------------------------------------------------------


def test_default_variance_model_is_scalar():
    """Models built without the new options behave as they did before it existed."""
    for model in _models():
        assert model.variance_model == "scalar"
        assert model.n_parameters == 2
        assert model.endpoint_correction is False


def test_endpoint_correction_follows_variance_model_unless_set():
    for model in _models():
        model.set_variance_model("components")
        assert model.endpoint_correction is True
        assert model.n_parameters == 3

        model.set_variance_model("components", endpoint_correction=False)
        assert model.endpoint_correction is False

        model.set_variance_model("scalar", endpoint_correction=True)
        assert model.endpoint_correction is True


def test_unknown_variance_model_is_rejected():
    for model in _models():
        try:
            model.set_variance_model("componentwise")
        except ValueError:
            continue
        raise AssertionError("expected ValueError for an unknown variance model")


def test_parameter_names_track_the_variance_model():
    semi, constant = _models()

    assert semi.parameter_names == ["hrz0", "sigma_dep"]
    assert semi.transformed_parameter_names == ["log(log(hrz0))", "log(sigma_dep)"]
    assert constant.parameter_names == ["mu_tilde", "sigma_dep"]
    assert constant.transformed_parameter_names == ["log(mu_tilde)", "log(sigma_dep)"]

    for model in (semi, constant):
        model.set_variance_model("components")
        assert model.parameter_names[-1] == "kappa"
        assert model.transformed_parameter_names[-1] == "logit(kappa)"


# ---------------------------------------------------------------------------
# 2. transform_scale
# ---------------------------------------------------------------------------


def test_transform_scale_round_trips():
    """forward and inverse are mutual inverses, with and without kappa."""
    for model in _models():
        natural_two = np.array([2.5 if model._mean_parameter_transform == "log_log" else 0.02,
                                8e-4])
        for natural in (natural_two, np.append(natural_two, 0.3)):
            if natural.size == 3:
                model.set_variance_model("components")
            transformed = model.transform_scale(natural, direction="forward")
            back = model.transform_scale(transformed, direction="inverse")
            np.testing.assert_allclose(back, natural, rtol=1e-12, atol=ATOL)


def test_transform_scale_matches_the_stated_transforms():
    semi, constant = _models()
    semi.set_variance_model("components")
    constant.set_variance_model("components")

    y = np.array([-0.4, -7.0, 0.8])
    kappa = 1.0 / (1.0 + np.exp(-y[2]))

    np.testing.assert_allclose(
        semi.transform_scale(y), [np.exp(np.exp(y[0])), np.exp(y[1]), kappa], rtol=RTOL
    )
    np.testing.assert_allclose(
        constant.transform_scale(y), [np.exp(y[0]), np.exp(y[1]), kappa], rtol=RTOL
    )


def test_transform_scale_jacobian_matches_finite_differences():
    """The Hessian transform uses d(natural)/d(fitting); check it numerically.

    transform_scale(y, H) applies J^-T H J^-1, so recovering J from the returned
    Hessian and comparing with central differences of the inverse transform tests the
    analytic Jacobian without restating it.
    """
    step = 1e-6
    for model in _models():
        model.set_variance_model("components")
        y = np.array([-0.35, -7.1, 0.6])

        # Identity Hessian in => J^-T J^-1 out => J J^T recoverable by inversion.
        _, H = model.transform_scale(y, likelihood_hessian=np.eye(3))
        jacobian_from_transform = np.sqrt(np.diag(np.linalg.inv(H)))

        numerical = np.empty(3)
        for i in range(3):
            up, down = y.astype(float).copy(), y.astype(float).copy()
            up[i] += step
            down[i] -= step
            numerical[i] = (
                model.transform_scale(up)[i] - model.transform_scale(down)[i]
            ) / (2 * step)

        np.testing.assert_allclose(jacobian_from_transform, numerical, rtol=1e-6, atol=0.0)


# ---------------------------------------------------------------------------
# 3. update_model_parameters
# ---------------------------------------------------------------------------


def test_update_model_parameters_sets_the_variance_split():
    sigma_dep, kappa = 8e-4, 0.36
    for model in _models():
        model.update_model_parameters([0.02, sigma_dep, kappa])
        assert model.common_variance_fraction == kappa
        np.testing.assert_allclose(model.sigma_c, sigma_dep * np.sqrt(kappa), rtol=RTOL)
        np.testing.assert_allclose(model.sigma_m, sigma_dep * np.sqrt(1 - kappa), rtol=RTOL)
        # sigma_dep keeps its meaning as the total, so downstream code is unaffected.
        np.testing.assert_allclose(
            model.sigma_c**2 + model.sigma_m**2, sigma_dep**2, rtol=RTOL
        )


def test_update_model_parameters_rejects_out_of_range_fraction():
    for model in _models():
        for bad in (-0.1, 1.5):
            try:
                model.update_model_parameters([0.02, 8e-4, bad])
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for kappa = {bad}")


def test_two_parameter_update_leaves_the_split_untouched():
    """The scalar path must not silently invent a variance split."""
    for model in _models():
        model.update_model_parameters([0.02, 8e-4])
        assert model.sigma_c is None
        assert model.sigma_m is None


# ---------------------------------------------------------------------------
# 4. Identifiability guard
# ---------------------------------------------------------------------------


def _reflectance_stub(mirror_counts):
    return types.SimpleNamespace(
        average={f: np.zeros((5, p)) for f, p in enumerate(mirror_counts)}
    )


def test_single_mirror_experiments_are_rejected_for_components():
    for model in _models():
        try:
            model._check_variance_components_identifiable(_reflectance_stub([1, 1]))
        except ValueError as error:
            assert "two mirrors" in str(error)
            continue
        raise AssertionError("expected ValueError when no experiment has two mirrors")


def test_one_multi_mirror_experiment_is_enough():
    for model in _models():
        model._check_variance_components_identifiable(_reflectance_stub([1, 4, 1]))


# ---------------------------------------------------------------------------
# 5. Delta-method summary for sigma_c and sigma_m
# ---------------------------------------------------------------------------


def test_variance_component_summary_point_estimates():
    """sigma_c and sigma_m are the back-transform of (sigma_dep, kappa)."""
    model = _models()[0]
    model.set_variance_model("components")

    sigma_dep, kappa = 9e-4, 0.4
    y_hat = np.array([-0.3, np.log(sigma_dep), np.log(kappa) - np.log1p(-kappa)])
    summary = model._variance_component_summary(y_hat, np.eye(3) * 1e-4)

    np.testing.assert_allclose(summary["sigma_c"][0], sigma_dep * np.sqrt(kappa), rtol=1e-9)
    np.testing.assert_allclose(summary["sigma_m"][0], sigma_dep * np.sqrt(1 - kappa), rtol=1e-9)
    for name in ("sigma_c", "sigma_m"):
        estimate, lower, upper = summary[name]
        assert 0.0 < lower < estimate < upper


def test_variance_component_intervals_shrink_with_precision():
    """A tighter parameter covariance gives tighter component intervals."""
    model = _models()[0]
    model.set_variance_model("components")
    y_hat = np.array([-0.3, np.log(9e-4), 0.2])

    wide = model._variance_component_summary(y_hat, np.eye(3) * 1e-2)
    tight = model._variance_component_summary(y_hat, np.eye(3) * 1e-6)
    for name in ("sigma_c", "sigma_m"):
        assert (tight[name][2] - tight[name][1]) < (wide[name][2] - wide[name][1])
