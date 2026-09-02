"""
Tests for the least-squares search bracket over the mean parameter.

``fit_least_squares`` uses a bounded one-dimensional search, which returns an endpoint
when the optimum lies outside the bracket. That failure is silent -- a bound is still a
valid float -- so the bracket has to suit the model's mean parameter. ``hrz0`` exceeds
one; ``mu_tilde`` is a small positive number, and a fixed ``hrz0``-shaped bracket
excludes its entire admissible range.

The ``mu_tilde`` bound is derived from the data: mirror ``p`` accumulates
``mu_tilde * sum_j alpha_j cos(tilt_j_p)`` and the predicted reflectance is ``rho0``
minus ``b`` times that, so requiring a positive prediction bounds ``mu_tilde``.

1. ``test_semi_physical_*`` / ``test_constant_mean_*`` -- the brackets themselves.
2. ``test_fit_least_squares_*`` -- recovery of a known value, and the boundary warning.
"""

import types
import warnings
import numpy as np

from heliosoil.fitting import SemiPhysical, ConstantMeanDeposition, CommonFittingMethods
from heliosoil.base_models import ConstantMeanBase, PhysicalBase


RTOL = 1e-10

F = 0
NOMINAL_REFLECTANCE = 0.95
# Small enough that the accumulated loss over the span leaves the mirrors reflective:
# with this loading, mu_tilde above ~1.6e-2 would drive the prediction negative.
MU_TILDE = 0.003
DENSITY = 40.0
INC_REF_FACTOR = 2.0 / np.cos(np.radians(15.0))
T_GRID = 21
PREDICTION_INDICES = [0, 5, 11, 20]
RHO0 = np.array([0.94, 0.93])


def _constant_mean_model(tilt=None):
    model = ConstantMeanDeposition.__new__(ConstantMeanDeposition)
    ConstantMeanBase.__init__(model)
    model.mu_tilde, model.sigma_dep = MU_TILDE, None
    model.helios.nominal_reflectance = NOMINAL_REFLECTANCE
    model.helios.tilt = {F: np.zeros((2, T_GRID)) if tilt is None else tilt}
    model.helios.inc_ref_factor = {F: np.array(INC_REF_FACTOR)}
    for name in ("delta_soiled_area", "delta_soiled_area_variance", "soiling_factor", "soiling_factor_prediction_variance"):
        setattr(model.helios, name, {})
    return model


def _sim_in(concentration=None):
    if concentration is None:
        concentration = np.linspace(20.0, 90.0, T_GRID)
    return types.SimpleNamespace(
        files=["experiment_0"],
        time={F: np.arange(T_GRID, dtype=float)},
        dust=types.SimpleNamespace(PM10={F: DENSITY}),
        dust_concentration={F: concentration},
        dust_type={F: "PM10"},
    )


def _reflectance_data(average):
    return types.SimpleNamespace(
        files=["experiment_0"],
        times={F: np.asarray(PREDICTION_INDICES, dtype=float)},
        average={F: average},
        rho0={F: RHO0},
        prediction_indices={F: PREDICTION_INDICES},
        sigma_of_the_mean={F: np.full_like(average, 1e-3)},
    )


def _exact_measurements(model, sim_in, mu_tilde=MU_TILDE):
    model.update_model_parameters([mu_tilde])
    model.predict_soiling_factor(sim_in, rho0={F: RHO0}, verbose=False)
    stub = types.SimpleNamespace(prediction_indices={F: PREDICTION_INDICES})
    return model._predicted_reflectance(F, stub)


# ---------------------------------------------------------------------------
# 1. The brackets
# ---------------------------------------------------------------------------


def test_base_class_refuses_to_guess_a_bracket():
    """A model that has not declared a bracket must not silently inherit one."""
    try:
        CommonFittingMethods()._mean_parameter_bounds(None, None)
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError from the base class")


def test_semi_physical_bracket_admits_hrz0():
    model = SemiPhysical.__new__(SemiPhysical)
    PhysicalBase.__init__(model)
    lower, upper = model._mean_parameter_bounds(None, None)

    assert lower > 1.0  # log(log(hrz0)) requires hrz0 > 1
    assert upper == 1000.0


def test_constant_mean_bracket_admits_mu_tilde_and_matches_its_derivation():
    """The upper bound is min_p rho0_p / (b * accumulated loading)."""
    model = _constant_mean_model()
    sim_in = _sim_in()
    ref_dat = _reflectance_data(_exact_measurements(model, sim_in))

    lower, upper = model._mean_parameter_bounds(sim_in, ref_dat)

    alpha = sim_in.dust_concentration[F] / DENSITY
    accumulated = np.sum(alpha[: PREDICTION_INDICES[-1] + 1])  # cos(tilt) = 1
    expected = np.min(RHO0) / (NOMINAL_REFLECTANCE * INC_REF_FACTOR * accumulated)

    np.testing.assert_allclose(upper, expected, rtol=RTOL)
    assert lower < MU_TILDE < upper  # the true value is interior


def test_constant_mean_bracket_tightens_with_dust_loading():
    """More airborne dust means less mu_tilde is admissible."""
    model = _constant_mean_model()
    light = _sim_in(concentration=np.full(T_GRID, 20.0))
    heavy = _sim_in(concentration=np.full(T_GRID, 200.0))
    ref_light = _reflectance_data(_exact_measurements(model, light))
    ref_heavy = _reflectance_data(_exact_measurements(model, heavy))

    _, upper_light = model._mean_parameter_bounds(light, ref_light)
    _, upper_heavy = model._mean_parameter_bounds(heavy, ref_heavy)

    assert upper_heavy < upper_light
    np.testing.assert_allclose(upper_heavy / upper_light, 0.1, rtol=1e-9)


def test_constant_mean_bracket_survives_zero_loading():
    """A degenerate loading falls back rather than returning inf or zero."""
    model = _constant_mean_model()
    sim_in = _sim_in(concentration=np.zeros(T_GRID))
    ref_dat = _reflectance_data(_exact_measurements(model, sim_in))

    lower, upper = model._mean_parameter_bounds(sim_in, ref_dat)
    assert np.isfinite(upper) and upper > lower


# ---------------------------------------------------------------------------
# 2. fit_least_squares
# ---------------------------------------------------------------------------


def test_fit_least_squares_recovers_mu_tilde_and_stays_interior():
    """On data generated from the model the estimate is the true value, not a bound."""
    model = _constant_mean_model()
    sim_in = _sim_in()
    ref_dat = _reflectance_data(_exact_measurements(model, sim_in))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a boundary result must not occur here
        estimate, sse = model.fit_least_squares(sim_in, ref_dat, verbose=False)

    lower, upper = model._mean_parameter_bounds(sim_in, ref_dat)
    np.testing.assert_allclose(estimate, MU_TILDE, rtol=1e-6)
    assert lower < estimate < upper

    # Residuals are far below the measurement noise floor: sigma is 1e-3 per point
    # over 8 points, so a noise-level fit would give an SSE of order 1e-5.
    assert sse < 1e-12


def test_fit_least_squares_warns_when_pinned_to_a_bound():
    """A bracket that excludes the optimum is reported rather than passed off as a fit."""
    model = _constant_mean_model()
    sim_in = _sim_in()
    ref_dat = _reflectance_data(_exact_measurements(model, sim_in))

    # The historical hrz0 bracket, which excludes every admissible mu_tilde.
    lower, upper = 1.0 + 1e-6, 1000.0
    model._mean_parameter_bounds = lambda *_: (lower, upper)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate, _ = model.fit_least_squares(sim_in, ref_dat, verbose=False)

    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert any("edge of the search bracket" in str(w.message) for w in caught)

    # Pinned to the lower bound, not converged to an interior optimum. Tested with the
    # same criterion the warning uses rather than an exact value, since where a bounded
    # search stops near a bound is an optimiser detail.
    assert estimate - lower <= 1e-6 * (upper - lower)
    assert estimate > 100 * MU_TILDE  # nowhere near the truth


def test_fit_least_squares_warns_on_a_narrow_bracket():
    """The edge test must not be defeated by the optimiser's own stopping tolerance.

    A bounded search stops within an absolute tolerance that, by default, ignores the
    bracket width. On a narrow bracket that tolerance can exceed the edge test, so a
    pinned result would look interior. fit_least_squares therefore sets xatol from the
    width. Here the bracket lies entirely below the true mu_tilde, so the optimum is at
    the upper bound.
    """
    model = _constant_mean_model()
    sim_in = _sim_in()
    ref_dat = _reflectance_data(_exact_measurements(model, sim_in))

    upper = MU_TILDE / 3.0  # narrow, and excludes the optimum
    model._mean_parameter_bounds = lambda *_: (1e-6, upper)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate, _ = model.fit_least_squares(sim_in, ref_dat, verbose=False)

    assert any("edge of the search bracket" in str(w.message) for w in caught)
    assert upper - estimate <= 1e-6 * (upper - 1e-6)
