"""
Tests for the predicted reflectance shared by the fitting objectives.

The per-mirror initial reflectance ``rho0`` is carried by the soiling factor, so the
predicted reflectance is ``nominal_reflectance * soiling_factor``. Scaling by ``rho0``
instead applies the initial cleanliness twice and starts the prediction at
``rho0**2 / nominal_reflectance``; the discrepancy vanishes when
``rho0 == nominal_reflectance``, so every case here deliberately uses mirrors whose
initial reflectance differs from nominal and from each other.

The reference is derived independently of the implementation, from

    rho(t) = rho0 - rho_nom * inc_ref_factor * cumsum(delta_soiled_area)

which follows from the back-calculated initial soiled area
``A_0 = (1 - rho0/rho_nom)/inc_ref_factor``.

Groups:

1. ``test_predicted_reflectance_*`` -- the helper, and the property that with no
   deposition the prediction is the measured ``rho0``.
2. ``test_sse_*`` -- ``_sse`` is zero on data generated exactly from the model.
3. ``test_negative_log_likelihood_*`` -- the likelihood uses the same mean as ``_sse``.
"""

import types
import numpy as np
import pytest

from heliosoil.fitting import ConstantMeanDeposition
from heliosoil.base_models import ConstantMeanBase


# Tight tolerance: these are exact re-derivations, not approximations.
RTOL = 1e-10
ATOL = 0.0

F = 0  # single experiment key throughout

NOMINAL_REFLECTANCE = 0.95
MU_TILDE = 0.02
DENSITY = 40.0  # PM10 reference value

# Deliberately != NOMINAL_REFLECTANCE, and different per mirror: this is exactly
# the configuration in which the two conventions disagree.
RHO0 = np.array([0.93, 0.90, 0.95])

TILT = np.array(
    [
        [0.0, 30.0, 60.0, 10.0, 20.0, 45.0],
        [10.0, 45.0, 80.0, 15.0, 25.0, 5.0],
        [20.0, 0.0, 35.0, 50.0, 60.0, 70.0],
    ]
)
DUST_CONCENTRATION = np.array([20.0, 50.0, 80.0, 110.0, 45.0, 65.0])
PREDICTION_INDICES = [0, 2, 3, 5]  # measurement times, as indices into the sim grid
INC_REF_FACTOR = 2.0 / np.cos(np.radians(15.0))  # second-surface, 15 deg reflectometer


def _model():
    """A ConstantMeanDeposition with site data stubbed instead of read from Excel."""
    model = ConstantMeanDeposition.__new__(ConstantMeanDeposition)
    ConstantMeanBase.__init__(model)

    model.mu_tilde = MU_TILDE
    model.sigma_dep = None

    model.helios.nominal_reflectance = NOMINAL_REFLECTANCE
    model.helios.tilt = {F: TILT}
    model.helios.inc_ref_factor = {F: INC_REF_FACTOR}
    model.helios.delta_soiled_area = {}
    model.helios.delta_soiled_area_variance = {}
    model.helios.soiling_factor = {}
    model.helios.soiling_factor_prediction_variance = {}
    return model


def _sim_in():
    return types.SimpleNamespace(
        files=["experiment_0"],
        time={F: np.arange(TILT.shape[1], dtype=float)},
        dust=types.SimpleNamespace(PM10={F: DENSITY}),
        dust_concentration={F: DUST_CONCENTRATION},
        dust_type={F: "PM10"},
    )


def _reference_reflectance():
    """Independent reference: rho(t) = rho0 - rho_nom * inc * cumsum(dA).

    The value at grid index j includes deposition *during* interval j, matching
    ``compute_soiling_factor``.
    """
    alpha = DUST_CONCENTRATION / DENSITY
    delta_area = alpha[None, :] * np.cos(np.radians(TILT)) * MU_TILDE  # (P, T)
    cumulative_area = np.cumsum(delta_area, axis=1)[:, PREDICTION_INDICES]  # (P, N)
    soiled_loss = NOMINAL_REFLECTANCE * INC_REF_FACTOR * cumulative_area
    return (RHO0[:, None] - soiled_loss).transpose()  # (N, P)


def _reflectance_data(average):
    return types.SimpleNamespace(
        files=["experiment_0"],
        times={F: np.asarray(PREDICTION_INDICES, dtype=float)},
        average={F: average},
        rho0={F: RHO0},
        prediction_indices={F: PREDICTION_INDICES},
        sigma_of_the_mean={F: np.full_like(average, 1e-3)},
    )


# ---------------------------------------------------------------------------
# 1. The helper
# ---------------------------------------------------------------------------


def test_predicted_reflectance_matches_independent_reference():
    """rho_nom * sf reproduces rho0 - rho_nom * inc * cumsum(dA)."""
    model = _model()
    sim_in = _sim_in()

    model.predict_soiling_factor(sim_in, rho0={F: RHO0}, verbose=False)
    got = model._predicted_reflectance(F, _reflectance_data(_reference_reflectance()))

    np.testing.assert_allclose(got, _reference_reflectance(), rtol=RTOL, atol=ATOL)
    assert got.shape == (len(PREDICTION_INDICES), TILT.shape[0])


def test_predicted_reflectance_starts_at_measured_rho0():
    """With no deposition the prediction is the measured rho0, per mirror."""
    model = _model()
    model.mu_tilde = 0.0  # no deposition -> reflectance stays at its initial value
    sim_in = _sim_in()

    model.predict_soiling_factor(sim_in, rho0={F: RHO0}, verbose=False)
    got = model._predicted_reflectance(F, _reflectance_data(np.zeros((4, 3))))

    for row in got:
        np.testing.assert_allclose(row, RHO0, rtol=RTOL, atol=ATOL)

    # The rho0-scaled alternative is materially different for this fixture, so the
    # assertion above is a meaningful discriminator rather than a tautology.
    assert np.max(np.abs(RHO0**2 / NOMINAL_REFLECTANCE - RHO0)) > 1e-3


# ---------------------------------------------------------------------------
# 2. Sum of squared errors
# ---------------------------------------------------------------------------


def test_sse_is_zero_on_exact_model_data():
    """Data generated from the model gives SSE = 0 at the generating parameters."""
    model = _model()
    sse = model._sse(MU_TILDE, _sim_in(), _reflectance_data(_reference_reflectance()))
    assert sse == pytest.approx(0.0, abs=1e-24)


def test_sse_is_positive_and_finite_off_the_truth():
    """Sanity: perturbing mu_tilde away from the truth increases SSE."""
    ref_dat = _reflectance_data(_reference_reflectance())
    sim_in = _sim_in()

    sse_true = _model()._sse(MU_TILDE, sim_in, ref_dat)
    sse_low = _model()._sse(0.5 * MU_TILDE, sim_in, ref_dat)
    sse_high = _model()._sse(1.5 * MU_TILDE, sim_in, ref_dat)

    assert sse_low > sse_true
    assert sse_high > sse_true
    assert np.isfinite(sse_low) and np.isfinite(sse_high)


# ---------------------------------------------------------------------------
# 3. Likelihood consistency
# ---------------------------------------------------------------------------


def test_negative_log_likelihood_uses_the_same_mean_as_sse():
    """On exact model data the residuals vanish, leaving only the normalising terms."""
    model = _model()
    ref_dat = _reflectance_data(_reference_reflectance())
    sim_in = _sim_in()
    sigma_dep = 1e-4

    nll = model._negative_log_likelihood([MU_TILDE, sigma_dep], sim_in, ref_dat)

    s2total = model._compute_variance_of_measurements(
        sigma_dep, sim_in, reflectance_data=ref_dat
    )
    expected = 0.5 * s2total[F].size * np.log(2 * np.pi) + 0.5 * np.sum(np.log(s2total[F]))

    np.testing.assert_allclose(nll, expected, rtol=1e-12, atol=0.0)


def _normal_neg_log_density(obs, mu, s2):
    """Negative log density of independent normals, written out longhand."""
    return -np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * np.log(s2) - (obs - mu) ** 2 / (2 * s2))


def test_negative_log_likelihood_matches_independent_gaussian_density():
    """NLL equals the summed normal negative log density over the differences.

    Uses perturbed measurements so the residual term is exercised alongside the
    normalising constant.
    """
    model = _model()
    sim_in = _sim_in()
    sigma_dep = 5e-4

    reference = _reference_reflectance()
    offsets = 1e-3 * np.array([[1.0, -2.0, 0.5], [-1.5, 0.25, 2.0], [0.75, 1.25, -1.0],
                               [-0.5, 0.0, 1.5]])
    ref_dat = _reflectance_data(reference + offsets)

    nll = model._negative_log_likelihood([MU_TILDE, sigma_dep], sim_in, ref_dat)

    s2total = model._compute_variance_of_measurements(
        sigma_dep, sim_in, reflectance_data=ref_dat
    )
    expected = _normal_neg_log_density(
        np.diff(ref_dat.average[F], axis=0), np.diff(reference, axis=0), s2total[F]
    )

    np.testing.assert_allclose(nll, expected, rtol=1e-12, atol=0.0)


def test_negative_log_likelihood_normalisation_sums_over_experiments():
    """The 2*pi term counts P*N terms per experiment, summed over experiments.

    Guards both the mirror dimension and the difference-vs-measurement-time count:
    with L experiments of P mirrors and N differences, a count of measurement times
    would give sum_f (N_f + 1) instead of sum_f N_f * P_f.
    """
    keys = [0, 1]
    tilts = {0: TILT, 1: TILT[::-1] + 5.0}  # distinct tilt histories per experiment

    model = _model()
    model.helios.tilt = tilts
    model.helios.inc_ref_factor = {f: INC_REF_FACTOR for f in keys}

    sim_in = types.SimpleNamespace(
        files=["experiment_0", "experiment_1"],
        time={f: np.arange(TILT.shape[1], dtype=float) for f in keys},
        dust=types.SimpleNamespace(PM10={f: DENSITY for f in keys}),
        dust_concentration={f: DUST_CONCENTRATION for f in keys},
        dust_type={f: "PM10" for f in keys},
    )
    rho0 = {f: RHO0 for f in keys}
    sigma_dep = 5e-4

    # Exact model data for both experiments, so only the normalising terms remain.
    model.predict_soiling_factor(sim_in, rho0=rho0, verbose=False)
    stub = types.SimpleNamespace(prediction_indices={f: PREDICTION_INDICES for f in keys})
    average = {f: model._predicted_reflectance(f, stub) for f in keys}

    ref_dat = types.SimpleNamespace(
        files=sim_in.files,
        times={f: np.asarray(PREDICTION_INDICES, dtype=float) for f in keys},
        average=average,
        rho0=rho0,
        prediction_indices={f: PREDICTION_INDICES for f in keys},
        sigma_of_the_mean={f: np.full_like(average[f], 1e-3) for f in keys},
    )

    nll = model._negative_log_likelihood([MU_TILDE, sigma_dep], sim_in, ref_dat)

    s2total = model._compute_variance_of_measurements(
        sigma_dep, sim_in, reflectance_data=ref_dat
    )
    n_terms = sum(s2total[f].size for f in keys)
    expected = 0.5 * n_terms * np.log(2 * np.pi) + 0.5 * sum(
        np.sum(np.log(s2total[f])) for f in keys
    )

    n_differences = len(PREDICTION_INDICES) - 1
    assert n_terms == len(keys) * n_differences * TILT.shape[0]
    np.testing.assert_allclose(nll, expected, rtol=1e-12, atol=0.0)
