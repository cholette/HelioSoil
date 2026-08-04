"""
Tests for the multi-mirror variance-components likelihood.

Mirrors observed simultaneously at one site do not receive independent deposition
noise: a dusty or gusty interval loads every mirror at once. The deposition noise is
therefore split into a common component ``sigma_c**2 = kappa * sigma_dep**2``, shared by
all mirrors during an interval, and a mirror-specific component
``sigma_m**2 = (1 - kappa) * sigma_dep**2``. Each experiment then contributes one
multivariate normal over its stacked reflectance differences rather than a product of
univariate ones.

References are built independently of the implementation -- from the scalar likelihood
and from the Kronecker form of the covariance -- so they guard the formulas rather than
restate them.

1. ``test_reduces_*`` / ``test_single_mirror_*`` -- the model nests the scalar one.
2. ``test_covariance_*`` -- assembly against the Kronecker form, and positive
   definiteness over the admissible range of kappa.
"""

import types
import numpy as np

from heliosoil.fitting import ConstantMeanDeposition
from heliosoil.base_models import ConstantMeanBase


RTOL = 1e-10
ATOL = 0.0

F = 0
NOMINAL_REFLECTANCE = 0.95
MU_TILDE = 0.015
DENSITY = 40.0
INC_REF_FACTOR = 2.0 / np.cos(np.radians(15.0))

T_GRID = 21
PREDICTION_INDICES = [0, 4, 9, 17, 20]  # uneven spacing: dk = 4, 5, 8, 3
N_DIFF = len(PREDICTION_INDICES) - 1


def _model(tilt, sigma_dep=None):
    """A ConstantMeanDeposition with site data stubbed instead of read from Excel."""
    model = ConstantMeanDeposition.__new__(ConstantMeanDeposition)
    ConstantMeanBase.__init__(model)
    model.mu_tilde, model.sigma_dep = MU_TILDE, sigma_dep
    model.helios.nominal_reflectance = NOMINAL_REFLECTANCE
    model.helios.tilt = {F: tilt}
    model.helios.inc_ref_factor = {F: np.array(INC_REF_FACTOR)}
    for name in ("delta_soiled_area", "delta_soiled_area_variance",
                 "soiling_factor", "soiling_factor_prediction_variance"):
        setattr(model.helios, name, {})
    return model


def _sim_in():
    return types.SimpleNamespace(
        files=["experiment_0"],
        time={F: np.arange(T_GRID, dtype=float)},
        dust=types.SimpleNamespace(PM10={F: DENSITY}),
        dust_concentration={F: np.linspace(20.0, 90.0, T_GRID)},
        dust_type={F: "PM10"},
    )


def _fixture(n_mirrors, common_tilt=False, common_sigma=False, seed=0):
    """Model, inputs and perturbed measurements for ``n_mirrors`` mirrors."""
    rng = np.random.default_rng(seed)
    if common_tilt:
        tilt = np.tile(np.linspace(5.0, 40.0, T_GRID), (n_mirrors, 1))
    else:
        tilt = rng.uniform(0.0, 60.0, size=(n_mirrors, T_GRID))

    model = _model(tilt)
    sim_in = _sim_in()

    rho0 = {F: np.full(n_mirrors, 0.93)}
    model.predict_soiling_factor(sim_in, rho0=rho0, verbose=False)
    stub = types.SimpleNamespace(prediction_indices={F: PREDICTION_INDICES})
    exact = model._predicted_reflectance(F, stub)

    if common_sigma:
        sigma = np.tile(np.linspace(1e-3, 2e-3, len(PREDICTION_INDICES))[:, None],
                        (1, n_mirrors))
    else:
        sigma = rng.uniform(5e-4, 2e-3, size=exact.shape)

    ref_dat = types.SimpleNamespace(
        files=["experiment_0"],
        times={F: np.asarray(PREDICTION_INDICES, dtype=float)},
        average={F: exact + 1e-3 * rng.standard_normal(exact.shape)},
        rho0=rho0,
        prediction_indices={F: PREDICTION_INDICES},
        sigma_of_the_mean={F: sigma},
    )
    return model, sim_in, ref_dat


# ---------------------------------------------------------------------------
# 1. Nesting of the scalar model
# ---------------------------------------------------------------------------


def test_reduces_to_scalar_likelihood_when_kappa_is_zero():
    """kappa = 0 with independent differences recovers the scalar likelihood exactly."""
    model, sim_in, ref_dat = _fixture(n_mirrors=4)
    sigma_dep = 8e-4

    scalar = model._negative_log_likelihood([MU_TILDE, sigma_dep], sim_in, ref_dat)
    components = model._negative_log_likelihood_components(
        [MU_TILDE, sigma_dep, 0.0], sim_in, ref_dat, endpoint_correction=False
    )

    np.testing.assert_allclose(components, scalar, rtol=1e-11, atol=ATOL)


def test_single_mirror_likelihood_does_not_depend_on_kappa():
    """With one mirror only the total sigma_dep is identified, not the split."""
    model, sim_in, ref_dat = _fixture(n_mirrors=1)
    sigma_dep = 8e-4

    values = [
        model._negative_log_likelihood_components(
            [MU_TILDE, sigma_dep, kappa], sim_in, ref_dat, endpoint_correction=True
        )
        for kappa in (0.0, 0.25, 0.5, 1.0)
    ]
    for value in values[1:]:
        np.testing.assert_allclose(value, values[0], rtol=1e-12, atol=ATOL)


# ---------------------------------------------------------------------------
# 2. Covariance assembly
# ---------------------------------------------------------------------------


def test_covariance_matches_kronecker_form_for_common_tilt():
    """Common tilt and measurement noise give (sc^2 11' + sm^2 I) kron S + I kron R."""
    n_mirrors = 3
    model, sim_in, ref_dat = _fixture(n_mirrors, common_tilt=True, common_sigma=True)
    sigma_dep, kappa = 7e-4, 0.4
    s2_common, s2_mirror = kappa * sigma_dep**2, (1 - kappa) * sigma_dep**2

    got = model._experiment_covariance(F, sigma_dep, kappa, sim_in, ref_dat)

    # Independent reference: per-difference dust loading S_i and tridiagonal R.
    m = model._loading_matrix(F, sim_in)
    b = model._reflectance_loss_factor(F)
    S = np.diag([
        b**2 * np.sum(m[PREDICTION_INDICES[i] + 1 : PREDICTION_INDICES[i + 1] + 1, 0] ** 2)
        for i in range(N_DIFF)
    ])

    s2 = ref_dat.sigma_of_the_mean[F][:, 0] ** 2
    R = np.diag(s2[1:] + s2[:-1])
    for i in range(N_DIFF - 1):
        R[i, i + 1] = R[i + 1, i] = -s2[i + 1]

    ones = np.ones((n_mirrors, n_mirrors))
    expected = np.kron(s2_common * ones + s2_mirror * np.eye(n_mirrors), S) + np.kron(
        np.eye(n_mirrors), R
    )

    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=1e-300)


def test_covariance_is_positive_definite_across_kappa():
    """Cholesky succeeds over the whole admissible range of kappa."""
    model, sim_in, ref_dat = _fixture(n_mirrors=5)
    for kappa in np.linspace(0.0, 1.0, 11):
        cov = model._experiment_covariance(F, 9e-4, kappa, sim_in, ref_dat)
        np.testing.assert_allclose(cov, cov.transpose(), rtol=RTOL, atol=ATOL)
        np.linalg.cholesky(cov)  # raises LinAlgError if not positive definite
