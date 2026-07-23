"""
Unit / characterization tests for core deterministic model components.

These tests pin numerical behaviour so that future refactors that change a
result *without* raising an exception are caught. They deliberately avoid the
notebook/SolarPILOT path and any data under ``data/`` (gitignored); everything
here runs on small synthetic inputs and is fast.

Three groups:

1. ``test_dust_distribution_*``
   Round-trip and mean-shift identities for the dust size-distribution
   conversions in ``heliosoil.dust_distributions``. The conversions have
   closed forms, so a correct implementation must recover the original mixture
   parameters exactly under an inverse conversion.

2. ``test_constant_mean_delta_soiled_area``
   Pins ``ConstantMeanBase.calculate_delta_soiled_area`` (the core of
   ``SimplifiedFieldModel``) against an independent vectorised reference of
   ``alpha * cos(tilt) * mu_tilde``.

3. ``test_physical_delta_soiled_area``
   Pins ``PhysicalBase.calculate_delta_soiled_area`` (the core of
   ``FieldModel``) against an independent vectorised reference of the
   extinction-weighted area integral.

The reference computations are written independently of the implementation
under test, so they guard the formulas rather than restating them.
"""

import types
import numpy as np
import pytest

from heliosoil.dust_distributions import GaussianMixtureModel, NumberDistribution, MassDistribution, AreaDistribution
from heliosoil.base_models import ConstantMeanBase, PhysicalBase


# numpy>=2 exposes trapezoid; fall back to trapz on older numpy.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz"))

# Tight tolerance: these are exact analytic identities / re-derivations.
RTOL = 1e-10
ATOL = 0.0


# ---------------------------------------------------------------------------
# 1. Dust distribution conversions
# ---------------------------------------------------------------------------

# A non-trivial multi-component mixture (parameters in log10(D/1um) space).
WEIGHTS = np.array([0.6, 0.3, 0.1])
MUS = np.array([-0.4, 0.2, 0.9])
SIGMAS = np.array([0.30, 0.22, 0.45])
RHO = 2.65  # particle density [g/cm^3]


def _gmm():
    return GaussianMixtureModel(WEIGHTS.copy(), MUS.copy(), SIGMAS.copy())


def _assert_gmm_equal(got: GaussianMixtureModel, ref: GaussianMixtureModel):
    np.testing.assert_allclose(got.weights, ref.weights, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got.mus, ref.mus, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got.sigmas, ref.sigmas, rtol=RTOL, atol=ATOL)


def test_dust_distribution_number_mass_roundtrip():
    """Number -> Mass -> Number recovers the original mixture parameters.

    Verified analytically to be an exact identity: the exponential prefactors
    in the two conversions are conjugate and cancel.
    """
    start = NumberDistribution(_gmm())
    back = start.to_mass(rho=RHO).to_number(rho=RHO)
    _assert_gmm_equal(back.distribution, start.distribution)


def test_dust_distribution_mass_number_roundtrip():
    """Mass -> Number -> Mass recovers the original mixture parameters."""
    start = MassDistribution(_gmm())
    back = start.to_number(rho=RHO).to_mass(rho=RHO)
    _assert_gmm_equal(back.distribution, start.distribution)


def test_dust_distribution_number_area_roundtrip():
    """Number -> Area -> Number recovers the full mixture (weights, mus, sigmas).

    Verified analytically to be an exact identity: the forward weight factor
    exp(2 mu/ln10 + 2 sig^2/ln10^2) and the inverse factor cancel, and the
    +/- 2 sig^2/ln10 mean shifts undo each other.
    """
    start = NumberDistribution(_gmm())
    back = start.to_area().to_number()
    _assert_gmm_equal(back.distribution, start.distribution)


def test_dust_distribution_area_mass_roundtrip():
    """Area -> Mass -> Area recovers the mixture (exercises the routed paths)."""
    start = AreaDistribution(_gmm())
    back = start.to_mass(rho=RHO).to_area(rho=RHO)
    _assert_gmm_equal(back.distribution, start.distribution)


def test_dust_distribution_sigma_invariant():
    """Every conversion leaves the component widths (sigmas) unchanged."""
    num = NumberDistribution(_gmm())
    for dist in (num.to_mass(rho=RHO), num.to_area(), num.to_mass(rho=RHO).to_area(rho=RHO)):
        np.testing.assert_allclose(dist.distribution.sigmas, SIGMAS, rtol=RTOL, atol=ATOL)


def test_dust_distribution_mean_shifts():
    """
    Mean shifts match the closed forms, written here in an independent
    constant convention: the implementation uses sig^2 / log10(e); this test
    uses the equivalent sig^2 * ln(10).
    """
    num = NumberDistribution(_gmm())

    # number -> area: mus shift by +2 sig^2 ln(10)
    area_mus = num.to_area().distribution.mus
    np.testing.assert_allclose(area_mus, MUS + 2 * SIGMAS**2 * np.log(10), rtol=RTOL, atol=ATOL)

    # number -> mass: mus shift by +3 sig^2 ln(10)
    mass_mus = num.to_mass(rho=RHO).distribution.mus
    np.testing.assert_allclose(mass_mus, MUS + 3 * SIGMAS**2 * np.log(10), rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Shared helpers for the soiling-rate tests
# ---------------------------------------------------------------------------


def _helios_stub(**arrays):
    """A minimal stand-in for the Heliostats object.

    Only the attributes read/written by ``calculate_delta_soiled_area`` are
    provided; output dicts start empty and are populated by the method.
    """
    helios = types.SimpleNamespace(delta_soiled_area={}, delta_soiled_area_variance={})
    for name, value in arrays.items():
        setattr(helios, name, value)
    return helios


# ---------------------------------------------------------------------------
# 2. ConstantMeanBase.calculate_delta_soiled_area  (SimplifiedFieldModel core)
# ---------------------------------------------------------------------------


def test_constant_mean_delta_soiled_area():
    """delta = (c_dust / density) * cos(tilt) * mu_tilde, elementwise."""
    f = 0
    mu_tilde = 0.65
    density = 35.0  # PM10 reference value [arbitrary, consistent units]

    # tilt deliberately includes 0 deg (cos=1) and 90 deg (cos=0) limits.
    tilt = np.array([[0.0, 30.0, 60.0, 90.0], [10.0, 45.0, 80.0, 15.0]])
    dust_conc = np.array([20.0, 50.0, 80.0, 110.0])

    model = ConstantMeanBase()
    model.mu_tilde = mu_tilde
    model.sigma_dep = None
    model.helios = _helios_stub(tilt={f: tilt})

    sim_in = types.SimpleNamespace(
        time={f: None},  # only keys are used
        dust=types.SimpleNamespace(PM10={f: density}),
        dust_concentration={f: dust_conc},
        dust_type={f: "PM10"},
    )

    model.calculate_delta_soiled_area(sim_in, verbose=False)

    alpha = dust_conc / density  # shape (N_times,)
    expected = alpha[None, :] * np.cos(np.radians(tilt)) * mu_tilde
    np.testing.assert_allclose(model.helios.delta_soiled_area[f], expected, rtol=RTOL, atol=ATOL)

    # Explicit limit checks: tilt=0 -> alpha*mu_tilde; tilt=90 -> 0.
    np.testing.assert_allclose(model.helios.delta_soiled_area[f][0, 0], alpha[0] * mu_tilde, rtol=RTOL)
    assert abs(model.helios.delta_soiled_area[f][0, 3]) < 1e-12


def test_constant_mean_delta_soiled_area_variance():
    """With sigma_dep set, variance = sigma_dep^2 * alpha^2 * cos(tilt)^2."""
    f = 0
    mu_tilde = 0.5
    sigma_dep = 0.12
    density = 40.0

    tilt = np.array([[5.0, 25.0, 55.0]])
    dust_conc = np.array([30.0, 60.0, 90.0])

    model = ConstantMeanBase()
    model.mu_tilde = mu_tilde
    model.sigma_dep = sigma_dep
    model.helios = _helios_stub(tilt={f: tilt})

    sim_in = types.SimpleNamespace(
        time={f: None}, dust=types.SimpleNamespace(PM10={f: density}), dust_concentration={f: dust_conc}, dust_type={f: "PM10"}
    )

    model.calculate_delta_soiled_area(sim_in, verbose=False)

    alpha = dust_conc / density
    expected_var = sigma_dep**2 * (alpha**2 * np.cos(np.radians(tilt)) ** 2)
    np.testing.assert_allclose(model.helios.delta_soiled_area_variance[f], expected_var, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 3. PhysicalBase.calculate_delta_soiled_area  (FieldModel core)
# ---------------------------------------------------------------------------


def test_physical_delta_soiled_area():
    """
    delta[i,j] = alpha[j] * (pi/4) *
                 integral_over_log10(D) [ pdfqN[i,j,:] * (D_m^2) * dt * Qext[i,:] ]

    pdfqN already carries the cos(tilt) dependence, so tilt enters only through
    that array here.
    """
    f = 0
    density = 50.0
    dt = 3600.0  # s

    # Diameter grid in micrometres (log-spaced, as in the model).
    D_um = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
    N_D = D_um.size
    N_helios, N_times = 2, 3

    rng = np.random.default_rng(0)
    pdfqN = rng.uniform(0.1, 1.0, size=(N_helios, N_times, N_D))
    extinction_weighting = rng.uniform(0.5, 2.0, size=(N_helios, N_D))
    tilt = np.array([[10.0, 40.0, 70.0], [20.0, 50.0, 80.0]])  # only for shape
    dust_conc = np.array([25.0, 75.0, 125.0])

    model = PhysicalBase()
    model.sigma_dep = None
    model.helios = _helios_stub(tilt={f: tilt}, pdfqN={f: pdfqN}, extinction_weighting={f: extinction_weighting})

    sim_in = types.SimpleNamespace(
        wind_speed={f: np.zeros(N_times)},  # only keys are used
        dust=types.SimpleNamespace(D={f: D_um}, PM10={f: density}),
        dust_concentration={f: dust_conc},
        dust_type={f: "PM10"},
        dt={f: dt},
    )

    model.calculate_delta_soiled_area(sim_in, verbose=False)

    # Independent vectorised reference.
    alpha = dust_conc / density  # (N_times,)
    D_m = D_um * 1e-6
    log10_D = np.log10(D_um)
    integrand = pdfqN * (D_m**2)[None, None, :] * dt * extinction_weighting[:, None, :]
    integral = _trapezoid(integrand, log10_D, axis=2)  # (N_helios, N_times)
    expected = alpha[None, :] * (np.pi / 4.0) * integral

    np.testing.assert_allclose(model.helios.delta_soiled_area[f], expected, rtol=RTOL, atol=ATOL)


def test_physical_delta_soiled_area_scales_with_concentration():
    """delta is linear in the airborne dust concentration (alpha)."""
    f = 0
    density = 50.0
    dt = 3600.0
    D_um = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
    N_D = D_um.size
    N_helios, N_times = 2, 3

    rng = np.random.default_rng(1)
    pdfqN = rng.uniform(0.1, 1.0, size=(N_helios, N_times, N_D))
    Qext = rng.uniform(0.5, 2.0, size=(N_helios, N_D))
    tilt = np.zeros((N_helios, N_times))
    base_conc = np.array([25.0, 75.0, 125.0])

    def run(conc):
        model = PhysicalBase()
        model.sigma_dep = None
        model.helios = _helios_stub(tilt={f: tilt}, pdfqN={f: pdfqN}, extinction_weighting={f: Qext})
        sim_in = types.SimpleNamespace(
            wind_speed={f: np.zeros(N_times)},
            dust=types.SimpleNamespace(D={f: D_um}, PM10={f: density}),
            dust_concentration={f: conc},
            dust_type={f: "PM10"},
            dt={f: dt},
        )
        model.calculate_delta_soiled_area(sim_in, verbose=False)
        return model.helios.delta_soiled_area[f]

    np.testing.assert_allclose(run(3.0 * base_conc), 3.0 * run(base_conc), rtol=RTOL, atol=ATOL)
