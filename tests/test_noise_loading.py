"""
Tests for the noise loading of a wind-driven soiling mechanism.

Each mechanism in heliosoil.horizontal_impaction carries a single deposition-noise term
whose loading is the sum of that mechanism's mean bases -- for the two-face mechanisms
(normal_wind, impaction_retention) only one face is ever nonzero at a time, so one shared
noise term suffices. Its variance contribution is therefore the square of that loading:

    Var contribution of mechanism c  =  sigma_c**2 * ( sum_k x_{c,k} )**2

This file pins that relation two ways:

1. ``test_variance_basis_matches_reference`` -- every mechanism's variance basis against a
   reference expression written out independently here, in plain numpy, from the module
   docstring's formulae. These are the five expressions the registry used to carry as
   hand-written functions; keeping them here means the derived form cannot silently drift
   from what was verified.
2. ``test_variance_basis_is_the_squared_noise_loading`` / ``..._is_the_summed_mean_bases`` --
   the derivation itself, so a mechanism added later gets the same treatment for free.

The nonnegativity check matters beyond bookkeeping: the multi-mirror covariance assembly
weights *products* of loadings across mirror pairs, so a loading that could go negative
would silently flip the sign of a cross-mirror covariance term.
"""

import types

import numpy as np
import pytest

from heliosoil.horizontal_impaction import _COMPONENTS, COMPONENT_KEYS, check_tilt_range

RTOL = 1e-12
ATOL = 0.0

N_HELIOS, N_TIMES = 4, 32


def _inputs(seed=0, tilt_low=0.0, tilt_high=180.0):
    """Randomised loading inputs, in the shapes the registry is called with."""
    rng = np.random.default_rng(seed)
    return dict(
        alpha=rng.uniform(0.2, 3.0, size=N_TIMES),
        tilt=rng.uniform(tilt_low, tilt_high, size=(N_HELIOS, N_TIMES)),
        azimuth=np.tile(rng.uniform(0.0, 360.0, size=N_HELIOS)[:, None], (1, N_TIMES)),
        wind_dir=rng.uniform(0.0, 360.0, size=N_TIMES),
        wind_speed=rng.uniform(0.5, 12.0, size=N_TIMES),
    )


# ---------------------------------------------------------------------------
# Reference expressions, written independently of the library
# ---------------------------------------------------------------------------


def _cos_plus(deg):
    return np.maximum(0.0, np.cos(np.radians(deg)))


def _faces(tilt, azimuth, wind_dir):
    """Windward / leeward normal-impaction projections, from the module docstring."""
    cos_dg = np.cos(np.radians(azimuth - wind_dir[None, :]))
    s = np.sin(np.radians(tilt))
    return s * np.maximum(0.0, cos_dg), s * np.maximum(0.0, -cos_dg)


def _reference_variance_basis(key, alpha, tilt, azimuth, wind_dir, wind_speed):
    a = alpha[None, :]
    v = wind_speed[None, :]
    if key == "gravitational":
        return (a * _cos_plus(tilt)) ** 2
    if key == "turbulent_wind":
        return (a * _cos_plus(tilt) * v) ** 2
    if key == "normal_wind":
        p_w, p_l = _faces(tilt, azimuth, wind_dir)
        return (a * v) ** 2 * (p_w + p_l) ** 2
    if key == "tangential_wind":
        cos_dg = np.cos(np.radians(azimuth - wind_dir[None, :]))
        tangential = np.sqrt(
            np.clip(1.0 - (np.sin(np.radians(tilt)) * cos_dg) ** 2, 0.0, None)
        )
        return (a * v * tangential) ** 2
    if key == "impaction_retention":
        p_w, p_l = _faces(tilt, azimuth, wind_dir)
        retention = _cos_plus(tilt)
        return (a * v) ** 2 * (retention * (p_w + p_l)) ** 2
    raise AssertionError(f"no reference expression for {key!r}")


# ---------------------------------------------------------------------------
# 1. The variance basis of every mechanism, against the reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", COMPONENT_KEYS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_variance_basis_matches_reference(key, seed):
    """Every mechanism's variance basis equals the expression it was verified against."""
    inputs = _inputs(seed)
    got = _COMPONENTS[key].variance_basis(**inputs)
    expected = _reference_variance_basis(key, **inputs)
    assert got.shape == (N_HELIOS, N_TIMES)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("key", COMPONENT_KEYS)
def test_variance_basis_matches_reference_past_vertical(key):
    """Same, restricted to face-down mirrors, where the cos clip is what is being tested."""
    inputs = _inputs(seed=3, tilt_low=90.0, tilt_high=180.0)
    np.testing.assert_allclose(
        _COMPONENTS[key].variance_basis(**inputs),
        _reference_variance_basis(key, **inputs),
        rtol=RTOL,
        atol=ATOL,
    )


# ---------------------------------------------------------------------------
# 2. The derivation: loading = sum of mean bases, variance basis = loading**2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", COMPONENT_KEYS)
def test_noise_loading_is_the_summed_mean_bases(key):
    """The loading of a mechanism's single noise term is the sum of its mean bases."""
    inputs = _inputs(seed=4)
    component = _COMPONENTS[key]
    summed = sum(component.mean_bases(**inputs))
    np.testing.assert_allclose(
        component.noise_loading(**inputs), summed, rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("key", COMPONENT_KEYS)
def test_variance_basis_is_the_squared_noise_loading(key):
    """variance_basis is derived from the loading, not maintained separately."""
    inputs = _inputs(seed=5)
    component = _COMPONENTS[key]
    np.testing.assert_allclose(
        component.variance_basis(**inputs),
        component.noise_loading(**inputs) ** 2,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("key", COMPONENT_KEYS)
@pytest.mark.parametrize("seed", [6, 7])
def test_noise_loading_is_nonnegative_over_the_physical_tilt_range(key, seed):
    """Loadings must not go negative over tilt in [0, 180]: the multi-mirror covariance
    weights their PRODUCTS across mirror pairs, so a negative loading would silently flip
    the sign of a cross-mirror covariance term rather than fail loudly."""
    inputs = _inputs(seed, tilt_low=0.0, tilt_high=180.0)
    loading = _COMPONENTS[key].noise_loading(**inputs)
    assert np.all(loading >= 0.0), f"{key}: min loading {loading.min()}"
    assert np.all(np.isfinite(loading))


@pytest.mark.parametrize("key", ["normal_wind", "impaction_retention"])
def test_noise_loading_precondition_tilt_within_half_turn(key):
    """Pins the precondition behind the test above, so it is visible rather than assumed.

    wind_projection_factors clips cos(Delta_gamma) but NOT sin(tilt), so the windward and
    leeward projections -- and hence these two mechanisms' loadings -- go negative once tilt
    leaves [0, 180]. This is why `check_tilt_range` guards the model's entry point: the
    geometry primitives themselves stay unguarded (they are called per likelihood
    evaluation), so the range is enforced once, where the tilts arrive.
    """
    inputs = _inputs(seed=9, tilt_low=-40.0, tilt_high=-5.0)  # deliberately unphysical
    loading = _COMPONENTS[key].noise_loading(**inputs)
    assert np.any(loading < 0.0), (
        f"{key}: loadings no longer go negative below tilt=0 -- if sin(tilt) is now clipped, "
        "delete this test and widen the range in the nonnegativity test above"
    )


# ---------------------------------------------------------------------------
# 3. The tilt-range guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_tilt", [-0.5, -30.0, 180.5, 270.0, 361.0])
def test_check_tilt_range_rejects_tilts_outside_the_half_turn(bad_tilt):
    tilt = np.array([[0.0, 45.0, bad_tilt]])
    with pytest.raises(ValueError, match=r"outside \[0, 180\]"):
        check_tilt_range(tilt, f=0)


@pytest.mark.parametrize("good_tilt", [0.0, 45.0, 90.0, 179.9, 180.0])
def test_check_tilt_range_accepts_the_physical_range(good_tilt):
    check_tilt_range(np.full((2, 3), good_tilt), f=0)  # must not raise


def test_check_tilt_range_tolerates_float_noise_at_the_endpoints():
    """Values meant to be exactly 0 or 180 can arrive a hair outside after arithmetic."""
    check_tilt_range(np.array([[-1e-12, 180.0 + 1e-12]]), f=0)


def test_check_tilt_range_ignores_non_finite_tilts():
    """Real campaign files carry NaN tilts for mirrors not yet in service -- the first
    yadnarie campaign has 728 of them -- so NaN must not be an error."""
    check_tilt_range(np.array([[np.nan, 30.0, np.nan]]), f=0)
    check_tilt_range(np.full((2, 2), np.nan), f=0)  # all-NaN: nothing to check


def test_calculate_delta_soiled_area_rejects_out_of_range_tilt():
    """The guard is wired into the model's entry point, not just available as a helper."""
    from heliosoil.base_models import ConstantMeanBase
    from heliosoil.horizontal_impaction import ConstantMeanWindBase

    model = ConstantMeanWindBase.__new__(ConstantMeanWindBase)
    ConstantMeanWindBase.__init__(model, components=["gravitational", "normal_wind"])
    model.mu_tilde, model.omega_windward, model.omega_leeward = 1e-5, 1e-7, 1e-7
    model.helios = types.SimpleNamespace(
        delta_soiled_area={},
        delta_soiled_area_variance={},
        tilt={0: np.array([[10.0, -20.0, 30.0]])},
        azimuth={0: np.zeros((1, 3))},
    )
    sim_in = types.SimpleNamespace(
        time={0: np.arange(3.0)},
        dust=types.SimpleNamespace(PM10={0: 40.0}),
        dust_concentration={0: np.full(3, 50.0)},
        dust_type={0: "PM10"},
        wind_speed={0: np.full(3, 4.0)},
        wind_direction={0: np.zeros(3)},
    )
    with pytest.raises(ValueError, match=r"outside \[0, 180\]"):
        model.calculate_delta_soiled_area(sim_in, verbose=False)


@pytest.mark.parametrize("key", COMPONENT_KEYS)
def test_noise_loading_vanishes_exactly_where_the_mean_does(key):
    """A mechanism that deposits nothing must also contribute no noise.

    Squaring is sign-blind, so a mechanism whose mean is clipped to zero but whose variance
    is not would give a face-down mirror zero predicted deposition and full deposition noise
    -- an inconsistency that feeds both the likelihood weights and the prediction interval.
    """
    inputs = _inputs(seed=8, tilt_low=0.0, tilt_high=180.0)
    component = _COMPONENTS[key]
    mean_total = sum(np.abs(b) for b in component.mean_bases(**inputs))
    variance = component.variance_basis(**inputs)
    np.testing.assert_array_equal(mean_total == 0.0, variance == 0.0)
