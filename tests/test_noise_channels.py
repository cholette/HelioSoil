"""
Tests for the noise-channel interface.

A model's deposition noise is a sum of independent channels, each a loading array paired
with a magnitude. ``noise_channels`` is the single place a model describes them; everything
that consumes noise -- the per-difference variance today, the multi-mirror covariance next
-- reads that list rather than reaching into the model's internals. A model therefore gains
noise handling by describing its channels, not by reimplementing the assembly.

The plain constant-mean and semi-physical models have one channel; the wind model has one
per active mechanism. The point of these tests is that both are consumed by the *same*
variance assembly in CommonFittingMethods: the wind model no longer carries its own copy.

Sections:

1. What each model reports as its channels, and that the loadings are the ones the rest of
   the code already uses (``gravitational_settling_factor`` here, ``noise_loading`` there).
2. That the variance really is assembled from the channels -- probed by perturbing a
   channel's magnitude and checking only the expected contribution moves.
3. ``NoiseChannel.scale``, the fixed RMS loading that makes magnitudes of different
   mechanisms comparable (soiling model notes, eq. 88).
"""

import types

import numpy as np
import pytest

from heliosoil.base_models import ConstantMeanBase
from heliosoil.fitting import ConstantMeanDeposition, NoiseChannel
from heliosoil.horizontal_impaction import ConstantMeanWindBase, ConstantMeanWindDeposition, _COMPONENTS
from heliosoil.utilities import gravitational_settling_factor

RTOL = 1e-12
ATOL = 0.0

F = 0
T_GRID = 24
PREDICTION_INDICES = [0, 5, 11, 19, 23]
N_DIFF = len(PREDICTION_INDICES) - 1
DENSITY = 40.0
NOMINAL_REFLECTANCE = 0.95
INC_REF_FACTOR = 2.0 / np.cos(np.radians(15.0))
SIGMA_DEP = 8e-4


def _blank(model):
    for name in ("delta_soiled_area", "delta_soiled_area_variance", "soiling_factor", "soiling_factor_prediction_variance"):
        setattr(model.helios, name, {})


def _constant_mean_model(tilt, sigma_dep=SIGMA_DEP):
    model = ConstantMeanDeposition.__new__(ConstantMeanDeposition)
    ConstantMeanBase.__init__(model)
    model.mu_tilde, model.sigma_dep = 0.015, sigma_dep
    model.helios.nominal_reflectance = NOMINAL_REFLECTANCE
    model.helios.tilt = {F: tilt}
    model.helios.inc_ref_factor = {F: np.array(INC_REF_FACTOR)}
    _blank(model)
    return model


def _wind_model(components, tilt, azimuth, sigmas=None):
    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanWindBase.__init__(model, components=components)
    model.verbose = False
    for name, value in (
        ("mu_tilde", 0.015),
        ("omega_turbulent", 3e-6),
        ("omega_windward", 6e-7),
        ("omega_leeward", 9e-7),
        ("omega_tangential", 5e-7),
        ("omega_ret_windward", 4e-7),
        ("omega_ret_leeward", 7e-7),
    ):
        setattr(model, name, value)
    for i, name in enumerate(model._sigma_param_names):
        setattr(model, name, SIGMA_DEP * (1.0 + 0.4 * i) if sigmas is None else sigmas[i])
    model.helios.nominal_reflectance = NOMINAL_REFLECTANCE
    model.helios.tilt = {F: tilt}
    model.helios.azimuth = {F: azimuth}
    model.helios.inc_ref_factor = {F: np.array(INC_REF_FACTOR)}
    _blank(model)
    return model


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


def _ref_dat(n_mirrors, measurement_sigma=1e-3):
    shape = (len(PREDICTION_INDICES), n_mirrors)
    return types.SimpleNamespace(
        files=["experiment_0"],
        times={F: np.asarray(PREDICTION_INDICES, dtype=float)},
        prediction_indices={F: PREDICTION_INDICES},
        rho0={F: np.full(n_mirrors, 0.93)},
        sigma_of_the_mean={F: np.full(shape, measurement_sigma)},
        average={F: np.zeros(shape)},
    )


CONCENTRATION = np.linspace(20.0, 90.0, T_GRID)
WIND_SPEED = np.linspace(1.0, 9.0, T_GRID)
WIND_DIRECTION = np.linspace(0.0, 350.0, T_GRID)


# ---------------------------------------------------------------------------
# 1. What each model reports
# ---------------------------------------------------------------------------


def test_constant_mean_model_reports_one_channel():
    """One channel: dust loading times the horizontal projection, carrying sigma_dep."""
    tilt = np.linspace(0.0, 80.0, 3)[:, None] * np.ones((1, T_GRID))
    model = _constant_mean_model(tilt)
    sim_in = _sim_in(CONCENTRATION)

    channels = model.noise_channels(F, sim_in)

    assert len(channels) == 1
    assert channels[0].name == "sigma_dep"
    assert channels[0].sigma == SIGMA_DEP
    assert channels[0].kappa == 0.0
    expected = (CONCENTRATION / DENSITY)[None, :] * gravitational_settling_factor(tilt)
    np.testing.assert_allclose(channels[0].loading, expected, rtol=RTOL, atol=ATOL)


def test_constant_mean_channel_honours_the_sigma_override():
    """Callers may pass a candidate magnitude positionally instead of setting it first."""
    model = _constant_mean_model(np.full((2, T_GRID), 30.0))
    channels = model.noise_channels(F, _sim_in(CONCENTRATION), sigma_override=3e-3)
    assert channels[0].sigma == 3e-3


@pytest.mark.parametrize(
    "components",
    [
        ["gravitational"],
        ["gravitational", "normal_wind"],
        ["turbulent_wind", "tangential_wind"],
        ["gravitational", "normal_wind", "tangential_wind", "impaction_retention"],
    ],
)
def test_wind_model_reports_one_channel_per_mechanism(components):
    """Channel order, names and loadings follow the active mechanisms."""
    n_mirrors = 3
    tilt = np.full((n_mirrors, T_GRID), 40.0)
    azimuth = np.tile(np.array([0.0, 120.0, 240.0])[:, None], (1, T_GRID))
    model = _wind_model(components, tilt, azimuth)
    sim_in = _sim_in(CONCENTRATION, WIND_SPEED, WIND_DIRECTION)

    channels = model.noise_channels(F, sim_in)

    assert [c.name for c in channels] == list(model._sigma_param_names)
    assert len(channels) == len(model.components)
    alpha = CONCENTRATION / DENSITY
    for channel, component in zip(channels, model.components):
        expected = component.noise_loading(alpha, tilt, azimuth, WIND_DIRECTION, WIND_SPEED)
        np.testing.assert_allclose(channel.loading, expected, rtol=RTOL, atol=ATOL)
        assert channel.sigma == getattr(model, component.sigma_param_name)


def test_wind_model_channels_validate_geometry_like_the_mean_path():
    """The channels go through the same tilt/wind validation the mean assembly does."""
    model = _wind_model(["gravitational", "normal_wind"], np.full((1, T_GRID), 200.0), np.zeros((1, T_GRID)))
    with pytest.raises(ValueError, match=r"outside \[0, 180\]"):
        model.noise_channels(F, _sim_in(CONCENTRATION, WIND_SPEED, WIND_DIRECTION))


def test_all_loadings_are_nonnegative():
    """The covariance weights products of loadings, so their sign matters."""
    tilt = np.linspace(0.0, 180.0, 5)[:, None] * np.ones((1, T_GRID))
    azimuth = np.tile(np.linspace(0.0, 300.0, 5)[:, None], (1, T_GRID))
    model = _wind_model(list(_COMPONENTS), tilt, azimuth)
    for channel in model.noise_channels(F, _sim_in(CONCENTRATION, WIND_SPEED, WIND_DIRECTION)):
        assert np.all(channel.loading >= 0.0), channel.name


# ---------------------------------------------------------------------------
# 2. The variance really is assembled from the channels
# ---------------------------------------------------------------------------


def _variance(model, sim_in, ref_dat, sigma_arg=None):
    model.predict_soiling_factor(sim_in, reflectance_data=ref_dat, verbose=False)
    return model._compute_variance_of_measurements(sigma_arg, sim_in, reflectance_data=ref_dat)[F]


@pytest.mark.parametrize("components", [["gravitational", "normal_wind"], ["gravitational", "tangential_wind", "impaction_retention"]])
def test_variance_is_the_summed_channel_contributions(components):
    """The assembled variance equals sum_c sigma_c^2 b^2 (sum over window of loading_c^2)."""
    n_mirrors = 2
    tilt = np.full((n_mirrors, T_GRID), 35.0)
    azimuth = np.tile(np.array([10.0, 200.0])[:, None], (1, T_GRID))
    model = _wind_model(components, tilt, azimuth)
    sim_in, ref_dat = _sim_in(CONCENTRATION, WIND_SPEED, WIND_DIRECTION), _ref_dat(n_mirrors)

    got = _variance(model, sim_in, ref_dat)

    b = model._reflectance_loss_factor(F, ref_dat)
    windows = model._difference_windows(F, ref_dat)
    expected = np.zeros((N_DIFF, n_mirrors))
    for channel in model.noise_channels(F, sim_in):
        per_difference = np.stack([(channel.loading**2)[:, w].sum(axis=1) for w in windows])
        expected = expected + channel.sigma**2 * b**2 * per_difference
    expected += 2.0 * 1e-3**2  # the two measurement-noise terms

    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


def test_a_channel_with_no_magnitude_contributes_nothing():
    """A least-squares fit leaves sigmas unset; those channels must drop out silently."""
    n_mirrors = 2
    tilt = np.full((n_mirrors, T_GRID), 35.0)
    azimuth = np.tile(np.array([10.0, 200.0])[:, None], (1, T_GRID))
    sim_in, ref_dat = _sim_in(CONCENTRATION, WIND_SPEED, WIND_DIRECTION), _ref_dat(n_mirrors)

    both = _wind_model(["gravitational", "normal_wind"], tilt, azimuth)
    only_gravitational = _wind_model(["gravitational", "normal_wind"], tilt, azimuth)
    only_gravitational.sigma_dep_gamma = None
    gravitational_alone = _wind_model(["gravitational"], tilt, azimuth)

    with_both = _variance(both, sim_in, ref_dat)
    with_one_muted = _variance(only_gravitational, sim_in, ref_dat)
    with_one_active = _variance(gravitational_alone, sim_in, ref_dat)

    assert np.all(with_both > with_one_muted)
    np.testing.assert_allclose(with_one_muted, with_one_active, rtol=RTOL, atol=ATOL)


def test_perturbing_one_channel_moves_only_its_own_contribution():
    """Doubling a mechanism's sigma quadruples exactly that mechanism's share."""
    n_mirrors = 2
    tilt = np.full((n_mirrors, T_GRID), 50.0)
    azimuth = np.tile(np.array([30.0, 210.0])[:, None], (1, T_GRID))
    sim_in, ref_dat = _sim_in(CONCENTRATION, WIND_SPEED, WIND_DIRECTION), _ref_dat(n_mirrors, 0.0)

    base = _wind_model(["gravitational", "normal_wind"], tilt, azimuth)
    doubled = _wind_model(["gravitational", "normal_wind"], tilt, azimuth)
    doubled.sigma_dep_gamma = base.sigma_dep_gamma * 2.0

    v_base = _variance(base, sim_in, ref_dat)
    v_doubled = _variance(doubled, sim_in, ref_dat)

    b = base._reflectance_loss_factor(F, ref_dat)
    windows = base._difference_windows(F, ref_dat)
    wind_channel = [c for c in base.noise_channels(F, sim_in) if c.name == "sigma_dep_gamma"][0]
    share = wind_channel.sigma**2 * b**2 * np.stack([(wind_channel.loading**2)[:, w].sum(axis=1) for w in windows])

    np.testing.assert_allclose(v_doubled - v_base, 3.0 * share, rtol=1e-9, atol=ATOL)


def test_single_channel_model_uses_the_same_assembly():
    """The constant-mean model goes through the shared consumer, not a private copy."""
    n_mirrors = 3
    tilt = np.linspace(10.0, 70.0, n_mirrors)[:, None] * np.ones((1, T_GRID))
    model = _constant_mean_model(tilt)
    sim_in, ref_dat = _sim_in(CONCENTRATION), _ref_dat(n_mirrors)

    got = _variance(model, sim_in, ref_dat, sigma_arg=SIGMA_DEP)

    channel = model.noise_channels(F, sim_in)[0]
    b = model._reflectance_loss_factor(F, ref_dat)
    windows = model._difference_windows(F, ref_dat)
    expected = SIGMA_DEP**2 * b**2 * np.stack([(channel.loading**2)[:, w].sum(axis=1) for w in windows])
    expected += 2.0 * 1e-3**2

    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 3. NoiseChannel.scale
# ---------------------------------------------------------------------------


def test_channel_scale_is_the_rms_loading():
    loading = np.array([[1.0, 2.0], [3.0, 4.0]])
    channel = NoiseChannel("sigma_dep", loading, 1e-3)
    assert channel.scale == pytest.approx(np.sqrt(np.mean(loading**2)))


def test_channel_scale_makes_mechanisms_comparable():
    """sigma * scale is in soiled-area units for every mechanism.

    The raw sigmas are not comparable: the gravitational loading is dimensionless while the
    wind loadings carry a speed, so their ratio reflects the units as much as the physics.
    Scaling by the design's RMS loading removes that. See notes eq. (88).
    """
    n_mirrors = 2
    tilt = np.full((n_mirrors, T_GRID), 45.0)
    azimuth = np.tile(np.array([0.0, 180.0])[:, None], (1, T_GRID))
    model = _wind_model(["gravitational", "normal_wind"], tilt, azimuth)
    channels = model.noise_channels(F, _sim_in(CONCENTRATION, WIND_SPEED, WIND_DIRECTION))

    scales = {c.name: c.scale for c in channels}
    assert scales["sigma_dep_gamma"] > scales["sigma_dep"], "the wind loading carries a wind speed"
    for channel in channels:
        assert channel.scale > 0.0
        assert np.isfinite(channel.scale)


def test_channel_scale_is_zero_when_the_mechanism_is_unloaded():
    """A mechanism no mirror ever feels has no scale to speak of -- and must not raise."""
    tilt = np.zeros((2, T_GRID))  # flat mirrors: sin(tilt) = 0, so normal wind never loads
    azimuth = np.zeros((2, T_GRID))
    model = _wind_model(["gravitational", "normal_wind"], tilt, azimuth)
    channels = {c.name: c for c in model.noise_channels(F, _sim_in(CONCENTRATION, WIND_SPEED, WIND_DIRECTION))}

    assert channels["sigma_dep_gamma"].scale == 0.0
    assert channels["sigma_dep"].scale > 0.0
