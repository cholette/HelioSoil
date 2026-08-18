"""
Constant-mean soiling model extended with modular, additive wind-driven soiling
mechanisms layered on top of the standard gravitational constant-mean term.

The standard constant-mean model (heliosoil.base_models.ConstantMeanBase) accumulates
soiled area from dust loading and the horizontal projection of the mirror only
(max(0, cos(tilt))) -- the "gravitational" mechanism below. This module adds three more
mechanisms driven by wind hitting the mirror, and lets a model be built from any
subset:

  gravitational:    alpha_j * max(0, cos(theta_ij)) * (mu_tilde + eps_j)

  turbulant wind:     alpha_j * max(0, cos(theta_ij)) * U_j * (omega_turbulent + eps_turb_j)

  normal_wind:      alpha_j * U_j * ( p_windward_ij * (omega_windward + eps_gamma_j)
                                     + p_leeward_ij  * (omega_leeward  + eps_gamma_j) )
                    where p_windward = sin(theta) * max(0, cos(Delta_gamma)),
                          p_leeward  = sin(theta) * max(0, -cos(Delta_gamma))

  tangential_wind:  alpha_j * U_j * sqrt(1 - sin^2(theta_ij)*cos^2(Delta_gamma_ij))
                    * (omega_tangential + eps_tan_j)

  impaction_retention:
                    alpha_j * U_j * ( p_ret_windward_ij * (omega_ret_windward + eps_ret_j)
                                     + p_ret_leeward_ij  * (omega_ret_leeward  + eps_ret_j) )
                    where p_ret_windward = sin(theta)*max(0,cos(theta)) * max(0,  cos(Delta_gamma)),
                          p_ret_leeward  = sin(theta)*max(0,cos(theta)) * max(0, -cos(Delta_gamma))

where Delta_gamma = azimuth (mirror normal) - wind_direction (meteorological, "from"),
and U_j is wind speed. delta_A_ij is the sum of the active mechanisms' contributions.

The settling factor max(0, cos(theta)) is clipped rather than bare cos(theta) because a
mirror tilted past vertical faces the ground and collects no settled dust; an unclipped
cos(theta) < 0 would instead make it *gain* reflectance without bound (real data: yadnarie's
OSW_M1_T180). At tilt > 90 every mechanism here vanishes except tangential_wind, which is
therefore the only channel able to soil a face-down mirror.

The windward and leeward faces of a tilted mirror are given independent fitted
impaction coefficients since there is no a priori reason to expect them to be equal;
only one is nonzero at a time, so a single shared noise term eps_gamma (sigma_dep_gamma)
is used for both. Geometrically, sin^2(theta)*cos^2(Delta_gamma) = (p_windward+p_leeward)^2,
so the normal and tangential variance bases sum to U^2 (an orthogonal decomposition of
wind speed into components normal and tangential to the mirror plane).

impaction_retention reuses normal_wind's windward/leeward split (with its own shared
noise eps_ret, sigma_dep_ret) but additionally weights the normal impaction flux
sin(theta) by the horizontal retention factor max(0, cos(theta)): impacted dust is held
on the surface only in proportion to how horizontal it is. Its orientation
(windward/leeward) contrast therefore vanishes at BOTH tilt=0 and tilt=90 and peaks near
tilt=45, whereas normal_wind's sin(theta) contrast grows monotonically to tilt=90. Real
soiling-orientation contrast peaks at moderate tilt, which normal_wind alone cannot
reproduce -- this mechanism was added to capture it.

A model is configured with an (ordered) subset of {"gravitational", "normal_wind",
"tangential_wind", "impaction_retention"} via the `components` argument; the default
["gravitational", "normal_wind"] is named "constant-mean_gravitational_normal-wind" and
reproduces the original two-mechanism model exactly.

Per-component dust channels
---------------------------
alpha_j above is the dimensionless dust loading -- the measured airborne mass divided
by the same measure's mass in the reference size distribution. By default every
mechanism shares one channel, the simulation's own `dust_type`. Because the mechanisms
are driven by different physics, they need not be driven by the same particle sizes:
`component_dust_types` gives each component its own dust spec, either a single measure
("PM17", "PM10", "PM2.5") or the difference of two ("PM17-PM10"), which isolates the mass
carried by particles between the two cutoffs. For example

    components=["gravitational", "tangential_wind"],
    component_dust_types={"gravitational": "PM17", "tangential_wind": "PM10-PM2.5"}

settles the sub-17 µm mass gravitationally while scouring only the 2.5-10 µm
fraction tangentially. Such a model is named
"constant-mean_PM17xgravitational_PM10-PM2.5xtangential-wind" and reports itself as
"PM17*gravitational + (PM10-PM2.5)*tangential_wind" (see `component_expression`).
Concentrations come from `SimulationInputs.dust_concentration_channels` and reference
masses from `Dust.pm_mass`, so the weather file must carry the PM columns involved.

Fitting
-------
ConstantMeanWindDeposition offers two fits, both returning (estimate, covariance) on the
same scale convention:

  fit_mle: maximizes the likelihood over every parameter at once -- a numerical
    optimization over n_mean + n_sigma parameters followed by a numerical Hessian.

  fit_ls:  exploits the fact that the predicted reflectance is *affine* in the mean
    parameters (delta_A is sum_k theta_k * basis_k, and compute_soiling_factor only
    accumulates it and adds a theta-independent initial condition). The means therefore
    come from a single bounded LINEAR least-squares solve, whose design matrix is read
    off the existing forward model rather than re-derived (see mean_design_matrix). It
    needs no starting point, cannot land in a local optimum, and costs n_mean + 1 forward
    evaluations instead of the hundreds fit_mle's optimizer and Hessian take.

    It fits the MEAN parameters only. The sum of squares does not depend on the sigmas, so
    a least-squares model carries no noise process: fit_ls leaves every sigma set to None,
    predict_soiling_factor then leaves helios.soiling_factor_prediction_variance empty, and
    the model yields a mean prediction with no prediction interval. Its estimate is ordered
    as `mean_parameter_names` rather than `parameter_names`.
"""

import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import heliosoil.base_models as smb
from heliosoil.fitting import ConstantMeanDeposition, CommonFittingMethods, NoiseChannel, kappa_param_name, TRANSFORM_LOG, TRANSFORM_IDENTITY
from heliosoil.utilities import (
    _print_if,
    _check_keys,
    _parse_dust_str,
    _std_errors_from_cov,
    _nominal_reflectance_anchor,
    canonical_dust_spec,
    resolve_dust_concentration,
    cosd,
    sind,
    gravitational_settling_factor,
    cardinal_to_azimuth,
)


def wind_projection_factors(tilt_deg, azimuth_deg, wind_dir_deg):
    """
    Compute the windward and leeward horizontal-impaction geometry factors.

    p_windward = sin(tilt) * max(0, cos(Delta_gamma))
    p_leeward  = sin(tilt) * max(0, -cos(Delta_gamma))

    Delta_gamma = azimuth_deg - wind_dir_deg, so p_windward is nonzero when the
    mirror normal points into the wind (cos(Delta_gamma) > 0) and p_leeward is
    nonzero when the wind hits the back of the mirror.

    Args:
        tilt_deg: array broadcastable against azimuth_deg, mirror tilt [deg].
        azimuth_deg: array broadcastable against tilt_deg, mirror-normal azimuth [deg],
            compass convention (0=N, 90=E, 180=S, 270=W).
        wind_dir_deg: array broadcastable against tilt_deg/azimuth_deg, meteorological
            wind direction (direction the wind is coming FROM) [deg], same convention.

    Returns:
        (p_windward, p_leeward): tuple of arrays, same broadcast shape as the inputs.
    """
    delta_gamma = np.subtract(azimuth_deg, wind_dir_deg)
    cos_dg = cosd(delta_gamma)
    s = sind(tilt_deg)
    p_windward = s * np.maximum(0.0, cos_dg)
    p_leeward = s * np.maximum(0.0, -cos_dg)
    return p_windward, p_leeward


def wind_tangential_factor(tilt_deg, azimuth_deg, wind_dir_deg):
    """
    Fraction of wind speed tangential to (lying in) the mirror plane:

        sqrt(1 - sin^2(tilt) * cos^2(Delta_gamma))

    Delta_gamma = azimuth_deg - wind_dir_deg. This is 1 when the mirror is edge-on
    to the wind (tilt=0, or wind perpendicular to the mirror normal) and 0 when the
    wind blows straight into (or out of) the mirror face at maximum tilt (tilt=90,
    Delta_gamma=0/180). Clipped at 0 to guard against floating-point negatives.

    Args:
        tilt_deg, azimuth_deg, wind_dir_deg: see wind_projection_factors.

    Returns:
        Array, same broadcast shape as the inputs.
    """
    delta_gamma = np.subtract(azimuth_deg, wind_dir_deg)
    normal_component_sq = (sind(tilt_deg) * cosd(delta_gamma)) ** 2
    return np.sqrt(np.clip(1.0 - normal_component_sq, 0.0, None))


def wind_retention_factors(tilt_deg, azimuth_deg, wind_dir_deg):
    """
    Windward/leeward horizontal-impaction geometry factors weighted by the
    gravitational retention of impacted dust on the near-horizontal surface:

        p_ret_windward = sin(tilt) * max(0, cos(tilt)) * max(0,  cos(Delta_gamma))
        p_ret_leeward  = sin(tilt) * max(0, cos(tilt)) * max(0, -cos(Delta_gamma))

    Same windward/leeward split as wind_projection_factors, but the normal
    impaction flux sin(tilt) is additionally scaled by the horizontal projection
    of the surface, cos(tilt) -- impacted dust is retained only in proportion to
    how horizontal the face is. The factor therefore vanishes at BOTH tilt=0 (no
    windward-facing area) and tilt=90 (vertical: nothing is retained) and peaks
    near tilt=45, unlike wind_projection_factors' sin(tilt), which grows
    monotonically to tilt=90. cos(tilt) is clipped at 0 so a face tilted past
    vertical (pointing downward) retains nothing.

    Args:
        tilt_deg, azimuth_deg, wind_dir_deg: see wind_projection_factors.

    Returns:
        (p_ret_windward, p_ret_leeward): tuple of arrays, same broadcast shape
        as the inputs.
    """
    delta_gamma = np.subtract(azimuth_deg, wind_dir_deg)
    cos_dg = cosd(delta_gamma)
    s = sind(tilt_deg) * np.maximum(0.0, cosd(tilt_deg))
    p_windward = s * np.maximum(0.0, cos_dg)
    p_leeward = s * np.maximum(0.0, -cos_dg)
    return p_windward, p_leeward


_TILT_ATOL_DEG = 1e-9  # absorbs float noise on values meant to be exactly 0 or 180


def check_tilt_range(tilt, f=None):
    """
    Raise if any mirror tilt lies outside [0, 180] deg.

    Tilt is measured from horizontal, so the physical range is a half turn. The wind
    mechanisms depend on that: `wind_projection_factors` clips cos(Delta_gamma) but NOT
    sin(tilt), so a tilt outside the range makes the windward/leeward projections -- and
    hence the normal_wind and impaction_retention noise loadings -- negative. Nothing
    downstream would raise: it would quietly subtract deposition from the mean, and once
    the multi-mirror covariance weights products of loadings across mirror pairs it would
    flip the sign of a cross-mirror covariance term. Cheaper to refuse the input here.

    Non-finite entries are ignored: real campaign files carry NaN tilts for mirrors not yet
    in service (728 of them in the first yadnarie campaign), and those columns are trimmed
    or masked downstream rather than being an error.

    Args:
        tilt: array of mirror tilts [deg], any shape.
        f: optional experiment (file) key, used only in the error message.

    Raises:
        ValueError: If any finite tilt is below 0 or above 180 deg.
    """
    tilt = np.asarray(tilt, dtype=float)
    finite = np.isfinite(tilt)
    if not finite.any():
        return
    low, high = float(tilt[finite].min()), float(tilt[finite].max())
    if low < -_TILT_ATOL_DEG or high > 180.0 + _TILT_ATOL_DEG:
        where = "" if f is None else f" for file {f}"
        raise ValueError(
            f"Mirror tilts{where} lie outside [0, 180] deg (found [{low:g}, {high:g}]). Tilt is "
            "measured from horizontal, and the wind mechanisms rely on that range: sin(tilt) is "
            "not clipped, so a tilt outside it makes the windward/leeward projections and the "
            "normal_wind / impaction_retention noise loadings negative, which would corrupt both "
            "the mean deposition and the sign of cross-mirror covariance terms without raising. "
            "Check the 'Tilts' sheet of the campaign file."
        )


def parse_orientation_names(names):
    """
    Parse mirror-normal azimuths [deg] from mirror column names of the form
    "O{cardinal}_..." e.g. "ON_M1_T00" (North), "OE_M2_T85" (East),
    "OS_M2_T30" (South), "OW_M4_T05" (West).

    Args:
        names: iterable of mirror column-name strings.

    Returns:
        np.ndarray of azimuths [deg], same length as `names`.
    """
    azimuths = np.empty(len(names))
    for i, name in enumerate(names):
        token = str(name).split("_")[0].strip().upper()
        if not token.startswith("O"):
            raise ValueError(
                f"Cannot parse orientation from mirror name '{name}': expected a leading 'O' + cardinal-direction token (e.g. 'ON_M1_T00')."
            )
        cardinal = token[1:]
        try:
            azimuths[i] = cardinal_to_azimuth(cardinal)
        except ValueError as e:
            raise ValueError(f"Cannot parse orientation from mirror name '{name}': {e}") from e
    return azimuths


# ---------------------------------------------------------------------------
# Wind-component registry: each component is a pure geometry descriptor (no
# parameter values). A component defines only its MEAN bases; its noise loading and
# variance basis are derived from them (see _WindComponent), so a mechanism added here
# cannot end up with a variance that disagrees with its mean. mean_bases() takes a common
# signature so callers don't need to special-case components that ignore wind.
# ---------------------------------------------------------------------------


def _gravitational_mean_bases(alpha, tilt, azimuth, wind_dir, wind_speed):
    return [alpha[None, :] * gravitational_settling_factor(tilt)]


def _turbulant_wind_mean_bases(alpha, tilt, azimuth, wind_dir, wind_speed):
    return [alpha[None, :] * gravitational_settling_factor(tilt) * wind_speed[None, :]]


def _normal_wind_mean_bases(alpha, tilt, azimuth, wind_dir, wind_speed):
    p_windward, p_leeward = wind_projection_factors(tilt, azimuth, wind_dir)
    u = alpha[None, :] * wind_speed[None, :]
    return [u * p_windward, u * p_leeward]


def _tangential_wind_mean_bases(alpha, tilt, azimuth, wind_dir, wind_speed):
    t = wind_tangential_factor(tilt, azimuth, wind_dir)
    return [alpha[None, :] * wind_speed[None, :] * t]


def _impaction_retention_mean_bases(alpha, tilt, azimuth, wind_dir, wind_speed):
    p_windward, p_leeward = wind_retention_factors(tilt, azimuth, wind_dir)
    u = alpha[None, :] * wind_speed[None, :]
    return [u * p_windward, u * p_leeward]


class _WindComponent:
    """Pure-geometry descriptor for one additive constant-mean-wind soiling mechanism.

    A component supplies only its MEAN bases; the loading of its noise term and its
    variance basis are derived from those, so the three can never disagree.

    mean_bases()/noise_loading()/variance_basis() always take (alpha, tilt, azimuth,
    wind_dir, wind_speed): alpha has shape (N_times,); tilt/azimuth/wind_dir/wind_speed
    broadcast to (N_helios, N_times). Components that ignore wind (gravitational)
    simply don't use the azimuth/wind_dir/wind_speed arguments.
    """

    def __init__(self, key, name_fragment, mean_param_names, sigma_param_name, requires_wind, mean_bases_fn, identifiability_factor):
        self.key = key
        self.name_fragment = name_fragment
        self.mean_param_names = mean_param_names
        self.sigma_param_name = sigma_param_name
        self.requires_wind = requires_wind
        self._mean_bases_fn = mean_bases_fn
        # This mechanism's geometry factor at a given tilt, maximised over wind angle, on
        # a 0-1 scale. Near zero means a mirror at that tilt carries essentially no
        # information about the mechanism, however long the campaign runs. Used to warn
        # when an independent-mirror fit is about to estimate a coefficient it cannot see.
        self.identifiability_factor = identifiability_factor

    def mean_bases(self, alpha, tilt, azimuth, wind_dir, wind_speed):
        """List of arrays (N_helios, N_times), one per entry of mean_param_names."""
        return self._mean_bases_fn(alpha, tilt, azimuth, wind_dir, wind_speed)

    def noise_loading(self, alpha, tilt, azimuth, wind_dir, wind_speed):
        """Array (N_helios, N_times) multiplying this mechanism's single noise term.

        A mechanism carries one noise term regardless of how many mean coefficients it
        has, so the loading is the sum of its mean bases. For the two-face mechanisms
        (normal_wind, impaction_retention) at most one face is nonzero at a time, so that
        sum is just whichever face is active -- see the module docstring.

        This is the quantity the multi-mirror covariance is assembled from: the deposition
        covariance between two mirrors in one interval is weighted by the PRODUCT of their
        loadings, which is why the bases are kept nonnegative.

        Precondition: tilt in [0, 180] deg. wind_projection_factors clips cos(Delta_gamma)
        but not sin(tilt), so a tilt outside that range makes the windward/leeward
        projections -- and hence this loading -- negative, which would flip the sign of a
        cross-mirror covariance term rather than raise. Physical tilts are measured from
        horizontal and never leave the range; nothing here enforces it.
        """
        bases = self.mean_bases(alpha, tilt, azimuth, wind_dir, wind_speed)
        total = bases[0]
        for basis in bases[1:]:
            total = total + basis
        return total

    def variance_basis(self, alpha, tilt, azimuth, wind_dir, wind_speed):
        """Array (N_helios, N_times); this component's variance contribution is
        sigma**2 * variance_basis(...), i.e. the square of its noise loading."""
        return self.noise_loading(alpha, tilt, azimuth, wind_dir, wind_speed) ** 2


# ------------------------------ geometry factors ------------------------------
# Each mechanism's coefficient multiplier at a given tilt, maximised over wind angle and
# normalised to peak at 1. These are module-level named functions rather than lambdas
# **because the model is pickled**: `save` pickles `self`, the model holds its
# `_WindComponent`s, and a lambda attribute makes the whole save fail with
# "Can't pickle <function <lambda>>". Same reason the mean-bases callables above are named.


def _settling_identifiability(tilt_deg):
    """cos+(tilt) -- gravitational settling, and the turbulent term that rides on it."""
    return float(gravitational_settling_factor(tilt_deg))


def _normal_wind_identifiability(tilt_deg):
    """|sin(tilt)| -- normal wind cannot reach a horizontal mirror's face."""
    return float(abs(sind(tilt_deg)))


def _tangential_wind_identifiability(tilt_deg):
    """Tangential wind is carried at every tilt, so nothing is lost to geometry."""
    return 1.0


def _retention_identifiability(tilt_deg):
    """2 sin(tilt) cos+(tilt) -- vanishes at BOTH 0 and 90 deg; peak at 45 deg reads as 1."""
    return float(2.0 * abs(sind(tilt_deg)) * gravitational_settling_factor(tilt_deg))


_GRAVITATIONAL = _WindComponent(
    "gravitational", "gravitational", ("mu_tilde",), "sigma_dep", False, _gravitational_mean_bases, identifiability_factor=_settling_identifiability
)
_TURBULENT = _WindComponent(
    "turbulent_wind", "turbulent-wind", ("omega_turbulent",), "sigma_dep_turb", True, _turbulant_wind_mean_bases, identifiability_factor=_settling_identifiability
)
_NORMAL_WIND = _WindComponent(
    "normal_wind", "normal-wind", ("omega_windward", "omega_leeward"), "sigma_dep_gamma", True, _normal_wind_mean_bases, identifiability_factor=_normal_wind_identifiability
)
_TANGENTIAL_WIND = _WindComponent(
    "tangential_wind", "tangential-wind", ("omega_tangential",), "sigma_dep_tan", True, _tangential_wind_mean_bases, identifiability_factor=_tangential_wind_identifiability
)
_IMPACTION_RETENTION = _WindComponent(
    "impaction_retention",
    "impaction-retention",
    ("omega_ret_windward", "omega_ret_leeward"),
    "sigma_dep_ret",
    True,
    _impaction_retention_mean_bases,
    identifiability_factor=_retention_identifiability,
)


# Order here is the canonical order: it fixes the parameter-vector layout and the
# model_name string regardless of the order `components` is passed in.
_CANONICAL_ORDER = ["gravitational", "turbulent_wind", "normal_wind", "tangential_wind", "impaction_retention"]
# Public view of the same list, so callers (the analysis CLI's component pool, progress-bar
# abbreviations) read the vocabulary from here instead of re-declaring it.
COMPONENT_KEYS = tuple(_CANONICAL_ORDER)
_COMPONENTS = {
    "gravitational": _GRAVITATIONAL,
    "turbulent_wind": _TURBULENT,
    "normal_wind": _NORMAL_WIND,
    "tangential_wind": _TANGENTIAL_WIND,
    "impaction_retention": _IMPACTION_RETENTION,
}


def poorly_identified_components(component_keys, tilt_deg, threshold=0.05):
    """Which mechanisms a mirror at this tilt carries almost no information about.

    A mechanism's geometry factor is what multiplies its coefficient in the forward model,
    so where that factor vanishes the coefficient has nothing to be estimated from: a fit
    restricted to such a mirror returns whatever the optimiser started at, or exactly zero.
    A horizontal mirror does this to every wind mechanism, since normal wind goes as
    sin(tilt) and impaction retention as sin(tilt) cos+(tilt).

    Args:
        component_keys: iterable of component keys, or None for the default pair.
        tilt_deg: the mirror's tilt [deg].
        threshold: geometry factors at or below this count as uninformative.

    Returns:
        list[tuple[str, float]]: (component key, factor) for each starved mechanism, in
        canonical order.
    """
    if tilt_deg is None:
        return []
    starved = []
    for key in resolve_component_keys(component_keys):
        factor = _COMPONENTS[key].identifiability_factor(float(tilt_deg))
        if factor <= threshold:
            starved.append((key, factor))
    return starved


_ALL_MEAN_PARAM_NAMES = (
    "mu_tilde",
    "omega_turbulent",
    "omega_windward",
    "omega_leeward",
    "omega_tangential",
    "omega_ret_windward",
    "omega_ret_leeward",
)
_ALL_SIGMA_PARAM_NAMES = ("sigma_dep", "sigma_dep_turb", "sigma_dep_gamma", "sigma_dep_tan", "sigma_dep_ret")
# Which mean parameters must be positive (log-transformed during fitting); the
# omegas may legitimately be ~0 or negative, so they're left in linear space.
_MEAN_PARAM_LOG = {
    "mu_tilde": True,
    "omega_turbulent": False,
    "omega_windward": False,
    "omega_leeward": False,
    "omega_tangential": False,
    "omega_ret_windward": False,
    "omega_ret_leeward": False,
}


def _resolve_components(components):
    """Validate and canonically order a `components` list of keys, defaulting to
    the original ["gravitational", "normal_wind"] model. Returns a list of
    _WindComponent objects."""
    if components is None:
        components = ["gravitational", "normal_wind"]
    unknown = sorted(set(components) - set(_COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown wind component(s) {unknown}; choose from {_CANONICAL_ORDER}.")
    return [_COMPONENTS[key] for key in _CANONICAL_ORDER if key in components]


def resolve_component_keys(components):
    """The component keys of a `components` list, validated and in canonical order --
    the layout a model built from it will use."""
    return [c.key for c in _resolve_components(components)]


def resolve_component_dust_types(components, component_dust_types=None):
    """Validate a per-component dust assignment into {component key: dust spec or None}
    covering exactly the active components, in canonical order.

    `component_dust_types` may be None (every component uses the simulation's own
    dust_type -- the behaviour of every model built before per-component dust channels
    existed), a single spec applied to every component, or a dict naming a spec for some
    or all of them. Specs are canonicalized (see heliosoil.utilities.parse_dust_spec), so
    an unusable one is rejected here rather than at fit time."""
    keys = resolve_component_keys(components)
    if component_dust_types is None:
        return dict.fromkeys(keys)
    if isinstance(component_dust_types, str):
        component_dust_types = dict.fromkeys(keys, component_dust_types)

    unknown = sorted(set(component_dust_types) - set(keys))
    if unknown:
        raise ValueError(f"Dust types given for component(s) {unknown}, which this model does not have; its components are {keys}.")
    return {key: (canonical_dust_spec(component_dust_types[key]) if component_dust_types.get(key) is not None else None) for key in keys}


def describe_components(components, component_dust_types=None):
    """Human-readable summary of the mechanisms a model sums and the dust channel each
    is driven by, e.g. "PM17*gravitational + (PM10-PM2.5)*tangential_wind". Components
    without an explicit dust type are driven by the simulation's own dust_type and are
    left unqualified ("gravitational")."""
    dust_types = resolve_component_dust_types(components, component_dust_types)
    terms = []
    for key, spec in dust_types.items():
        if spec is None:
            terms.append(key)
        else:
            terms.append(f"({spec})*{key}" if "-" in spec else f"{spec}*{key}")
    return " + ".join(terms)


def parse_model_expression(expression):
    """Inverse of describe_components: read a model expression into the
    (components, component_dust_types) pair a model is built from.

    An expression is a "+"-separated sum of terms, each either a bare component
    ("gravitational" -- driven by the simulation's own dust_type) or SPEC*component
    ("PM17*gravitational", "(PM10-PM2.5)*tangential_wind"). Splitting on "+" is
    unambiguous because a difference spec uses "-". Parentheses around a spec are
    optional and component keys may be spelled with either separator, so
    "PM10-PM2.5*tangential-wind" parses the same as describe_components' own output.

    Specs are canonicalized (parse_dust_spec) and keys validated (_resolve_components),
    so an unusable expression is rejected here rather than at fit time. Returns the keys
    in canonical order alongside a {key: spec or None} dict covering exactly them, i.e.
    parse_model_expression(describe_components(c, d)) round-trips."""
    if expression is None or not str(expression).strip():
        raise ValueError("Empty model expression; expected e.g. 'PM17*gravitational + (PM10-PM2.5)*tangential_wind'.")

    dust_types = {}
    for term in str(expression).split("+"):
        term = term.strip()
        if not term:
            raise ValueError(f"Empty term in model expression {expression!r}.")
        if "*" in term:
            spec_part, key_part = term.split("*", 1)
            spec = canonical_dust_spec(spec_part.strip().strip("()").strip())
        else:
            spec_part, key_part, spec = None, term, None

        key = key_part.strip().lower().replace("-", "_").replace(" ", "_")
        if key not in _COMPONENTS:
            raise ValueError(f"Unknown wind component {key_part.strip()!r} in model expression {expression!r}; choose from {_CANONICAL_ORDER}.")
        if key in dust_types:
            raise ValueError(f"Component {key!r} appears more than once in model expression {expression!r}; each mechanism is fitted at most once.")
        dust_types[key] = spec

    keys = resolve_component_keys(dust_types)
    return keys, {key: dust_types[key] for key in keys}


def _dust_alpha(simulation_inputs, f, dust_spec=None):
    """Dimensionless dust loading alpha_j for file `f`: the measured airborne mass in a
    channel divided by that channel's mass in the reference size distribution.

    dust_spec=None uses the simulation's own dust_type/dust_concentration, exactly as
    every other model in the package does. A spec ("PM10", "PM17-PM10", ...) instead
    reads that channel from simulation_inputs.dust_concentration_channels and divides by
    the matching Dust.pm_mass."""
    sim_in = simulation_inputs
    if dust_spec is None:
        try:
            attr = _parse_dust_str(sim_in.dust_type[f])
            den = getattr(sim_in.dust, attr)
        except Exception:
            raise ValueError(
                "Dust measurement = "
                + sim_in.dust_type[f]
                + " not present in dust class. Use dust_type="
                + sim_in.dust_type[f]
                + " option when initializing the model"
            )
        return sim_in.dust_concentration[f] / den[f]

    channels = getattr(sim_in, "dust_concentration_channels", {}).get(f)
    concentration, _ = resolve_dust_concentration(channels, dust_spec)
    if concentration is None:
        raise ValueError(
            f"Dust channel {dust_spec!r} is not available for file {f}: the weather file must contain the particulate-matter "
            f"column(s) it is built from (this file has {sorted(channels or {})})."
        )
    return concentration / sim_in.dust.pm_mass(f, dust_spec)


class ConstantMeanWindBase(smb.ConstantMeanBase):
    """
    Constant-mean deposition model with a modular set of additive wind-driven
    soiling mechanisms. See module docstring for the governing equations.

    Args:
        components: ordered subset of {"gravitational", "normal_wind",
            "tangential_wind", "impaction_retention"} to include (any order
            accepted; canonicalized internally). Defaults to ["gravitational",
            "normal_wind"], reproducing the original two-mechanism model.
        component_dust_types: optional dust channel per component, either a dict
            {component key: dust spec} or one spec for all of them. A spec is a single
            measure ("PM17", "PM10", "PM2.5") or a difference ("PM17-PM10"). Components
            left out use the simulation's own dust_type; the default (None) means every
            component does, reproducing the single-channel model exactly.
    """

    def __init__(self, components=None, component_dust_types=None):
        # Explicit (non-cooperative) call: a bare super().__init__() would resolve
        # via the instance's MRO and land on ConstantMeanDeposition.__init__ (which
        # requires file_params) when this runs as part of ConstantMeanWindDeposition.
        smb.ConstantMeanBase.__init__(self)
        self.components = _resolve_components(components)
        self._component_keys = [c.key for c in self.components]
        self._needs_wind = any(c.requires_wind for c in self.components)
        self.component_dust_types = resolve_component_dust_types(self._component_keys, component_dust_types)

        # Every possible parameter attribute exists (None if inactive) so that
        # introspection/pickling never hits AttributeError regardless of which
        # components are active.
        for name in _ALL_MEAN_PARAM_NAMES:
            if not hasattr(self, name):
                setattr(self, name, None)
        for name in _ALL_SIGMA_PARAM_NAMES:
            if not hasattr(self, name):
                setattr(self, name, None)

        self._mean_param_names = [n for c in self.components for n in c.mean_param_names]
        self._sigma_param_names = [c.sigma_param_name for c in self.components]
        self._param_names = self._mean_param_names + self._sigma_param_names
        # mu_tilde is a deposition rate and must stay positive; the wind coefficients are
        # carried linearly, since a negative one is meaningful (scouring).
        self._mean_transform_codes = tuple(TRANSFORM_LOG if _MEAN_PARAM_LOG[n] else TRANSFORM_IDENTITY for n in self._mean_param_names)

        # One common-variance fraction per mechanism, for variance_model="per_mechanism".
        # Left unset until set_variance_model seeds them; the other variance models never
        # read these. Not part of _param_names -- kappa is model state, not a fitted
        # parameter, until the parameter vector is extended.
        for name in self._sigma_param_names:
            if not hasattr(self, kappa_param_name(name)):
                setattr(self, kappa_param_name(name), None)

    @property
    def model_name(self):
        # A component driven by its own dust channel is tagged with it, e.g.
        # "constant-mean_PM17xgravitational_PM10-PM2.5xtangential-wind". "x" rather than
        # "*" keeps the name usable as a results-folder name on every platform.
        fragments = []
        for c in self.components:
            spec = self.component_dust_types[c.key]
            fragments.append(c.name_fragment if spec is None else f"{spec}x{c.name_fragment}")
        return "constant-mean_" + "_".join(fragments)

    @property
    def component_expression(self):
        """The mechanisms this model sums and the dust channel driving each, e.g.
        "PM17*gravitational + (PM10-PM2.5)*tangential_wind" (see describe_components)."""
        return describe_components(self._component_keys, self.component_dust_types)

    @property
    def parameter_names(self):
        """Every fitted parameter: means, then magnitudes, then the common fractions the
        current variance model carries.

        Agrees with ``CommonFittingMethods.parameter_names`` -- ``_param_names`` is exactly
        means + magnitudes -- but is restated here so this base class, which has no fitting
        mixin, can still report its own layout. ``_param_names`` deliberately stops at the
        magnitudes: it is what the forward model accepts as overrides, and a common fraction
        changes the likelihood rather than the prediction.
        """
        return list(self._param_names) + list(getattr(self, "fitted_kappa_names", ()))

    # mean_parameter_names (the least-squares subset) comes from CommonFittingMethods, off
    # the _mean_param_names/_sigma_param_names this class sets per active component.

    def _component_alphas(self, simulation_inputs, f):
        """{component key: alpha_j} for file `f`. Components sharing a dust channel --
        by default all of them, on the simulation's own dust_type -- share one array."""
        alphas, by_spec = {}, {}
        for key in self._component_keys:
            spec = self.component_dust_types[key]
            if spec not in by_spec:
                by_spec[spec] = _dust_alpha(simulation_inputs, f, spec)
            alphas[key] = by_spec[spec]
        return alphas

    def import_site_data_and_constants(self, file_params, verbose=True):
        super().import_site_data_and_constants(file_params, verbose=verbose)
        table = pd.read_excel(file_params, index_col="Parameter")
        for name in _ALL_MEAN_PARAM_NAMES[1:] + _ALL_SIGMA_PARAM_NAMES[1:]:
            try:
                setattr(self, name, float(table.loc[name].Value))
            except Exception:
                setattr(self, name, None)
                _print_if(f"No {name} model defined in {file_params}.", self.verbose)

    def _geometry_and_wind(self, f, simulation_inputs):
        """
        The geometry and weather every mechanism is evaluated on, validated once.

        Single funnel for this model's inputs, so the tilt-range check and the
        missing-data errors cover both the mean assembly and the noise channels.

        Returns:
            tuple: (tilt, azimuth, wind_dir, wind_speed); the last three are None when no
            active mechanism needs wind.

        Raises:
            ValueError: If tilts leave [0, 180] deg, or a wind-driven mechanism is active
                without azimuth or wind data.
        """
        helios = self.helios
        tilt = helios.tilt[f]
        check_tilt_range(tilt, f)

        if not self._needs_wind:
            return tilt, None, None, None

        if getattr(helios, "azimuth", None) is None or helios.azimuth.get(f) is None:
            raise ValueError(
                "helios.azimuth is not populated for file "
                + str(f)
                + ". Call helios_angles(..., orientations=...) before "
                + "calculate_delta_soiled_area on a wind-impaction model."
            )
        wind_dir = getattr(simulation_inputs, "wind_direction", {}).get(f)
        wind_speed = getattr(simulation_inputs, "wind_speed", {}).get(f)
        if wind_dir is None or wind_speed is None:
            raise ValueError(
                "Wind speed and/or wind direction are not available for file "
                + str(f)
                + ". The weather file must contain wind-speed and wind-direction "
                + "columns (e.g. 'WindSpeed'/'WD') for the wind-impaction model."
            )
        return tilt, helios.azimuth[f], wind_dir, wind_speed

    def noise_channels(self, f, simulation_inputs, sigma_override=None):
        """
        One deposition-noise channel per active mechanism.

        Each mechanism carries its own noise process, whose loading is the sum of that
        mechanism's mean bases (see _WindComponent.noise_loading). Overrides the
        single-channel implementation in CommonFittingMethods, which is what lets the
        variance assembly there serve both model families unchanged.

        Args:
            f: Experiment (file) key.
            simulation_inputs (SimulationInputs): Simulation inputs.
            sigma_override: Ignored. There is no single magnitude to replace when several
                mechanisms are active; every sigma is read from the model, which
                update_model_parameters keeps in sync with the parameter vector.

        Returns:
            list[NoiseChannel]: One per active mechanism, in canonical order.

        Raises:
            ValueError: If a wind-driven mechanism is active but azimuth or wind data are
                missing for this experiment.
        """
        tilt, azimuth, wind_dir, wind_speed = self._geometry_and_wind(f, simulation_inputs)
        alphas = self._component_alphas(simulation_inputs, f)
        return [
            NoiseChannel(
                component.sigma_param_name,
                component.noise_loading(alphas[component.key], tilt, azimuth, wind_dir, wind_speed),
                getattr(self, component.sigma_param_name),
                self._kappa_for(component.sigma_param_name),
            )
            for component in self.components
        ]

    def calculate_delta_soiled_area(self, simulation_inputs, verbose=True, **param_overrides):
        # _print_if("Calculating soil deposited in a timestep [m^2/m^2]", verbose)

        unknown = sorted(set(param_overrides) - set(self._param_names))
        if unknown:
            raise ValueError(
                f"Unknown parameter override(s) {unknown} for model '{self.model_name}'; this model's parameters are {self._param_names}."
            )

        resolved = {}
        for name in self._param_names:
            value = param_overrides.get(name, None)
            if value is None:
                value = getattr(self, name)
            else:
                _print_if(f"Using supplied value for {name} = {value}", verbose)
            resolved[name] = value

        sim_in = simulation_inputs
        helios = self.helios

        files = list(sim_in.time.keys())
        for f in files:
            tilt, azimuth, wind_dir, wind_speed = self._geometry_and_wind(f, sim_in)
            alphas = self._component_alphas(sim_in, f)

            total_mean = np.zeros_like(tilt, dtype=float)
            for component in self.components:
                bases = component.mean_bases(alphas[component.key], tilt, azimuth, wind_dir, wind_speed)
                for base, name in zip(bases, component.mean_param_names):
                    total_mean = total_mean + base * resolved[name]
            helios.delta_soiled_area[f] = total_mean

            sigmas = {name: resolved[name] for name in self._sigma_param_names}
            if any(s is not None for s in sigmas.values()):
                total_var = np.zeros_like(tilt, dtype=float)
                for component in self.components:
                    sigma = sigmas[component.sigma_param_name]
                    if sigma is None:
                        continue
                    total_var = total_var + sigma**2 * component.variance_basis(alphas[component.key], tilt, azimuth, wind_dir, wind_speed)
                helios.delta_soiled_area_variance[f] = total_var

        self.helios = helios

    # random_delta_soiled_area lives on heliosoil.fitting.CommonFittingMethods now: the
    # draw is per mechanism, with each one's noise split into a site-common and a
    # mirror-specific part, so it needs the channel list this class does not have.


class ConstantMeanWindDeposition(ConstantMeanWindBase, ConstantMeanDeposition):
    """
    Fitting counterpart of ConstantMeanWindBase, following the same MLE-fitting
    machinery as heliosoil.fitting.ConstantMeanDeposition but with a parameter
    vector sized/ordered by the active `components` (see ConstantMeanWindBase).
    """

    def __init__(self, file_params, components=None, component_dust_types=None, verbose=True):
        ConstantMeanWindBase.__init__(self, components=components, component_dust_types=component_dust_types)
        self.verbose = verbose
        self.import_site_data_and_constants(file_params, verbose=self.verbose)
        table = pd.read_excel(file_params, index_col="Parameter")
        self.helios.nominal_reflectance = float(table.loc["nominal_reflectance"].Value)

    def helios_angles(self, simulation_inputs, reflectance_data, verbose=True, second_surface=True, orientations=None):
        """
        Sets helios.tilt (as in ConstantMeanDeposition.helios_angles) and, only if
        an active component needs wind data, additionally populates
        helios.azimuth[f], shape (N_helios, N_times), from mirror-name parsing or
        an explicit override.

        Args:
            orientations: optional override for the mirror-normal azimuths [deg].
                - None (default): parsed from reflectance_data.mirror_names[f] via
                  parse_orientation_names (expects names like 'ON_M1_T00').
                - dict[file -> array-like of length N_helios]: explicit azimuths per file.
                - array-like of length N_helios: same azimuths applied to every file.
        """
        ConstantMeanDeposition.helios_angles(self, simulation_inputs, reflectance_data, verbose=verbose, second_surface=second_surface)

        if not self._needs_wind:
            return

        files = list(simulation_inputs.time.keys())
        helios = self.helios
        helios.azimuth = {f: None for f in files}
        for f in files:
            N_helios, N_times = helios.tilt[f].shape
            if orientations is None:
                az = parse_orientation_names(reflectance_data.mirror_names[f])
            elif isinstance(orientations, dict):
                az = np.asarray(orientations[f], dtype=float)
            else:
                az = np.asarray(orientations, dtype=float)

            if az.shape[0] != N_helios:
                raise ValueError(f"Number of orientations ({az.shape[0]}) does not match the number of heliostats ({N_helios}) for file {f}.")

            helios.azimuth[f] = np.tile(az[:, None], (1, N_times))

        self.helios = helios

    def predict_soiling_factor(self, simulation_inputs, reflectance_data=None, verbose=True, **param_overrides):
        sim_in = simulation_inputs
        self.calculate_delta_soiled_area(sim_in, verbose=verbose, **param_overrides)
        self.compute_soiling_factor(reflectance_data=reflectance_data)

        if any(getattr(self, name) is not None for name in self._sigma_param_names):
            for f in self.helios.soiling_factor.keys():
                inc_factor = self.helios.inc_ref_factor[f]
                dsav = self.helios.delta_soiled_area_variance[f]
                self.helios.soiling_factor_prediction_variance[f] = inc_factor**2 * np.cumsum(dsav, axis=1)
        else:
            self.helios.soiling_factor_prediction_variance = {}

    def update_model_parameters(self, x):
        if isinstance(x, (list, np.ndarray)):
            core, kappas = self._split_parameters(x)
            for name, value in zip(self._param_names, core):
                setattr(self, name, value)
            self._update_kappa_parameters(kappas)
        else:
            # Scalar: only used by the inherited fit_least_squares warm start, which
            # optimizes over the first mean parameter alone (mu_tilde, whenever the
            # gravitational component is active -- it is always first in canonical order).
            setattr(self, self._mean_param_names[0], x)

    # _negative_log_likelihood is inherited. It used to be overridden here only to avoid
    # extracting a single sigma positionally from `params`; the base version no longer does
    # that either, so the two bodies had become identical.

    def fit_map(self, *args, **kwargs):
        raise NotImplementedError(
            "fit_map is not available: the base CommonFittingMethods class does not "
            "define a fit_map method to inherit from (see heliosoil.fitting). Use "
            "fit_mle instead."
        )

    def _initial_parameter_guess(self, simulation_inputs, reflectance_data, verbose=True):
        """
        Warm start for ``fit_mle``: the exact least-squares means, then a magnitude.

        The means come from ``fit_ls`` -- one bounded LINEAR solve, no starting point, no
        local optima, and it fits every mean whatever the active components are. The
        magnitudes then come from a bounded scalar search over the likelihood with every
        mechanism sharing one value, and each common fraction starts at an even split,
        which is interior to [0, 1] so the search does not begin on a boundary.

        This used to call ``fit_least_squares`` instead -- the SCALAR bounded search, whose
        bracket ``_LS_SCALAR_BOUNDS = (1.000001, 1000)`` exists because ``hrz0`` must exceed
        one. Applied to ``mu_tilde``, about 2e-5 on real data, it returned the lower bound
        every time, so every fit started from ``mu_tilde = 1`` with an objective three
        orders of magnitude off its optimum. The optimiser climbed out of it while there
        were five parameters; adding a common variance fraction it no longer could, and the
        fit terminated on the boundary with inflated magnitudes and a likelihood worse than
        the null's. The sigma bracket had the same origin: the old upper bound was the sum
        of squares itself, 1.8e9 at a pinned mean, where a magnitude is on the scale of a
        reflectance increment.

        Args:
            simulation_inputs (SimulationInputs): Simulation inputs.
            reflectance_data (ReflectanceMeasurements): Measurement data.
            verbose (bool): Whether to print progress messages.

        Returns:
            numpy.ndarray: A starting vector in ``parameter_names`` order.
        """
        _print_if("Getting initial mean-parameter guess by linear least squares ...", verbose)
        n_sigma = len(self._sigma_param_names)
        kappa0 = [0.5] * len(self.fitted_kappa_names)

        # Choosing a starting point must not change the model. Two things here would
        # otherwise: fit_ls clears every magnitude (a least-squares fit estimates none), and
        # the scalar search below leaves the last value it probed on the model, since
        # evaluating the likelihood goes through update_model_parameters. Both are restored
        # before returning.
        saved = {name: getattr(self, name, None) for name in self._sigma_param_names}
        means, _ = self.fit_ls(simulation_inputs, reflectance_data, verbose=False, transform_to_original_scale=True)
        means = [float(v) for v in np.asarray(means, dtype=float).reshape(-1)]

        # A magnitude has the units of a reflectance increment, so the root of the sum of
        # squares is the right scale for the bracket -- not the sum itself.
        sse = float(self._sse(means, simulation_inputs, reflectance_data))
        upper = max(np.sqrt(max(sse, 0.0)), 10.0 * smb.tol)

        def nloglike1d(s):
            return self._negative_log_likelihood(means + [s] * n_sigma + kappa0, simulation_inputs, reflectance_data)

        s0 = minimize_scalar(nloglike1d, bounds=(smb.tol, upper), method="Bounded")
        for name, value in saved.items():
            setattr(self, name, value)
        return np.array(means + [s0.x] * n_sigma + kappa0)

    def fit_mle(self, simulation_inputs, reflectance_data, verbose=True, x0=None, transform_to_original_scale=False, save_file=None, **optim_kwargs):
        _check_keys(simulation_inputs, reflectance_data)

        if x0 is None:
            x0 = self._initial_parameter_guess(simulation_inputs, reflectance_data, verbose=verbose)
            _print_if("x0 = " + str(x0), verbose)

        _print_if("Getting MLE estimates ... ", verbose)
        y, y_cov = CommonFittingMethods.fit_mle(
            self, simulation_inputs, reflectance_data, verbose=False, x0=x0, transform_to_original_scale=False, **optim_kwargs
        )

        _print_if("========== MLE Estimates ======== ", verbose)
        if transform_to_original_scale:
            x_hat = self.transform_scale(y)
            # Transform the covariance directly rather than inverting y_cov back to a Hessian,
            # transforming that, and inverting again. The two are algebraically identical, but
            # the round trip inverts a matrix that is near-singular whenever a variance
            # parameter fits to its bound -- a routine outcome, since sigma >= 0 makes the
            # constraint active whenever the observed scatter is no larger than the
            # measurement noise already predicts -- and raises LinAlgError there.
            # d(natural)/d(fitted) is exp(y) = x_hat at the log-transformed entries and 1 at
            # the linear ones, so with J diagonal the delta method is cov -> J cov J.
            jacobian = self.natural_scale_jacobian(y)
            x_hat_cov = (jacobian[:, None] * y_cov) * jacobian[None, :]
        else:
            x_hat = y
            x_hat_cov = y_cov

        s = _std_errors_from_cov(x_hat_cov)
        x_ci = x_hat + 1.96 * s * np.array([[-1], [1]])
        for i, name in enumerate(self.parameter_names):
            label = name if transform_to_original_scale else self._fitted_scale_label(name, i)
            _print_if(f"{label} = {x_hat[i]:.3e}", verbose)
            if np.isfinite(s[i]):
                _print_if(f"95% confidence interval for {label}: [{x_ci[0, i]:.3e}, {x_ci[1, i]:.3e}]", verbose)
            else:
                # A non-finite standard error means the curvature at the optimum is not
                # usable -- what a variance parameter fitted to its bound produces. Saying
                # so beats printing a NaN interval, which reads as a number.
                _print_if(f"95% confidence interval for {label}: unavailable (estimate at or near a bound; the Wald interval is not defined there)", verbose)

        return x_hat, x_hat_cov

    # ------------------------------------------------------------------
    # Least-squares fitting: mean_design_matrix and fit_ls are inherited from
    # heliosoil.fitting.AffineMeanLeastSquares (this model is affine in its mean
    # parameters); what is model-specific is only how those parameters are transformed
    # and which of them must stay positive.
    # ------------------------------------------------------------------

    def save(self, file_name, log_p_hat=None, log_p_hat_cov=None, training_simulation_data=None, training_reflectance_data=None):
        with open(file_name, "wb") as f:
            save_data = {"model": self, "type": self.model_name}
            if log_p_hat is not None:
                save_data["transformed_parameters"] = log_p_hat
            if log_p_hat_cov is not None:
                save_data["transformed_parameter_covariance"] = log_p_hat_cov
            if training_simulation_data is not None:
                save_data["simulation_data"] = training_simulation_data
            if training_reflectance_data is not None:
                save_data["reflectance_data"] = training_reflectance_data

            pickle.dump(save_data, f)
