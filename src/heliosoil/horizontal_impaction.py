"""
Constant-mean soiling model extended with modular, additive wind-driven soiling
mechanisms layered on top of the standard gravitational constant-mean term.

The standard constant-mean model (heliosoil.base_models.ConstantMeanBase) accumulates
soiled area from dust loading and the vertical projection of the mirror only
(cos(tilt)) -- the "gravitational" mechanism below. This module adds three more
mechanisms driven by wind hitting the mirror, and lets a model be built from any
subset:

  gravitational:    alpha_j * cos(theta_ij) * (mu_tilde + eps_j)

  turbulant wind:     alpha_j * U_j * (omega_turbulent + eps_turb_j)

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

settles the whole airborne mass gravitationally while scouring only the 2.5-10 µm
fraction tangentially. Such a model is named
"constant-mean_PMM17xgravitational_PM10-PM2.5xtangential-wind" and reports itself as
"PM17*gravitational + (PM10-PM2.5)*tangential_wind" (see `component_expression`).
Concentrations come from `SimulationInputs.dust_concentration_channels` and reference
masses from `Dust.pm_mass`, so the weather file must carry the PM columns involved.
"""

import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import heliosoil.base_models as smb
from heliosoil.fitting import ConstantMeanDeposition, CommonFittingMethods
from heliosoil.utilities import (
    _print_if,
    _check_keys,
    _parse_dust_str,
    _std_errors_from_cov,
    _nominal_reflectance_series,
    _nominal_reflectance_anchor,
    canonical_dust_spec,
    resolve_dust_concentration,
    cosd,
    sind,
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
# parameter values). mean_bases()/variance_basis() take a common signature so
# callers don't need to special-case components that ignore wind.
# ---------------------------------------------------------------------------


def _gravitational_mean_bases(alpha, tilt, azimuth, wind_dir, wind_speed):
    return [alpha[None, :] * cosd(tilt)]


def _gravitational_variance_basis(alpha, tilt, azimuth, wind_dir, wind_speed):
    return (alpha[None, :] * cosd(tilt)) ** 2


def _turbulant_wind_mean_bases(alpha, tilt, azimuth, wind_dir, wind_speed):
    return [alpha[None, :] * cosd(tilt) * wind_speed[None, :]]


def _turbulant_wind_variance_basis(alpha, tilt, azimuth, wind_dir, wind_speed):
    return (alpha[None, :] * cosd(tilt) * wind_speed[None, :]) ** 2


def _normal_wind_mean_bases(alpha, tilt, azimuth, wind_dir, wind_speed):
    p_windward, p_leeward = wind_projection_factors(tilt, azimuth, wind_dir)
    u = alpha[None, :] * wind_speed[None, :]
    return [u * p_windward, u * p_leeward]


def _normal_wind_variance_basis(alpha, tilt, azimuth, wind_dir, wind_speed):
    p_windward, p_leeward = wind_projection_factors(tilt, azimuth, wind_dir)
    return (alpha[None, :] * wind_speed[None, :]) ** 2 * (p_windward + p_leeward) ** 2


def _tangential_wind_mean_bases(alpha, tilt, azimuth, wind_dir, wind_speed):
    t = wind_tangential_factor(tilt, azimuth, wind_dir)
    return [alpha[None, :] * wind_speed[None, :] * t]


def _tangential_wind_variance_basis(alpha, tilt, azimuth, wind_dir, wind_speed):
    t = wind_tangential_factor(tilt, azimuth, wind_dir)
    return (alpha[None, :] * wind_speed[None, :] * t) ** 2


def _impaction_retention_mean_bases(alpha, tilt, azimuth, wind_dir, wind_speed):
    p_windward, p_leeward = wind_retention_factors(tilt, azimuth, wind_dir)
    u = alpha[None, :] * wind_speed[None, :]
    return [u * p_windward, u * p_leeward]


def _impaction_retention_variance_basis(alpha, tilt, azimuth, wind_dir, wind_speed):
    p_windward, p_leeward = wind_retention_factors(tilt, azimuth, wind_dir)
    return (alpha[None, :] * wind_speed[None, :]) ** 2 * (p_windward + p_leeward) ** 2


class _WindComponent:
    """Pure-geometry descriptor for one additive constant-mean-wind soiling mechanism.

    mean_bases()/variance_basis() always take (alpha, tilt, azimuth, wind_dir,
    wind_speed): alpha has shape (N_times,); tilt/azimuth/wind_dir/wind_speed
    broadcast to (N_helios, N_times). Components that ignore wind (gravitational)
    simply don't use the azimuth/wind_dir/wind_speed arguments.
    """

    def __init__(self, key, name_fragment, mean_param_names, sigma_param_name, requires_wind, mean_bases_fn, variance_basis_fn):
        self.key = key
        self.name_fragment = name_fragment
        self.mean_param_names = mean_param_names
        self.sigma_param_name = sigma_param_name
        self.requires_wind = requires_wind
        self._mean_bases_fn = mean_bases_fn
        self._variance_basis_fn = variance_basis_fn

    def mean_bases(self, alpha, tilt, azimuth, wind_dir, wind_speed):
        """List of arrays (N_helios, N_times), one per entry of mean_param_names."""
        return self._mean_bases_fn(alpha, tilt, azimuth, wind_dir, wind_speed)

    def variance_basis(self, alpha, tilt, azimuth, wind_dir, wind_speed):
        """Array (N_helios, N_times); this component's variance contribution is
        sigma**2 * variance_basis(...)."""
        return self._variance_basis_fn(alpha, tilt, azimuth, wind_dir, wind_speed)


_GRAVITATIONAL = _WindComponent(
    "gravitational", "gravitational", ("mu_tilde",), "sigma_dep", False, _gravitational_mean_bases, _gravitational_variance_basis
)
_TURBULENT = _WindComponent(
    "turbulent_wind", "turbulent-wind", ("omega_turbulent",), "sigma_dep_turb", True, _turbulant_wind_mean_bases, _turbulant_wind_variance_basis
)
_NORMAL_WIND = _WindComponent(
    "normal_wind", "normal-wind", ("omega_windward", "omega_leeward"), "sigma_dep_gamma", True, _normal_wind_mean_bases, _normal_wind_variance_basis
)
_TANGENTIAL_WIND = _WindComponent(
    "tangential_wind", "tangential-wind", ("omega_tangential",), "sigma_dep_tan", True, _tangential_wind_mean_bases, _tangential_wind_variance_basis
)
_IMPACTION_RETENTION = _WindComponent(
    "impaction_retention",
    "impaction-retention",
    ("omega_ret_windward", "omega_ret_leeward"),
    "sigma_dep_ret",
    True,
    _impaction_retention_mean_bases,
    _impaction_retention_variance_basis,
)

# Order here is the canonical order: it fixes the parameter-vector layout and the
# model_name string regardless of the order `components` is passed in.
_CANONICAL_ORDER = ["gravitational", "turbulent_wind", "normal_wind", "tangential_wind", "impaction_retention"]
_COMPONENTS = {
    "gravitational": _GRAVITATIONAL,
    "turbulent_wind": _TURBULENT,
    "normal_wind": _NORMAL_WIND,
    "tangential_wind": _TANGENTIAL_WIND,
    "impaction_retention": _IMPACTION_RETENTION,
}
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
        self._log_transform = np.array([_MEAN_PARAM_LOG[n] for n in self._mean_param_names] + [True] * len(self._sigma_param_names))

    @property
    def model_name(self):
        # A component driven by its own dust channel is tagged with it, e.g.
        # "constant-mean_PMM17xgravitational_PM10-PM2.5xtangential-wind". "x" rather than
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
        return list(self._param_names)

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
            tilt = helios.tilt[f]

            azimuth = wind_dir = wind_speed = None
            if self._needs_wind:
                if getattr(helios, "azimuth", None) is None or helios.azimuth.get(f) is None:
                    raise ValueError(
                        "helios.azimuth is not populated for file "
                        + str(f)
                        + ". Call helios_angles(..., orientations=...) before "
                        + "calculate_delta_soiled_area on a wind-impaction model."
                    )
                azimuth = helios.azimuth[f]

                wind_dir = getattr(sim_in, "wind_direction", {}).get(f)
                wind_speed = getattr(sim_in, "wind_speed", {}).get(f)
                if wind_dir is None or wind_speed is None:
                    raise ValueError(
                        "Wind speed and/or wind direction are not available for file "
                        + str(f)
                        + ". The weather file must contain wind-speed and wind-direction "
                        + "columns (e.g. 'WindSpeed'/'WD') for the wind-impaction model."
                    )

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

    def random_delta_soiled_area(self, simulation_inputs, verbose=True, **param_overrides):
        """
        Simulates the delta soiled area with randomness from all active mechanisms'
        variance. The airborne dust loading is treated as a constant.

        Overrides ConstantMeanBase.random_delta_soiled_area, which calls
        calculate_delta_soiled_area positionally as (sim_in, mu_tilde, sigma_dep,
        verbose) -- incompatible with this class's **param_overrides signature.
        """
        self.calculate_delta_soiled_area(simulation_inputs, verbose=verbose, **param_overrides)
        mean_area_loss = self.helios.delta_soiled_area
        var_area_loss = self.helios.delta_soiled_area_variance
        files = list(mean_area_loss.keys())
        sim = {f: [] for f in files}
        for f in files:
            mu = mean_area_loss[f]
            sigma = np.sqrt(var_area_loss[f])
            sim[f] = mu + sigma * np.random.standard_normal(size=mu.shape)

        return sim


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
            for name, value in zip(self._param_names, x):
                setattr(self, name, value)
        else:
            # Scalar: only used by the inherited fit_least_squares warm start, which
            # optimizes over the first mean parameter alone (mu_tilde, whenever the
            # gravitational component is active -- it is always first in canonical order).
            setattr(self, self._mean_param_names[0], x)

    def _negative_log_likelihood(self, params, simulation_inputs, reflectance_data):
        # Structurally identical to CommonFittingMethods._negative_log_likelihood,
        # except sigma is not extracted positionally from `params` -- every active
        # component's sigma is read from self by _compute_variance_of_measurements,
        # which update_model_parameters keeps in sync immediately below.
        _check_keys(simulation_inputs, reflectance_data)

        sim_in = simulation_inputs
        files = list(reflectance_data.times.keys())
        pi = reflectance_data.prediction_indices
        meas = reflectance_data.average
        NL = [reflectance_data.average[f].shape[0] for f in files]

        loglike = -0.5 * np.sum(NL) * np.log(2 * np.pi)
        self.update_model_parameters(params)
        self.predict_soiling_factor(simulation_inputs, reflectance_data=reflectance_data, verbose=False)
        sf = self.helios.soiling_factor

        s2total = self._compute_variance_of_measurements(None, sim_in, reflectance_data=reflectance_data)

        for f in files:
            delta_r = np.diff(meas[f], axis=0)
            r0 = _nominal_reflectance_series(reflectance_data, f, self.helios.nominal_reflectance)
            rho_prediction = r0 * sf[f][:, pi[f]].transpose()
            mu_delta_r = np.diff(rho_prediction, axis=0)
            loglike += np.sum(-0.5 * np.log(s2total[f]) - (delta_r - mu_delta_r) ** 2 / (2 * s2total[f]))

        return -loglike

    def _compute_variance_of_measurements(self, sigma_dep, simulation_inputs, reflectance_data=None):
        # `sigma_dep` is kept as the first positional argument for signature
        # compatibility with CommonFittingMethods' calling convention, but ignored:
        # every active component's sigma is read from self (kept in sync by
        # update_model_parameters immediately before this is called).
        _check_keys(simulation_inputs, reflectance_data)

        sim_in = simulation_inputs
        files = list(sim_in.time.keys())
        sigmas = {name: getattr(self, name) for name in self._sigma_param_names}

        s2total = dict.fromkeys(files)
        for f in files:
            if reflectance_data is None:
                pif = range(0, len(sim_in.time[f]))
            else:
                pif = reflectance_data.prediction_indices[f]

            nom_ref_anchor = _nominal_reflectance_anchor(reflectance_data, f, self.helios.nominal_reflectance)
            b = nom_ref_anchor * self.helios.inc_ref_factor[f]

            alphas = self._component_alphas(sim_in, f)

            if reflectance_data is None:
                meas_sig = np.zeros(self.helios.tilt[f].shape).transpose()
            else:
                meas_sig = reflectance_data.sigma_of_the_mean[f]

            tilt = self.helios.tilt[f]
            # calculate_delta_soiled_area (called via predict_soiling_factor just
            # before this) already validated azimuth/wind data when any component
            # needs wind, so it's safe to index directly here.
            azimuth = self.helios.azimuth[f] if self._needs_wind else None
            wind_dir = sim_in.wind_direction[f] if self._needs_wind else None
            wind_speed = sim_in.wind_speed[f] if self._needs_wind else None

            ind1 = pif[0:-1]
            ind2 = [x - 1 for x in pif[1::]]

            total_delta = np.zeros((len(ind1), tilt.shape[0]))
            for component in self.components:
                sigma = sigmas[component.sigma_param_name]
                if sigma is None:
                    continue
                basis = component.variance_basis(alphas[component.key], tilt, azimuth, wind_dir, wind_speed)
                c2t = np.cumsum(basis, axis=1).transpose()
                total_delta = total_delta + sigma**2 * (c2t[ind2, :] - c2t[ind1, :])

            s2total[f] = b**2 * total_delta + meas_sig[0:-1, :] ** 2 + meas_sig[1::, :] ** 2

        return s2total

    def transform_scale(self, x, likelihood_hessian=None, direction="inverse"):
        if isinstance(x, (np.ndarray, list)):
            x = np.array(x, dtype=float)
            mask = self._log_transform
            if direction == "inverse":
                # np.where evaluates both branches elementwise, so np.exp(x) would
                # otherwise run (and could overflow) at unmasked entries whose value
                # is discarded anyway (the linear omega_* parameters). Zero those out
                # before exp so no spurious overflow is produced -- mirroring the
                # forward-direction safe_x guard below.
                safe_x = np.where(mask, x, 0.0)
                z = np.where(mask, np.exp(safe_x), x)
            elif direction == "forward":
                # np.where evaluates both branches elementwise, so np.log(x) would
                # otherwise be computed (and warn) even at unmasked entries, which
                # may legitimately be zero or negative (the omega_* parameters).
                safe_x = np.where(mask, x, 1.0)
                z = np.where(mask, np.log(safe_x), x)
            else:
                raise ValueError("Transformation direction not recognized.")

            if not isinstance(likelihood_hessian, np.ndarray):
                return z

            # Jacobian of the (partially log) reparameterization. Same safe_x guard:
            # exp only at the log-transformed entries, 1.0 (identity) elsewhere.
            safe_x = np.where(mask, x, 0.0)
            jac_diag = np.where(mask, np.exp(safe_x), 1.0)
            J = np.diag(jac_diag)

            if direction == "inverse":
                Ji = np.linalg.inv(J)
                H = Ji.transpose() @ likelihood_hessian @ Ji
            elif direction == "forward":
                H = J.transpose() @ likelihood_hessian @ J

            return z, H

    def fit_map(self, *args, **kwargs):
        raise NotImplementedError(
            "fit_map is not available: the base CommonFittingMethods class does not "
            "define a fit_map method to inherit from (see heliosoil.fitting). Use "
            "fit_mle instead."
        )

    def fit_mle(self, simulation_inputs, reflectance_data, verbose=True, x0=None, transform_to_original_scale=False, save_file=None, **optim_kwargs):
        _check_keys(simulation_inputs, reflectance_data)

        n_mean = len(self._mean_param_names)
        n_sigma = len(self._sigma_param_names)

        if x0 is None:
            if "gravitational" in self._component_keys:
                _print_if("Getting initial mean-parameter guess via a plain least-squares fit (all wind coefficients set to 0) ...", verbose)
                other_means = self._mean_param_names[1:]
                saved = {name: getattr(self, name) for name in other_means}
                for name in other_means:
                    setattr(self, name, 0.0)
                p0, sse = self.fit_least_squares(simulation_inputs, reflectance_data, verbose=False)
                for name, value in saved.items():
                    setattr(self, name, value)

                mean_inits = [0.0] * n_mean
                mean_inits[0] = p0

                def nloglike1d(s):
                    return self._negative_log_likelihood(mean_inits + [s] * n_sigma, simulation_inputs, reflectance_data)

                s0 = minimize_scalar(nloglike1d, bounds=(smb.tol, sse), method="Bounded")
                x0 = np.array(mean_inits + [s0.x] * n_sigma)
            else:
                _print_if("No gravitational component active; warm-starting all means at 0 and all sigmas at a small default (1e-4).", verbose)
                x0 = np.array([0.0] * n_mean + [1e-4] * n_sigma)
            _print_if("x0 = " + str(x0), verbose)

        _print_if("Getting MLE estimates ... ", verbose)
        y, y_cov = CommonFittingMethods.fit_mle(
            self, simulation_inputs, reflectance_data, verbose=False, x0=x0, transform_to_original_scale=False, **optim_kwargs
        )
        H_log = np.linalg.inv(y_cov)

        _print_if("========== MLE Estimates ======== ", verbose)
        if transform_to_original_scale:
            x_hat, H = self.transform_scale(y, H_log)
            x_hat_cov = np.linalg.inv(H)
        else:
            x_hat = y
            x_hat_cov = y_cov

        s = _std_errors_from_cov(x_hat_cov)
        x_ci = x_hat + 1.96 * s * np.array([[-1], [1]])
        for i, name in enumerate(self._param_names):
            label = name if transform_to_original_scale else f"log({name})" if self._log_transform[i] else name
            _print_if(f"{label} = {x_hat[i]:.3e}", verbose)
            _print_if(f"95% confidence interval for {label}: [{x_ci[0, i]:.3e}, {x_ci[1, i]:.3e}]", verbose)

        return x_hat, x_hat_cov

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
