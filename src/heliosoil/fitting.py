import heliosoil.base_models as smb
from heliosoil.utilities import (
    logger,
    _print_if,
    _check_keys,
    _parse_dust_str,
    _std_errors_from_cov,
    _nominal_reflectance_series,
    _nominal_reflectance_anchor,
    gravitational_settling_factor,
)
import numpy as np
from typing import NamedTuple, Optional
from numpy import radians as rad
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import lsq_linear, minimize_scalar, minimize
from scipy.special import expit, logit
import numdifftools as ndt
import pickle


# ----------------------------------------------------------------------------------
# Parameter transforms. Every parameter is optimised on an unconstrained "fitted" scale
# and stored on its natural one; these name the map between them. Stating the map per
# parameter, rather than per model class, is what lets one transform_scale serve models
# whose parameters are variously positive (log), greater than one (log-log), bounded to
# the unit interval (logit) or unconstrained (identity).
# ----------------------------------------------------------------------------------

TRANSFORM_IDENTITY = "identity"
TRANSFORM_LOG = "log"
TRANSFORM_LOGLOG = "loglog"
TRANSFORM_LOGIT = "logit"

_TRANSFORM_LABELS = {
    TRANSFORM_IDENTITY: "{0}",
    TRANSFORM_LOG: "log({0})",
    TRANSFORM_LOGLOG: "log(log({0}))",
    TRANSFORM_LOGIT: "logit({0})",
}

# Transforms whose natural scale is bounded below at zero. Used to decide which mean
# parameters a least-squares fit must keep positive.
_POSITIVE_TRANSFORMS = (TRANSFORM_LOG, TRANSFORM_LOGLOG)


def _to_natural(code, y):
    """One parameter, fitted scale -> natural scale."""
    if code == TRANSFORM_IDENTITY:
        return float(y)
    if code == TRANSFORM_LOG:
        return float(np.exp(y))
    if code == TRANSFORM_LOGLOG:
        # A double exponential overflows for a large fitted value. That is left to
        # overflow to inf rather than clipped: fit_mle checks the converged optimum for
        # finiteness and reports an unidentified model, which a clip would hide.
        return float(np.exp(np.exp(y)))
    if code == TRANSFORM_LOGIT:
        return float(expit(y))
    raise ValueError(f"Unknown parameter transform '{code}'.")


def _to_fitted(code, x):
    """One parameter, natural scale -> fitted scale."""
    if code == TRANSFORM_IDENTITY:
        return float(x)
    if code == TRANSFORM_LOG:
        return float(np.log(x))
    if code == TRANSFORM_LOGLOG:
        return float(np.log(np.log(x)))
    if code == TRANSFORM_LOGIT:
        return float(logit(x))
    raise ValueError(f"Unknown parameter transform '{code}'.")


def _natural_derivative(code, y):
    """d(natural)/d(fitted) for one parameter, evaluated at the FITTED point ``y``.

    This is the diagonal entry of the Jacobian the delta method needs. Evaluating at the
    fitted point (rather than the natural one) keeps the expressions in closed form:
    ``exp(y)`` for a log parameter, ``exp(y + exp(y))`` for a log-log one.
    """
    if code == TRANSFORM_IDENTITY:
        return 1.0
    if code == TRANSFORM_LOG:
        return float(np.exp(y))
    if code == TRANSFORM_LOGLOG:
        return float(np.exp(y + np.exp(y)))
    if code == TRANSFORM_LOGIT:
        p = expit(y)
        return float(p * (1.0 - p))
    raise ValueError(f"Unknown parameter transform '{code}'.")


def kappa_param_name(sigma_name):
    """The common-variance-fraction attribute paired with a noise magnitude.

    ``sigma_dep`` -> ``kappa_dep``, ``sigma_dep_gamma`` -> ``kappa_dep_gamma``. A module
    function rather than a method because the model bases that declare the magnitudes
    (heliosoil.horizontal_impaction.ConstantMeanWindBase) are separate from the fitting
    mixin that consumes them, and the naming rule should be stated once.

    Args:
        sigma_name (str): The magnitude's attribute name.

    Returns:
        str: The paired common-fraction attribute name.

    Raises:
        ValueError: If the magnitude does not follow the ``sigma...`` convention.
    """
    if not sigma_name.startswith("sigma"):
        raise ValueError(f"Noise magnitude '{sigma_name}' does not follow the 'sigma...' naming convention, so its kappa cannot be named.")
    return "kappa" + sigma_name[len("sigma") :]


class NoiseChannel(NamedTuple):
    """One independent deposition-noise process and the loading that scales it.

    A model's deposition noise is a sum of such channels. The plain constant-mean and
    semi-physical models have one; the wind model has one per active mechanism. Each
    contributes ``sigma**2 * loading**2`` to the variance of an interval's deposition, and
    -- once mirrors are no longer treated as independent -- couples mirrors in proportion
    to the PRODUCT of their loadings, weighted by ``kappa``.

    Attributes:
        name: The model attribute carrying this channel's magnitude, e.g. "sigma_dep".
        loading: Array (n_mirrors, n_times), nonnegative.
        sigma: The magnitude, or None when the model carries no noise process (a
            least-squares fit leaves every sigma unset).
        kappa: Fraction of this channel's variance common to every mirror. Zero means
            independent mirrors, which is what the current likelihoods assume.
    """

    name: str
    loading: np.ndarray
    sigma: Optional[float]
    kappa: float = 0.0

    @property
    def scale(self):
        """RMS loading over the design, the fixed constant that makes magnitudes comparable.

        Channels of different mechanisms carry different physical dimensions (the
        gravitational loading is dimensionless, the wind loadings carry a speed), so their
        sigmas cannot be compared or pooled directly. ``sigma * scale`` is in soiled-area
        units for every channel. See the soiling model notes, eq. (88).
        """
        return float(np.sqrt(np.mean(np.asarray(self.loading, dtype=float) ** 2)))


class CommonFittingMethods:

    # Fraction of the deposition variance common to every mirror at the site, used when
    # every mechanism shares one value. Zero gives each mirror independent noise, which is
    # what the diagonal likelihood assumes.
    common_variance_fraction = 0.0

    # How the deposition noise is correlated across mirrors. See set_variance_model.
    variance_model = "independent"
    _endpoint_correction = None

    # ------------------------------------------------------------------
    # The variance structure: how much of each mechanism's noise is shared between the
    # mirrors at a site. The magnitudes (sigma) live on the model as named attributes and
    # so do the common fractions (kappa), one per magnitude. Neither is read positionally
    # out of a parameter vector -- update_model_parameters writes them, noise_channels
    # reads them -- which is what lets one likelihood serve models with one mechanism and
    # models with five.
    # ------------------------------------------------------------------

    @property
    def _kappa_param_names(self):
        """One common-fraction name per noise magnitude, in the same order."""
        return tuple(kappa_param_name(name) for name in self._sigma_param_names)

    @property
    def fitted_kappa_names(self):
        """The common fractions carried in the parameter vector, under the current model.

        None under "independent"; one pooled fraction under "shared_kappa"; one per
        mechanism under "per_mechanism".
        """
        if self.variance_model == "independent":
            return ()
        if self.variance_model == "shared_kappa":
            return ("common_variance_fraction",)
        return self._kappa_param_names

    @property
    def parameter_names(self):
        """Every fitted parameter, in parameter-vector order: means, then magnitudes, then
        common fractions.

        Distinct from ``_param_names``, which names only the mean and magnitude parameters
        that enter the forward model. The common fractions affect the likelihood but not
        the prediction, so they belong here and not there.
        """
        return list(self._mean_param_names) + list(self._sigma_param_names) + list(self.fitted_kappa_names)

    @property
    def transform_codes(self):
        """One transform per entry of ``parameter_names``, in the same order."""
        return tuple(self._mean_transform_codes) + (TRANSFORM_LOG,) * len(self._sigma_param_names) + (TRANSFORM_LOGIT,) * len(self.fitted_kappa_names)

    def _codes_for_length(self, n):
        """The leading transforms for a parameter vector of length ``n``.

        Three lengths are meaningful: the mean parameters alone (what a least-squares fit
        returns), the means and magnitudes, and the complete vector. Anything else is a
        caller error worth catching here rather than in a silent broadcast downstream.
        """
        codes = self.transform_codes
        n_mean = len(self._mean_param_names)
        allowed = sorted({n_mean, n_mean + len(self._sigma_param_names), len(codes)})
        if n not in allowed:
            model = getattr(self, "model_name", type(self).__name__)
            labels = ", ".join(f"{k} ({', '.join(self.parameter_names[:k])})" for k in allowed)
            raise ValueError(f"Cannot transform a vector of {n} parameter(s) for model '{model}': expected one of {labels}.")
        return codes[:n]

    def _split_parameters(self, x):
        """Split a parameter vector into its forward-model part and its common fractions."""
        x = list(x)
        n_kappa = len(self.fitted_kappa_names)
        if n_kappa and len(x) == len(self.parameter_names):
            return x[: len(x) - n_kappa], x[len(x) - n_kappa :]
        return x, []

    def _update_kappa_parameters(self, values):
        """Write the common fractions from the tail of a parameter vector onto the model."""
        for name, value in zip(self.fitted_kappa_names, values):
            setattr(self, name, self._validate_kappa(value, name))

    def transform_scale(self, x, likelihood_hessian=None, direction="inverse"):
        """
        Map parameters between their natural scale and the unconstrained fitting scale.

        ``direction="forward"`` goes natural -> fitted, ``"inverse"`` fitted -> natural.
        Optimising on the fitted scale enforces each parameter's constraint without bounds:
        ``hrz0 > 1``, ``mu_tilde > 0``, ``sigma > 0``, ``0 < kappa < 1``. Which transform
        applies to which parameter is declared by ``transform_codes``.

        Supplying ``likelihood_hessian`` also returns it transformed by the delta method.
        The Jacobian is diagonal, so the map is elementwise division (or multiplication) by
        its entries -- no matrix inverse, which matters because that inverse is
        near-singular exactly when a variance parameter has fitted to its bound.

        Args:
            x: Parameter vector, on the scale ``direction`` starts from.
            likelihood_hessian (numpy.ndarray, optional): Hessian on that same scale.
            direction (str): "forward" or "inverse".

        Returns:
            numpy.ndarray, or (numpy.ndarray, numpy.ndarray) when a Hessian is supplied.

        Raises:
            ValueError: On an unrecognised direction, or a vector of unusable length.
        """
        if not isinstance(x, (list, tuple, np.ndarray)):
            raise ValueError(f"transform_scale takes a parameter vector, got {type(x).__name__}. Wrap a single parameter in a list.")
        x = np.asarray(x, dtype=float).reshape(-1)
        codes = self._codes_for_length(len(x))

        # The optimiser probes extreme values while searching; the resulting transient
        # overflow is not informative, and fit_mle checks the converged optimum instead.
        with np.errstate(over="ignore"):
            if direction == "inverse":
                z = np.array([_to_natural(c, v) for c, v in zip(codes, x)])
                jacobian = np.array([_natural_derivative(c, v) for c, v in zip(codes, x)])
            elif direction == "forward":
                z = np.array([_to_fitted(c, v) for c, v in zip(codes, x)])
                jacobian = np.array([_natural_derivative(c, v) for c, v in zip(codes, z)])
            else:
                raise ValueError("Transformation direction not recognized.")

        if not isinstance(likelihood_hessian, np.ndarray):
            # not `is None`: callers pass an array, and `array is None` is not the test.
            return z

        # With J = d(natural)/d(fitted) diagonal, a Hessian with respect to the fitted
        # parameters becomes J^-T H J^-1 with respect to the natural ones, and the reverse
        # is J^T H J.
        if direction == "inverse":
            return z, (likelihood_hessian / jacobian[:, None]) / jacobian[None, :]
        return z, (jacobian[:, None] * likelihood_hessian) * jacobian[None, :]

    def natural_scale_jacobian(self, y):
        """d(natural)/d(fitted) at the fitted point ``y``, elementwise.

        The delta method for a covariance runs the other way from ``transform_scale``'s
        Hessian: ``cov -> J cov J'``. Exposed so ``fit_mle`` can transform a covariance
        directly instead of inverting it to a Hessian and back, a round trip that fails
        whenever a variance parameter has fitted to its bound.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        codes = self._codes_for_length(len(y))
        with np.errstate(over="ignore"):
            return np.array([_natural_derivative(c, v) for c, v in zip(codes, y)])

    @property
    def endpoint_correction(self):
        """Whether consecutive differences share their endpoint measurement noise.

        Defaults to following the variance model, so that "independent" reproduces earlier
        fits exactly and the multi-mirror models get the correlation they imply.
        """
        if self._endpoint_correction is None:
            return self.variance_model != "independent"
        return bool(self._endpoint_correction)

    def set_variance_model(self, variance_model="independent", endpoint_correction=None, kappa=None):
        """
        Choose how the deposition noise is correlated across mirrors.

        - ``"independent"``: every mirror's deposition noise is its own. The differences are
          then independent too, so the likelihood is the diagonal one and each measurement
          contributes a scalar term. This is the default and reproduces earlier fits.
        - ``"shared_kappa"``: one common fraction ``kappa`` across all mechanisms, held in
          ``common_variance_fraction``. Mirrors at a site share that fraction of every
          mechanism's noise during an interval.
        - ``"per_mechanism"``: a common fraction per mechanism, held in the attributes named
          by ``_kappa_param_names``. Set them individually after calling this.

        Separating a mechanism's common part from its mirror-specific part needs at least
        two mirrors that the mechanism actually loads in the same experiment; with one, the
        likelihood does not depend on its kappa at all.

        Args:
            variance_model (str): "independent", "shared_kappa" or "per_mechanism".
            endpoint_correction (bool, optional): Whether to model the correlation between
                consecutive reflectance differences created by the measurement they share.
                Defaults to following ``variance_model``.
            kappa (float, optional): Initial common fraction. Sets
                ``common_variance_fraction`` under "shared_kappa", and seeds every
                per-mechanism attribute under "per_mechanism".

        Raises:
            ValueError: If ``variance_model`` is not recognised, or ``kappa`` is outside [0, 1].
        """
        allowed = ("independent", "shared_kappa", "per_mechanism")
        if variance_model not in allowed:
            raise ValueError(f"variance_model must be one of {allowed}, got {variance_model!r}.")

        if kappa is not None:
            kappa = self._validate_kappa(kappa, "kappa")

        self.variance_model = variance_model
        self._endpoint_correction = endpoint_correction

        if variance_model == "shared_kappa" and kappa is not None:
            self.common_variance_fraction = kappa
        elif variance_model == "per_mechanism":
            # Seed from the shared value so that switching models is a no-op until the
            # individual fractions are set (or, at Step 7, fitted).
            seed = self.common_variance_fraction if kappa is None else kappa
            for name in self._kappa_param_names:
                if getattr(self, name, None) is None:
                    setattr(self, name, seed)

    @staticmethod
    def _validate_kappa(value, name):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"The common variance fraction {name} must be in [0, 1], got {value}.")
        return value

    def loaded_mirror_counts(self, simulation_inputs, reflectance_data):
        """How many mirrors each mechanism actually loads, at best, in any one experiment.

        A mechanism's common fraction is identified by mirrors that feel it *together*: the
        scatter between them carries the mirror-specific part and their shared movement the
        common part. A mirror the mechanism does not load (gravitational settling on a
        face-down mirror, normal wind on a horizontal one) contributes to neither, so it is
        not counted. See the soiling model notes, eq. (94).

        Args:
            simulation_inputs (SimulationInputs): Simulation inputs.
            reflectance_data (ReflectanceMeasurements): Measurement data.

        Returns:
            dict: Magnitude name -> the largest number of mirrors it loads in one experiment.
        """
        counts = {name: 0 for name in self._sigma_param_names}
        for f in reflectance_data.times:
            for channel in self.noise_channels(f, simulation_inputs):
                loading = np.asarray(channel.loading, dtype=float)
                loaded = int(np.count_nonzero(np.any(loading != 0.0, axis=1)))
                counts[channel.name] = max(counts.get(channel.name, 0), loaded)
        return counts

    def _check_variance_components_identifiable(self, simulation_inputs, reflectance_data):
        """
        Confirm the data can separate the variance components the current model asks for.

        Splitting a mechanism's variance into a common and a mirror-specific part needs at
        least two mirrors that the mechanism loads in the same experiment. Under
        "shared_kappa" one such mechanism suffices, since the fraction is pooled; under
        "per_mechanism" every mechanism carrying a magnitude needs its own pair.

        Args:
            simulation_inputs (SimulationInputs): Simulation inputs.
            reflectance_data (ReflectanceMeasurements): Measurement data.

        Raises:
            ValueError: If the requested variance model is not identified by this data.
        """
        if self.variance_model == "independent":
            return

        counts = self.loaded_mirror_counts(simulation_inputs, reflectance_data)
        active = [name for name in self._sigma_param_names if getattr(self, name, None) is not None]
        # Before a fit no magnitude need be set yet; then every mechanism is a candidate.
        if not active:
            active = list(self._sigma_param_names)
        detail = ", ".join(f"{name}: {counts.get(name, 0)}" for name in active)

        if self.variance_model == "shared_kappa":
            if max((counts.get(name, 0) for name in active), default=0) < 2:
                raise ValueError(f"Variance model 'shared_kappa' needs at least two mirrors loaded by the same mechanism in one experiment, but no mechanism reaches two ({detail}). Use set_variance_model('independent') to fit the total deposition variance only.")
            return

        starved = [name for name in active if counts.get(name, 0) < 2]
        if starved:
            raise ValueError(f"Variance model 'per_mechanism' needs at least two mirrors loaded by EACH mechanism in one experiment, but {starved} fall short ({detail}). Drop those mechanisms, add mirrors whose orientation they act on, or use set_variance_model('shared_kappa').")

    def noise_channel_scales(self, simulation_inputs, reflectance_data):
        """RMS loading per mechanism, pooled over the whole training design.

        The fixed constant ``s_c`` of the soiling model notes, eq. (88). Magnitudes of
        different mechanisms carry different physical dimensions -- the gravitational
        loading is dimensionless, the wind loadings carry a speed -- so ``sigma_c`` alone
        cannot be compared across mechanisms, whereas ``tau_c = sigma_c * s_c`` is in
        soiled-area units for every one of them.

        Args:
            simulation_inputs (SimulationInputs): Simulation inputs.
            reflectance_data (ReflectanceMeasurements): Measurement data, which fixes which
                experiments make up the design.

        Returns:
            dict: Magnitude name -> its RMS loading.
        """
        totals = {name: [0.0, 0] for name in self._sigma_param_names}
        for f in reflectance_data.times:
            for channel in self.noise_channels(f, simulation_inputs):
                loading = np.asarray(channel.loading, dtype=float)
                total, count = totals.get(channel.name, [0.0, 0])
                totals[channel.name] = [total + float(np.sum(loading**2)), count + loading.size]
        return {name: (float(np.sqrt(total / count)) if count else 0.0) for name, (total, count) in totals.items()}

    @staticmethod
    def _delta_method_interval(centre, standard_deviation, inverse):
        """A 95% interval mapped through a monotone inverse transform.

        Returns ``None`` for the interval where the standard error is not usable -- a
        non-finite or negative curvature, which is what a parameter sitting on its bound
        produces. A bare NaN interval would read as a number; None does not.
        """
        if not np.isfinite(standard_deviation) or standard_deviation < 0.0:
            return None
        with np.errstate(over="ignore"):
            return (float(inverse(centre - 1.96 * standard_deviation)), float(inverse(centre + 1.96 * standard_deviation)))

    def variance_component_summary(self, y_hat, y_cov, simulation_inputs=None, reflectance_data=None):
        """
        Per-mechanism noise report: magnitude, scaled magnitude, variance share and common
        fraction, each with a 95% interval by the delta method.

        Everything is computed on the fitted scale, where the intervals are symmetric, and
        mapped back through the transform -- so ``sigma`` and ``tau`` stay positive and
        ``kappa`` and the share stay inside [0, 1] by construction.

        ``tau_c = sigma_c * s_c`` and the share ``pi_c = tau_c**2 / sum_c' tau_c'**2`` are
        diagnostics computed from the fit, not fitted parameters (see the port plan, D7 and
        D8). The share's gradient on the fitted scale is
        ``d logit(pi_c) / d log(sigma_j) = 2 (delta_cj - pi_j) / (1 - pi_c)``.

        Args:
            y_hat (numpy.ndarray): Estimates on the fitted scale, as ``fit_mle`` returns
                them with ``transform_to_original_scale=False``.
            y_cov (numpy.ndarray): Their covariance on the fitted scale.
            simulation_inputs (SimulationInputs, optional): Needed for ``tau`` and the
                share; without it those entries are omitted.
            reflectance_data (ReflectanceMeasurements, optional): Likewise.

        Returns:
            dict: Magnitude name -> dict of entries, each ``(estimate, interval_or_None)``.
        """
        y_hat = np.asarray(y_hat, dtype=float)
        y_cov = np.asarray(y_cov, dtype=float)
        n_mean = len(self._mean_param_names)
        sigma_names = list(self._sigma_param_names)
        kappa_names = list(self.fitted_kappa_names)

        def standard_deviation(gradient):
            return float(np.sqrt(max(float(gradient @ y_cov @ gradient), 0.0))) if np.all(np.isfinite(gradient)) else np.nan

        scales = None
        if simulation_inputs is not None and reflectance_data is not None:
            scales = self.noise_channel_scales(simulation_inputs, reflectance_data)
            tau_squared = {}
            for index, name in enumerate(sigma_names):
                magnitude = getattr(self, name, None)
                tau_squared[name] = 0.0 if magnitude is None else float(np.exp(2.0 * y_hat[n_mean + index]) * scales[name] ** 2)
            total = sum(tau_squared.values())
            shares = {name: (tau_squared[name] / total if total > 0.0 else 0.0) for name in sigma_names}

        summary = {}
        for index, name in enumerate(sigma_names):
            if getattr(self, name, None) is None:
                continue
            entry = {}
            i = n_mean + index

            gradient = np.zeros(len(y_hat))
            gradient[i] = 1.0
            sd = standard_deviation(gradient)
            entry["sigma"] = (float(np.exp(y_hat[i])), self._delta_method_interval(y_hat[i], sd, np.exp))

            if scales is not None:
                # tau differs from sigma by a constant factor, so on the log scale it is a
                # shift: the same standard error, a translated centre.
                log_tau = y_hat[i] + np.log(scales[name]) if scales[name] > 0.0 else -np.inf
                entry["tau"] = (float(np.exp(log_tau)), self._delta_method_interval(log_tau, sd, np.exp))

                share = shares[name]
                if 0.0 < share < 1.0:
                    share_gradient = np.zeros(len(y_hat))
                    for other, other_name in enumerate(sigma_names):
                        share_gradient[n_mean + other] = 2.0 * ((1.0 if other_name == name else 0.0) - shares[other_name]) / (1.0 - share)
                    entry["share"] = (share, self._delta_method_interval(logit(share), standard_deviation(share_gradient), expit))
                else:
                    # Exactly 0 or 1 means the mechanism is alone, or its magnitude sat on
                    # the boundary. Either way the logit is not finite and no interval is
                    # meaningful.
                    entry["share"] = (share, None)

            if kappa_names:
                j = len(y_hat) - len(kappa_names) + (0 if self.variance_model == "shared_kappa" else index)
                gradient = np.zeros(len(y_hat))
                gradient[j] = 1.0
                entry["kappa"] = (float(expit(y_hat[j])), self._delta_method_interval(y_hat[j], standard_deviation(gradient), expit))

            summary[name] = entry
        return summary

    def _kappa_for(self, sigma_name):
        """This mechanism's common fraction under the current variance model."""
        if self.variance_model == "independent":
            return 0.0
        if self.variance_model == "shared_kappa":
            return self._validate_kappa(self.common_variance_fraction, "common_variance_fraction")
        name = kappa_param_name(sigma_name)
        value = getattr(self, name, None)
        if value is None:
            raise ValueError(f"Variance model 'per_mechanism' needs a common fraction for every mechanism, but '{name}' is unset. Call set_variance_model(..., kappa=...) or set it directly.")
        return self._validate_kappa(value, name)

    def compute_soiling_factor(self, reflectance_data=None):
        # Converts helios.delta_soiled_area into an accumulated area loss and
        # populates helios.soiling_factor.

        files = list(self.helios.tilt.keys())
        helios = self.helios
        helios.soiling_factor = {f: None for f in files}  # clear the soiling factor

        for f in files:
            if reflectance_data is None:
                N_helios = helios.tilt[f].shape[0]
                cumulative_soil0 = np.zeros(N_helios)  # start from clean
            else:
                inc_factor = self.helios.inc_ref_factor[f].squeeze()
                # nom_ref_anchor is the nominal (as-if-freshly-cleaned) reflectance at
                # the moment rho0 was recorded: the fixed constant by default, or a
                # reference-mirror-derived, per-mirror value when reflectance_data
                # carries one (see heliosoil.utilities._nominal_reflectance_anchor).
                nom_ref_anchor = _nominal_reflectance_anchor(reflectance_data, f, self.helios.nominal_reflectance)
                cumulative_soil0 = (1 - reflectance_data.rho0[f] / nom_ref_anchor) / inc_factor  # back-calculated soiled area from measurement

            cumulative_soil = np.c_[cumulative_soil0, helios.delta_soiled_area[f]]
            cumulative_soil = np.cumsum(cumulative_soil, axis=1)  # accumulate soiling
            helios.soiling_factor[f] = (
                1 - cumulative_soil[:, 1::] * helios.inc_ref_factor[f]
            )  # soiling factor, still to be multiplied by nominal reflectance

        self.helios = helios

    def _dust_loading(self, f, simulation_inputs):
        """
        Dimensionless airborne dust loading alpha_j: the measured mass over the same
        measure's mass in the reference size distribution. A site quantity, shared by
        every mirror and every mechanism that draws on this dust channel.

        Args:
            f: Experiment (file) key.
            simulation_inputs (SimulationInputs): Supplies the concentration and type.

        Returns:
            numpy.ndarray: Loading, shape (n_times,).

        Raises:
            ValueError: If the simulation's dust type is not present on the Dust class.
        """
        sim_in = simulation_inputs
        try:
            attr = _parse_dust_str(sim_in.dust_type[f])
            den = getattr(sim_in.dust, attr)  # dust.(sim_in.dust_type[f])
        except Exception:
            raise ValueError(
                "Dust measurement "
                + sim_in.dust_type[f]
                + " not present in dust class. Use dust_type="
                + sim_in.dust_type[f]
                + " option when initializing the model"
            )
        return sim_in.dust_concentration[f] / den[f]

    def noise_channels(self, f, simulation_inputs, sigma_override=None):
        """
        The deposition-noise channels active for experiment ``f``.

        One channel per independent noise process. This model family has a single one:
        deposition scaled by the dust loading and the mirror's horizontal projection.
        Models with several mechanisms (heliosoil.horizontal_impaction) override this and
        return one channel each.

        Everything that consumes the noise -- the difference variance here, and the
        multi-mirror covariance built on top of it -- goes through this list, so a model
        gains noise handling by describing its channels rather than by reimplementing the
        assembly.

        Args:
            f: Experiment (file) key.
            simulation_inputs (SimulationInputs): Simulation inputs.
            sigma_override (float, optional): Replaces this single channel's magnitude.
                Used by callers that pass a candidate value positionally rather than
                setting it on the model first.

        Returns:
            list[NoiseChannel]: The active channels, in a fixed order.
        """
        alpha = self._dust_loading(f, simulation_inputs)
        loading = alpha[None, :] * gravitational_settling_factor(self.helios.tilt[f])
        sigma = self.sigma_dep if sigma_override is None else sigma_override
        return [NoiseChannel("sigma_dep", loading, sigma, self._kappa_for("sigma_dep"))]

    def _compute_variance_of_measurements(self, sigma_dep, simulation_inputs, reflectance_data=None):
        """
        Variance of each reflectance difference: deposition noise plus measurement noise.

        Sums the contribution of every channel from ``noise_channels``, treating the
        mirrors as independent and the differences as independent of one another -- so the
        result is one variance per (difference, mirror) rather than a covariance. The
        multi-mirror extension replaces this with a full covariance; the channels it is
        assembled from are the same ones used here.

        Args:
            sigma_dep (float): Candidate magnitude for the single-channel models. Ignored
                by models with more than one channel, whose magnitudes are read from the
                model itself.
            simulation_inputs (SimulationInputs): Simulation inputs.
            reflectance_data (ReflectanceMeasurements): Measurement data.

        Returns:
            dict: Per experiment, an array of shape (n_differences, n_mirrors).
        """
        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        sim_in = simulation_inputs
        files = list(sim_in.time.keys())

        s2total = dict.fromkeys(files)
        for f in files:
            pif = reflectance_data.prediction_indices[f]
            b = self._reflectance_loss_factor(f, reflectance_data)  # per mirror
            meas_sig = reflectance_data.sigma_of_the_mean[f]

            # Difference i spans deposition intervals k_{i-1}+1 ... k_i INCLUSIVE, matching
            # the inclusive cumulative sum in compute_soiling_factor that forms the mean:
            # soiling_factor[k] already contains delta_soiled_area[k]. Taking the cumulative
            # sum at ind2 = k_i (not k_i - 1) is what makes the variance cover the same
            # intervals the mean difference accumulates.
            ind1 = pif[0:-1]
            ind2 = pif[1::]

            deposition = np.zeros((len(ind1), self.helios.tilt[f].shape[0]))
            for channel in self.noise_channels(f, sim_in, sigma_override=sigma_dep):
                if channel.sigma is None:
                    continue
                c2t = np.cumsum(channel.loading**2, axis=1).transpose()
                deposition = deposition + channel.sigma**2 * (c2t[ind2, :] - c2t[ind1, :])

            s2total[f] = b**2 * deposition + meas_sig[0:-1, :] ** 2 + meas_sig[1::, :] ** 2

        return s2total

    # ------------------------------------------------------------------
    # Pieces of the multi-mirror difference covariance that do not depend on the
    # deposition mechanism. Assembled into a full covariance later; on their own they
    # describe how a reflectance DIFFERENCE relates to the measurements behind it.
    # ------------------------------------------------------------------

    def _difference_windows(self, f, reflectance_data):
        """
        Simulation-grid slices spanned by each reflectance difference.

        Difference ``i`` covers deposition intervals ``k_{i-1}+1 ... k_i`` inclusive,
        matching the inclusive cumulative sum in ``compute_soiling_factor`` that forms the
        mean. Because all mirrors in an experiment share a measurement grid, these windows
        partition the timeline: they are disjoint and together cover every interval between
        the first and last measurement.

        Args:
            f: Experiment (file) key.
            reflectance_data (ReflectanceMeasurements): Supplies ``prediction_indices``.

        Returns:
            list[slice]: One slice per difference.
        """
        pi = reflectance_data.prediction_indices[f]
        return [slice(pi[i] + 1, pi[i + 1] + 1) for i in range(len(pi) - 1)]

    def _reflectance_loss_factor(self, f, reflectance_data=None):
        """
        Reflectance lost per unit soiled area, per mirror: ``b = rho_nominal * inc_ref_factor``.

        Returned per mirror rather than as a scalar because the reference-mirror drift
        correction gives each mirror its own nominal reflectance at the moment its rho0 was
        recorded; without a reference column every entry is the same fixed constant.

        Assumes a fixed reflectometer incidence angle within an experiment, so that ``b`` is
        the same at both ends of a reflectance difference.

        Args:
            f: Experiment (file) key.
            reflectance_data (ReflectanceMeasurements, optional): Supplies the
                reference-derived nominal reflectance. Omitted or without one for this
                file, the fixed ``helios.nominal_reflectance`` is used.

        Returns:
            numpy.ndarray: Loss factor per mirror, shape (n_mirrors,).

        Raises:
            ValueError: If ``inc_ref_factor[f]`` is not a single value.
        """
        inc = np.asarray(self.helios.inc_ref_factor[f])
        if inc.size != 1:
            raise ValueError(
                f"Experiment {f} has {inc.size} incidence-reflectance factors. The fitting likelihoods assume a fixed reflectometer "
                "incidence angle per experiment."
            )
        anchor = _nominal_reflectance_anchor(reflectance_data, f, self.helios.nominal_reflectance)
        n_mirrors = self.helios.tilt[f].shape[0]
        return np.broadcast_to(np.asarray(anchor, dtype=float) * float(inc.reshape(-1)[0]), (n_mirrors,)).copy()

    def _differenced_measurement_covariance(self, f, reflectance_data, endpoint_correction=True):
        """
        Covariance of the differenced reflectometer noise, per mirror.

        Stacking a mirror's measurement noise as ``eps`` with diagonal covariance
        ``Sigma_r``, the differences carry ``Delta eps`` for the first-difference operator
        ``Delta``, so their covariance is ``R = Delta Sigma_r Delta'`` -- tridiagonal, with
        ``sigma_{k_i}^2 + sigma_{k_{i-1}}^2`` on the diagonal and ``-sigma_{k_i}^2`` off it.
        The off-diagonal is the negative correlation induced by the measurement two
        consecutive differences share at their common endpoint.

        Setting ``endpoint_correction`` False keeps only the diagonal, i.e. treats the
        differences as independent -- the approximation the current likelihoods make.

        Args:
            f: Experiment (file) key.
            reflectance_data (ReflectanceMeasurements): Supplies ``sigma_of_the_mean``.
            endpoint_correction (bool): Whether to include the shared-endpoint
                off-diagonal terms.

        Returns:
            numpy.ndarray: Covariance, shape (n_mirrors, n_differences, n_differences).
        """
        s2 = np.asarray(reflectance_data.sigma_of_the_mean[f]) ** 2  # (n_meas, n_mirrors)
        n_diff, n_mirrors = s2.shape[0] - 1, s2.shape[1]

        cov = np.zeros((n_mirrors, n_diff, n_diff))
        diag = np.arange(n_diff)
        for p in range(n_mirrors):
            cov[p, diag, diag] = s2[1:, p] + s2[:-1, p]
            if endpoint_correction and n_diff > 1:
                shared = -s2[1:n_diff, p]
                cov[p, diag[:-1], diag[:-1] + 1] = shared
                cov[p, diag[:-1] + 1, diag[:-1]] = shared
        return cov

    # ------------------------------------------------------------------
    # The multi-mirror difference covariance. Notation follows the soiling model notes
    # section 8.3: c indexes a mechanism, p and q mirrors, i and i' reflectance
    # differences, j simulation intervals. a_{c,j}^(p) = b^(p) g_{c,j}^(p) is mirror p's
    # response to mechanism c's noise in interval j, and W_i is the set of intervals
    # spanned by difference i.
    # ------------------------------------------------------------------

    def _deposition_cross_products(self, f, channel, simulation_inputs, reflectance_data):
        """
        Between-mirror cross-products of one channel's response, per reflectance difference.

        Entry ``[i, p, q]`` is ``sum_{j in W_i} a_j_p * a_j_q``, the Gram matrix of the
        mirrors' responses over difference ``i``'s window. Scaling this by the channel's
        variance components (see ``_experiment_covariance``) gives that mechanism's
        contribution to the difference covariance.

        Args:
            f: Experiment (file) key.
            channel (NoiseChannel): The mechanism whose loading is being accumulated.
            simulation_inputs (SimulationInputs): Unused here; kept so the signature matches
                the rest of the covariance builders.
            reflectance_data (ReflectanceMeasurements): Supplies the difference windows and
                the per-mirror nominal reflectance behind ``b``.

        Returns:
            numpy.ndarray: Cross-products, shape (n_differences, n_mirrors, n_mirrors).
        """
        b = self._reflectance_loss_factor(f, reflectance_data)  # (n_mirrors,)
        a = b[:, None] * np.asarray(channel.loading, dtype=float)  # (n_mirrors, n_intervals)
        windows = self._difference_windows(f, reflectance_data)

        n_mirrors = a.shape[0]
        cross_products = np.zeros((len(windows), n_mirrors, n_mirrors))
        for i, window in enumerate(windows):
            a_window = a[:, window]  # (n_mirrors, |W_i|)
            # The matrix product IS the sum over j in W_i, one entry per mirror pair.
            cross_products[i] = a_window @ a_window.transpose()
        return cross_products

    def _experiment_covariance(self, f, simulation_inputs, reflectance_data, endpoint_correction=True):
        """
        Covariance of every reflectance difference in one experiment, across all mirrors.

        Each mechanism's deposition noise is split into a part common to every mirror at the
        site during an interval and a mirror-specific part, in proportion ``kappa`` to
        ``1 - kappa``. The common part couples mirrors; the mirror-specific part and the
        reflectometer noise do not. Entry ``((p, i), (q, i'))`` is

            delta_ii' * sum_c sigma_c**2 * [kappa_c + (1 - kappa_c) * delta_pq]
                              * sum_{j in W_i} a_c_j_p * a_c_j_q
            + delta_pq * R_ii'_p

        The Kronecker delta on the difference index is the windows partitioning the
        timeline, which ``_difference_windows`` guarantees; ``R`` is the differenced
        measurement covariance.

        Entries are ordered mirror-major: difference ``i`` of mirror ``p`` sits at index
        ``p * n_differences + i``.

        The result is positive definite by construction, not by luck: each mechanism's
        contribution is the covariance of an actual random vector, and ``R = Delta Sigma_r
        Delta'`` with ``Delta`` of full row rank is strictly positive definite. A Cholesky
        failure here means the assembly is wrong, not that the problem is ill-conditioned.

        Args:
            f: Experiment (file) key.
            simulation_inputs (SimulationInputs): Simulation inputs, for the channels.
            reflectance_data (ReflectanceMeasurements): Measurement data.
            endpoint_correction (bool): Passed to ``_differenced_measurement_covariance``.

        Returns:
            numpy.ndarray: Covariance, shape (n_mirrors * n_differences, same).
        """
        meas_cov = self._differenced_measurement_covariance(f, reflectance_data, endpoint_correction=endpoint_correction)
        n_mirrors, n_diff = meas_cov.shape[0], meas_cov.shape[1]

        # Deposition first: for each difference, an (n_mirrors, n_mirrors) block summed
        # over mechanisms. Built one mechanism at a time so it reads as the formula above.
        deposition = np.zeros((n_diff, n_mirrors, n_mirrors))
        for channel in self.noise_channels(f, simulation_inputs):
            if channel.sigma is None:
                continue
            # sigma_c^2 * [kappa_c + (1 - kappa_c) * delta_pq], the mirror-pair weight of
            # mechanism c: kappa_c * sigma_c^2 everywhere, sigma_c^2 on the diagonal.
            kappa = self._validate_kappa(channel.kappa, f"kappa of '{channel.name}'")
            weights = channel.sigma**2 * (kappa + (1.0 - kappa) * np.eye(n_mirrors))
            cross_products = self._deposition_cross_products(f, channel, simulation_inputs, reflectance_data)
            for i in range(n_diff):
                deposition[i] = deposition[i] + weights * cross_products[i]

        # Then place block (p, q). Deposition is diagonal in the difference index; the
        # measurement noise is confined to one mirror and is where the off-diagonal in
        # that index comes from.
        cov = np.zeros((n_mirrors * n_diff, n_mirrors * n_diff))
        for p in range(n_mirrors):
            rows = slice(p * n_diff, (p + 1) * n_diff)
            for q in range(n_mirrors):
                columns = slice(q * n_diff, (q + 1) * n_diff)
                block = np.diag(deposition[:, p, q])
                if p == q:
                    block = block + meas_cov[p]
                cov[rows, columns] = block
        return cov

    def _sse(self, params, simulation_inputs, reflectance_data):
        # Computes the sum of squared errors between a soiling model and
        # the reflectance measurements.

        pi = reflectance_data.prediction_indices
        meas = reflectance_data.average

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        sse = 0
        self.update_model_parameters(params)
        self.predict_soiling_factor(simulation_inputs, reflectance_data=reflectance_data, verbose=False)
        sf = self.helios.soiling_factor
        files = list(sf.keys())
        for f in files:
            # nominal reflectance, not rho0: soiling_factor's own initial condition is
            # already anchored to rho0 via compute_soiling_factor's cumulative_soil0,
            # so multiplying by rho0 here again would double-count it (see the
            # identity documented on _nominal_reflectance_series).
            r0 = _nominal_reflectance_series(reflectance_data, f, self.helios.nominal_reflectance)
            rho_prediction = r0 * sf[f][:, pi[f]].transpose()
            sse += np.sum((rho_prediction - meas[f]) ** 2)
        return sse

    def _negative_log_likelihood(self, params, simulation_inputs, reflectance_data):
        """
        Negative log-likelihood under the currently selected variance model.

        Args:
            params: The model's parameter vector, in ``_param_names`` order.
            simulation_inputs (SimulationInputs): Simulation inputs.
            reflectance_data (ReflectanceMeasurements): Measurement data.

        Returns:
            float: The negative log-likelihood.
        """
        if self.variance_model == "independent" and not self.endpoint_correction:
            return self._negative_log_likelihood_diagonal(params, simulation_inputs, reflectance_data)
        return self._negative_log_likelihood_components(params, simulation_inputs, reflectance_data, endpoint_correction=self.endpoint_correction)

    def _difference_residuals(self, f, reflectance_data):
        """
        Observed minus predicted reflectance differences for one experiment.

        Assumes ``predict_soiling_factor`` has already been run for the current parameters.

        Args:
            f: Experiment (file) key.
            reflectance_data (ReflectanceMeasurements): Measurement data.

        Returns:
            numpy.ndarray: Residuals, shape (n_differences, n_mirrors).
        """
        pi = reflectance_data.prediction_indices[f]
        # nominal clean reflectance, not rho0 -- see the identity on _nominal_reflectance_series
        r0 = _nominal_reflectance_series(reflectance_data, f, self.helios.nominal_reflectance)
        rho_prediction = r0 * self.helios.soiling_factor[f][:, pi].transpose()
        return np.diff(reflectance_data.average[f], axis=0) - np.diff(rho_prediction, axis=0)

    def _negative_log_likelihood_diagonal(self, params, simulation_inputs, reflectance_data):
        """
        Negative log-likelihood treating every reflectance difference as independent.

        Each (difference, mirror) contributes a univariate normal term. This is the
        original likelihood and what ``variance_model="independent"`` selects.

        Args:
            params: The model's parameter vector, in ``_param_names`` order.
            simulation_inputs (SimulationInputs): Simulation inputs.
            reflectance_data (ReflectanceMeasurements): Measurement data.

        Returns:
            float: The negative log-likelihood.
        """
        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        sim_in = simulation_inputs
        files = list(reflectance_data.times.keys())

        self.update_model_parameters(params)
        self.predict_soiling_factor(simulation_inputs, reflectance_data=reflectance_data, verbose=False)

        # Compute variance in reflectance, not soiling factor. No sigma is passed
        # positionally: update_model_parameters has just written every noise magnitude onto
        # the model, and noise_channels reads them from there. That holds for one sigma or
        # for several, so this body serves every model in the family.
        s2total = self._compute_variance_of_measurements(None, sim_in, reflectance_data=reflectance_data)

        loglike = 0.0
        for f in files:
            residual = self._difference_residuals(f, reflectance_data)
            # One 2*pi per term, i.e. per (difference, mirror). This used to be counted
            # once per measurement row instead; the difference is an additive constant that
            # no caller reads and that the optimiser cannot see, but the multivariate
            # likelihood below has to normalise correctly, and the two only agree at
            # kappa = 0 if this one does too.
            loglike += -0.5 * residual.size * np.log(2 * np.pi)
            loglike += np.sum(-0.5 * np.log(s2total[f]) - residual**2 / (2 * s2total[f]))

        return -loglike

    def _negative_log_likelihood_components(self, params, simulation_inputs, reflectance_data, endpoint_correction=True):
        """
        Negative log-likelihood with the deposition noise split into variance components.

        Generalises ``_negative_log_likelihood_diagonal`` to mirrors observed together at one
        site, whose deposition noise is partly shared. Each experiment contributes one
        multivariate normal over its stacked reflectance differences, with the covariance
        built by ``_experiment_covariance``. Differences with a missing endpoint measurement
        are dropped, which marginalises them exactly rather than approximating them.

        Args:
            params: The model's parameter vector, in ``_param_names`` order. The common
                fractions are NOT part of it: they are read from the model (see
                ``set_variance_model``).
            simulation_inputs (SimulationInputs): Simulation inputs.
            reflectance_data (ReflectanceMeasurements): Measurement data.
            endpoint_correction (bool): Whether to model the correlation between consecutive
                differences induced by the measurement they share.

        Returns:
            float: The negative log-likelihood.
        """
        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        self.update_model_parameters(params)
        self.predict_soiling_factor(simulation_inputs, reflectance_data=reflectance_data, verbose=False)

        nll = 0.0
        for f in list(reflectance_data.times.keys()):
            # mirror-major ordering, matching _experiment_covariance: difference i of
            # mirror p at index p * n_differences + i.
            residual = self._difference_residuals(f, reflectance_data).transpose().ravel()
            cov = self._experiment_covariance(f, simulation_inputs, reflectance_data, endpoint_correction=endpoint_correction)

            observed = np.isfinite(residual)
            if not observed.all():
                residual = residual[observed]
                cov = cov[observed, :][:, observed]
            if residual.size == 0:
                continue

            factor = cho_factor(cov, lower=True)
            # log|cov| = 2 * sum(log(diag(L))) for the lower Cholesky factor L.
            nll += 0.5 * (residual.size * np.log(2 * np.pi) + 2.0 * np.sum(np.log(np.diag(factor[0]))) + residual @ cho_solve(factor, residual))

        return nll

    def _logpost(self, y, simulation_inputs, reflectance_data, priors):
        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        x = self.transform_scale(y)
        likelihood = -self._negative_log_likelihood(x, simulation_inputs, reflectance_data)

        unnormalized_posterior = likelihood
        names = list(priors.keys())
        for ii in range(len(names)):
            k = names[ii]
            unnormalized_posterior += priors[k].logpdf(y[ii])

        return unnormalized_posterior

    # ------------------------------------------------------------------
    # Least-squares fitting, shared by every model that opts in.
    #
    # A least-squares fit estimates the MEAN parameters only: the sum of squares does not
    # depend on the noise scales, so a model fitted this way carries no noise process at
    # all. Opting in means declaring the split below and implementing
    # _fitted_scale_jacobian; the fit itself is either AffineMeanLeastSquares.fit_ls (one
    # linear solve, for a model whose prediction is affine in its means) or a model-specific
    # nonlinear fit (see SemiPhysical.fit_ls).
    # ------------------------------------------------------------------

    _mean_param_names = ()
    _sigma_param_names = ()
    # One transform per mean parameter, aligned with _mean_param_names. Magnitudes are
    # always log and common fractions always logit, so only the means need declaring.
    _mean_transform_codes = ()
    # Search bracket for the single-parameter bounded scalar fits (fit_least_squares, and the
    # nonlinear SemiPhysical.fit_ls built on it). The lower bound sits just above 1 because
    # hrz0 is represented as log(log(hrz0)); a fit landing on either end is reported rather
    # than passed off as converged (see _warn_if_at_scalar_bound).
    _LS_SCALAR_BOUNDS = (1e-6 + 1.0, 1000.0)

    def _warn_if_at_scalar_bound(self, name, value):
        """Flag an estimate that stopped on the search bracket instead of at an interior
        optimum -- the objective was still improving when the search ran out of room, so the
        value is an artefact of the bracket and its curvature-based standard error is
        meaningless. The same class of diagnostic fit_mle emits when a parameter overflows.

        The tolerance is a fraction of the bracket width, not of the value: a bounded search
        converges only to its own xatol (1e-5 by default), so it lands *near* the bound rather
        than on it -- e.g. 999.99998 against an upper bound of 1000 -- and an exact comparison
        would never fire."""
        lower, upper = self._LS_SCALAR_BOUNDS
        tol = 1e-4 * (upper - lower)
        if value <= lower + tol or value >= upper - tol:
            logger.warning(
                f"Least-squares fit for {name} stopped on its search bound ({value:.4g}, bracket "
                f"[{lower:.4g}, {upper:.4g}]): the sum of squares was still improving there, so this "
                "is a bound artefact rather than an optimum and the parameter is not identified by "
                "this data. Its confidence interval is not meaningful."
            )

    @property
    def mean_parameter_names(self):
        """The mean parameters, in parameter-vector order -- what a least-squares fit
        estimates, and the ordering of the estimate fit_ls returns."""
        return list(self._mean_param_names)

    def _fitted_scale_jacobian(self, x):
        """d(transform_scale(x, "forward")) / dx, elementwise, over the mean parameters.

        The delta method needs this derivative to carry a least-squares covariance onto the
        fitted scale. It is the reciprocal of ``natural_scale_jacobian`` at the matching
        point, so it is derived from ``transform_codes`` rather than restated per class --
        the two cannot disagree because there is only one statement of the transform.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        codes = self._codes_for_length(len(x))
        with np.errstate(over="ignore"):
            return np.array([1.0 / _natural_derivative(c, _to_fitted(c, v)) for c, v in zip(codes, x)])

    def _fitted_scale_label(self, name, i):
        """How parameter `i` is spelled on the fitted scale, for reporting -- "log(x)",
        "log(log(x))", "logit(x)", or the bare name where it is not transformed."""
        return _TRANSFORM_LABELS[self.transform_codes[i]].format(name)

    def _ls_lower_bounded(self):
        """Which mean parameters a least-squares fit must keep positive: those whose
        transform implies it. The linear ones (the wind coefficients) may legitimately fit
        negative -- scouring -- and are left free."""
        codes = self.transform_codes[: len(self._mean_param_names)]
        return np.array([c in _POSITIVE_TRANSFORMS for c in codes])

    def _predicted_reflectance(self, simulation_inputs, reflectance_data):
        """Predicted reflectance at the measurement times, {file: (N_measurements, N_helios)}
        -- exactly the quantity _sse compares against reflectance_data.average, computed from
        the parameters currently set on self."""
        self.predict_soiling_factor(simulation_inputs, reflectance_data=reflectance_data, verbose=False)
        sf = self.helios.soiling_factor
        pi = reflectance_data.prediction_indices
        out = {}
        for f in sf:
            r0 = _nominal_reflectance_series(reflectance_data, f, self.helios.nominal_reflectance)
            out[f] = np.asarray(r0 * sf[f][:, pi[f]].transpose(), dtype=float)
        return out

    @staticmethod
    def _ls_stack(per_file, files, residuals):
        """Flatten a {file: (N_measurements, N_helios)} map into one observation vector.

        residuals="level" keeps the reflectance itself (the objective _sse defines);
        "increment" differences consecutive measurements, the quantity the likelihood is
        written on and that the reported daily-soiling-rate errors score."""
        if residuals not in ("level", "increment"):
            raise ValueError(f"residuals must be 'level' or 'increment', got {residuals!r}.")
        parts = [np.diff(per_file[f], axis=0) if residuals == "increment" else per_file[f] for f in files]
        return np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts])

    def _finish_least_squares(self, theta, cov_theta, transform_to_original_scale, verbose):
        """Common tail of every fit_ls: adopt the estimate, drop any noise model, and report.

        The sigmas are cleared rather than merely left alone: a stale value (imported from
        the parameter file, or left by an earlier MLE fit) would otherwise attach a
        prediction interval to a fit that never estimated one."""
        self.update_model_parameters(theta)  # zip-truncates to the means for a multi-parameter model
        for name in self._sigma_param_names:
            setattr(self, name, None)

        _print_if("========== Least-squares Estimates ======== ", verbose)
        if transform_to_original_scale:
            out, out_cov = theta, cov_theta
        else:
            out = self.transform_scale(theta, direction="forward")
            jac = self._fitted_scale_jacobian(theta)
            out_cov = cov_theta * np.outer(jac, jac)

        s = _std_errors_from_cov(out_cov)
        x_ci = out + 1.96 * s * np.array([[-1], [1]])
        for i, name in enumerate(self._mean_param_names):
            label = name if transform_to_original_scale else self._fitted_scale_label(name, i)
            _print_if(f"{label} = {out[i]:.3e}", verbose)
            _print_if(f"95% confidence interval for {label}: [{x_ci[0, i]:.3e}, {x_ci[1, i]:.3e}]", verbose)

        return out, out_cov

    def fit_least_squares(self, simulation_inputs, reflectance_data, verbose=True):
        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        def fun(x):
            return self._sse(x, simulation_inputs, reflectance_data)

        _print_if("Fitting parameters with least squares ...", verbose)
        xL, xU = self._LS_SCALAR_BOUNDS
        res = minimize_scalar(fun, bounds=(xL, xU), method="Bounded")  # use bounded to prevent evaluation at values <=1
        _print_if("... done! \n estimated parameter is = " + str(res.x), verbose)
        return res.x, res.fun

    def fit_mle(self, simulation_inputs, reflectance_data, verbose=True, x0=None, transform_to_original_scale=False, save_file=None, **optim_kwargs):
        """
        Fits the soiling model parameters using maximum likelihood estimation (MLE).

        This function fits the soiling model parameters, `hrz0` and `sigma_dep`, using maximum likelihood estimation (MLE).
        It can initialize the optimization using least squares and 1D MLE, and provides the option to transform the parameter
        estimates back to the original scale.

        Args:
            simulation_inputs (dict): A dictionary containing the simulation inputs.
            reflectance_data (dict): A dictionary containing the reflectance data.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
            x0 (numpy.ndarray, optional): Initial guess for the model parameters. If not provided, the function will initialize using least squares and 1D MLE.
            transform_to_original_scale (bool, optional): Whether to transform the parameter estimates back to the original scale. Defaults to False.
            save_file (str, optional): Path to save the optimization results to a file.
            **optim_kwargs: Additional keyword arguments to pass to the optimization function.

        Returns:
            tuple: A tuple containing the estimated model parameters (`p_hat`) and their covariance matrix (`p_cov`).
        """

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        ref_dat = reflectance_data
        sim_in = simulation_inputs
        self._check_variance_components_identifiable(simulation_inputs, reflectance_data)

        # Start every common fraction at an even split, which is interior to [0, 1] and so
        # does not begin the search on a boundary.
        kappa0 = [0.5] * len(self.fitted_kappa_names)

        if np.all(x0 is None):  # intialize using least squares and 1D MLE
            _print_if("Getting initial deposition parameter guess via least squares", verbose)
            p0, sse = self.fit_least_squares(simulation_inputs, reflectance_data, verbose=False)

            _print_if("Getting initial sigma_dep guess via MLE (at least-squares value for deposition parameters)", verbose)

            def nloglike1D(y):
                return self._negative_log_likelihood([p0, y] + kappa0, sim_in, ref_dat)

            s0 = minimize_scalar(nloglike1D, bounds=(smb.tol, sse), method="Bounded")  # use bounded to prevent evaluation at values <=1
            x0 = np.array([p0, s0.x] + kappa0)
            _print_if("x0 = [" + ", ".join(f"{v}" for v in x0) + "]", verbose)

        # MLE. Transform to logs to ensure parameters are positive
        _print_if("Maximizing likelihood ...", verbose)
        y0 = self.transform_scale(x0, direction="forward")

        def nloglike(y):
            return self._negative_log_likelihood(self.transform_scale(y), sim_in, ref_dat)

        res = minimize(nloglike, y0, **optim_kwargs)
        y = res.x
        _print_if("  " + res.message, verbose)

        # One-time divergence check. transform_scale silences the transient
        # per-iteration exp overflow the optimizer produces while probing extreme
        # parameter values; here, at the converged optimum, we surface the genuine
        # case where a parameter overflowed to inf in original scale -- i.e. the
        # optimizer did not find a finite optimum (unidentifiable model / too
        # little data), which the silenced per-step warnings would otherwise mask.
        with np.errstate(over="ignore"):
            x_final = self.transform_scale(y)
        if not np.all(np.isfinite(np.asarray(x_final, dtype=float))):
            logger.warning(
                "Fit did not converge to finite parameters: at least one parameter "
                "overflowed to inf in original scale (transformed optimum = "
                f"{np.array2string(np.asarray(y), precision=3)}). The model is "
                "likely unidentifiable from this data."
            )

        _print_if("Estimating parameter covariance using numerical approximation of Hessian ... ", verbose)
        H_log = ndt.Hessian(nloglike)(y)  # Hessian is in the log transformed space

        if transform_to_original_scale:
            # Get standard errors using observed information
            x_hat, H = self.transform_scale(y, likelihood_hessian=H_log)
            x_cov = np.linalg.inv(H)

            p_hat = x_hat
            p_cov = x_cov

        else:
            y_hat = y
            try:
                y_cov = np.linalg.inv(H_log)  # Parameter covariance in the log space
            except np.linalg.LinAlgError:
                if np.linalg.det(H_log) == 0:
                    y_cov = np.linalg.pinv(H_log)  # Use pseudoinverse if determinant is zero
                else:
                    raise  # Re-raise the exception if it's not due to zero determinant

            # print estimates
            fmt = "log(log(hrz0)) = {0:.2e}, log(sigma_dep) = {1:.2e}"
            _print_if("... done! \n" + fmt.format(y_hat[0], y_hat[1]), verbose)

            # print confidence intervals
            s = _std_errors_from_cov(y_cov)
            y_ci = y_hat + 1.96 * s * np.array([[-1], [1]])
            fmt = "95% confidence interval for {0:s}: [{1:.2e}, {2:.2e}]"
            _print_if(fmt.format("log(log(hrz0))", y_ci[0, 0], y_ci[1, 0]), verbose)
            _print_if(fmt.format("log(sigma_dep)", y_ci[0, 1], y_ci[1, 1]), verbose)
            p_hat = y_hat
            p_cov = y_cov

        _print_if("... done!\n", verbose)
        return p_hat, p_cov

    def save_data(self, log_p_hat=None, log_p_hat_cov=None, training_simulation_data=None, training_reflectance_data=None):
        save_data = {"model": self, "type": None}
        if log_p_hat is not None:
            save_data["transformed_parameters"] = log_p_hat
        if log_p_hat_cov is not None:
            save_data["transformed_parameter_covariance"] = log_p_hat_cov
        if training_simulation_data is not None:
            save_data["simulation_data"] = training_simulation_data
        if training_reflectance_data is not None:
            save_data["reflectance_data"] = training_reflectance_data

        return save_data

    def plot_soiling_factor(
        self,
        simulation_inputs,
        posterior_predictive_distribution_samples=None,
        reflectance_data=None,
        figsize=None,
        reflectance_std="measurements",
        save_path=None,
        fig_title=None,
        return_handles=False,
        repeat_y_labels=True,
        orientation_strings=None,
        names_mir_train=None,
    ):
        """
        Plots the soiling factor over time for a set of simulation inputs and reflectance data.

        Parameters:
            simulation_inputs (dict): A dictionary containing the simulation input data, including time, dust concentration, wind speed, and dust type for each experiment.
            posterior_predictive_distribution_samples (dict, optional): A dictionary containing the posterior predictive distribution samples for each experiment.
            reflectance_data (dict, optional): A dictionary containing the reflectance data for each experiment, including the average reflectance, reflectance standard deviation, and tilt angles.
            figsize (tuple, optional): The size of the figure to be plotted.
            reflectance_std (str, optional): Specifies whether to use the "measurements" or "mean" standard deviation for the reflectance data. Defaults to "measurements".
            save_path (str, optional): The file path to save the plot.
            fig_title (str, optional): The title of the figure.
            return_handles (bool, optional): If True, returns the figure, axis, and prediction data. Otherwise, only returns the prediction data.
            repeat_y_labels (bool, optional): If True, repeats the y-axis labels for each experiment.
            orientation_strings (list, optional): A list of orientation strings for each mirror in each experiment.

        Returns:
            tuple or dict: If return_handles is True, returns the figure, axis, and prediction data. Otherwise, returns only the prediction data.
        """

        if reflectance_data is not None:
            self.predict_soiling_factor(simulation_inputs, reflectance_data=reflectance_data)
        else:
            self.predict_soiling_factor(simulation_inputs)

        sim_in = simulation_inputs
        samples = posterior_predictive_distribution_samples
        files = list(sim_in.time.keys())
        N_mirrors = np.array([self.helios.tilt[f].shape[0] for f in files])
        if np.all(N_mirrors == N_mirrors[0]):
            N_mirrors = N_mirrors[0]
        else:
            raise ValueError("Number of mirrors must be the same for each experiment to use this function.")

        if reflectance_data is not None:
            # check to ensure that reflectance_data and simulation_input keys correspond to the same files
            _check_keys(sim_in, reflectance_data)

        N_experiments = sim_in.N_simulations
        ws_max = max([max(sim_in.wind_speed[f]) for f in files])  # max wind speed for setting y-axes
        mean_predictions = {f: np.array([]) for f in files}
        CI_upper_predictions = {f: np.array([]) for f in files}
        CI_lower_predictions = {f: np.array([]) for f in files}

        fig, ax = plt.subplots(N_mirrors + 1, N_experiments, figsize=figsize, sharex="col")
        ax_wind = []
        for ii in range(N_experiments):
            f = files[ii]
            N_times = self.helios.tilt[f].shape[1]
            mean_predictions[f] = np.zeros(shape=(N_mirrors, N_times))
            CI_upper_predictions[f] = np.zeros(shape=(N_mirrors, N_times))
            CI_lower_predictions[f] = np.zeros(shape=(N_mirrors, N_times))

            dust_conc = sim_in.dust_concentration[f]
            ws = sim_in.wind_speed[f]
            dust_type = sim_in.dust_type[f]
            ts = sim_in.time[f]
            if reflectance_data is not None:
                tr = reflectance_data.times[f]
                # Per-mirror nominal reflectance anchor (reference-mirror-derived when
                # reflectance_data carries one, else the model's fixed constant
                # broadcast to every mirror) -- indexed like reflectance_data.average/
                # self.helios.soiling_factor (jj below).
                r0_by_mirror = np.broadcast_to(
                    _nominal_reflectance_anchor(reflectance_data, f, self.helios.nominal_reflectance), reflectance_data.rho0[f].shape
                )

            for jj in range(0, N_mirrors):
                if jj == 0:
                    tilt_str = r"Experiment " + str(ii) + r", tilt = ${0:.0f}^{{\circ}}$"
                else:
                    tilt_str = r"tilt = ${0:.0f}^{{\circ}}$"  # tilt only

                if orientation_strings is not None:
                    tilt_str += ", Orientation: " + orientation_strings[ii][jj]
                    # u_idx = names_mir_train[0].find('_')
                    # tilt_str += ", Orientation: " + names_mir_train[0][:u_idx].replace('O','')

                # get the axis handles
                if N_experiments == 1:
                    a = ax[jj]  # experiment ii, mirror jj plot
                    a2 = ax[-1]  # weather plot
                    am = ax[0]  # plot to put legend on
                else:
                    a = ax[jj, ii]
                    a2 = ax[-1, ii]
                    am = ax[0, 0]

                if reflectance_data is not None:  # plot predictions and reflectance data
                    m = reflectance_data.average[f][:, jj]
                    r0 = r0_by_mirror[jj]

                    if reflectance_std == "measurements":
                        s = reflectance_data.sigma[f][:, jj]
                    elif reflectance_std == "mean":
                        s = reflectance_data.sigma_of_the_mean[f][:, jj]
                    else:
                        raise ValueError("reflectance_std=" + reflectance_std + ' not recognized. Must be either "measurements" or "mean" ')

                    # measurement plots
                    error_two_sigma = 1.96 * s
                    a.errorbar(tr, m, yerr=error_two_sigma, label="Measurement mean")

                    # mean prediction plot
                    if samples is None:  # use soiling factor in helios
                        ym = r0 * self.helios.soiling_factor[f][jj, :]
                        a.plot(sim_in.time[f], ym, label="Reflectance Prediction", color="black")
                    else:
                        y = r0 * samples[f][jj, :, :]
                        ym = y.mean(axis=1)
                        a.plot(sim_in.time[f], ym, label="Reflectance Prediction (Bayesian)", color="red")

                    tilt = reflectance_data.tilts[f][jj]
                    if all(tilt == tilt[0]):
                        a.set_title(tilt_str.format(tilt[0]), fontsize=20)
                    else:
                        a.set_title(tilt_str.format(tilt.mean()) + " (average)", fontsize=20)
                else:  # predictions are from clean
                    r0 = self.helios.nominal_reflectance
                    if samples is None:
                        ym = r0 * self.helios.soiling_factor[f][jj, :]
                        a.plot(sim_in.time[f], ym, label="Prediction", color="black")
                    else:
                        y = samples[f][jj, :, :]
                        ym = y.mean(y, axis=1)
                        a.plot(sim_in.time[f], ym, label="Prediction (Bayesian)", color="red")

                    tilt = self.helios.tilt[f][jj, :]
                    if all(tilt == tilt[0]):
                        a.set_title(tilt_str.format(tilt[0]), fontsize=20)
                    else:
                        a.set_title(tilt_str.format(tilt.mean()) + " (average)", fontsize=20)

                if (
                    samples is None and len(self.helios.soiling_factor_prediction_variance) > 0
                ):  # add +/- 2 sigma limits to the predictions, if sigma_dep is set
                    # var_predict = self.helios.delta_soiled_area_variance[f][jj,:]
                    var_predict = self.helios.soiling_factor_prediction_variance[f][jj, :]
                    sigma_predict = r0 * np.sqrt(var_predict)
                    Lp = ym - 1.96 * sigma_predict
                    Up = ym + 1.96 * sigma_predict
                    a.fill_between(ts, Lp, Up, color="black", alpha=0.1, label=r"$\pm 2\sigma$ CI")
                elif samples is not None:  # use percentiles of posterior predictive samples for confidence intervals
                    Lp = np.percentile(y, 2.5, axis=1)
                    Up = np.percentile(y, 97.5, axis=1)
                    a.fill_between(ts, Lp, Up, color="red", alpha=0.1, label=r"95% Bayesian CI")

                a.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # sets x ticks to day interval

                if reflectance_data is not None:  # reflectance is computed at reflectometer incidence angle
                    if repeat_y_labels or (ii == 0):
                        ang = reflectance_data.reflectometer_incidence_angle[f]
                        s = a.set_ylabel(r"$\rho(t)$ at " + str(ang) + r"$^{\circ}$")
                    else:
                        a.set_yticklabels([])

                else:  # reflectance is computed at heliostat incidence angle. Put average incidence angle on axis label
                    if repeat_y_labels or (ii > 0):
                        ang = np.mean(self.helios.incidence_angle[f])
                        s = a.set_ylabel(r"soiling factor at " + str(ang) + r"$^{{\circ}}$ \n (average)")
                    else:
                        a.set_yticklabels([])

                # set mean and CIs for output
                try:
                    mean_predictions[f][jj, :] = ym
                    CI_upper_predictions[f][jj, :] = Up
                    CI_lower_predictions[f][jj, :] = Lp
                except Exception:
                    mean_predictions[f][jj, :] = ym

            am.legend(fontsize=16)
            label_str = dust_type + r" (mean = {0:.2f} $\mu g$/$m^3$)"
            a2.plot(ts, dust_conc, label=label_str.format(dust_conc.mean()), color="blue")
            a2.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # sets x ticks to day interval
            myFmt = mdates.DateFormatter("%d-%m-%Y")
            a2.xaxis.set_major_formatter(myFmt)
            a2.tick_params(axis="y", labelcolor="blue")

            a2a = a2.twinx()
            a2a.plot(ts, ws, color="green", label="Wind Speed (mean = {0:.2f} m/s)".format(ws.mean()))
            ax_wind.append(a2a)
            a2a.tick_params(axis="y", labelcolor="green")
            a2a.set_ylim((0, ws_max))

            if ii == 0:  # ylabel for TSP on leftmost plot only
                fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
                a2.set_ylabel(fs.format(dust_type), color="blue")
            else:
                a2.set_yticklabels([])

            if ii == N_experiments - 1:  # ylabel for wind speed on rightmost plot only
                a2a.set_ylabel("Wind Speed (m/s)", color="green")
            else:
                a2a.set_yticklabels([])

            a2.set_title(label_str.format(dust_conc.mean()) + ", \n Wind Speed (mean = {0:.2f} m/s)".format(ws.mean()), fontsize=20)

        if N_experiments > 1:
            # share y axes for all reflectance measurments
            ymax = max([x.get_ylim()[1] for x in ax[0:-1, :].flatten()])
            ymin = min([x.get_ylim()[0] for x in ax[0:-1, :].flatten()])
            for a in ax[0:-1, :].flatten():
                a.set_ylim(ymin, 1)

            # share y axes for weather variables of the same type
            ymax_dust = max([x.get_ylim()[1] for x in ax[-1, :]])
            ymax_wind = max([x.get_ylim()[1] for x in ax_wind])
            for a in ax[-1, :]:
                a.set_ylim(0, 1.1 * ymax_dust)
            for a in ax_wind:
                a.set_ylim(0, 1.1 * ymax_wind)
        else:
            ymax = max([x.get_ylim()[1] for x in ax[0:-1].flatten()])
            ymin = min([x.get_ylim()[0] for x in ax[0:-1].flatten()])
            for a in ax[0:-1]:
                a.set_ylim(ymin, ymax)

        fig.autofmt_xdate()
        fig.suptitle(fig_title, fontsize=16)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)

        if return_handles:
            return fig, ax, mean_predictions, CI_lower_predictions, CI_upper_predictions
        else:
            return mean_predictions, CI_lower_predictions, CI_upper_predictions


class AffineMeanLeastSquares:
    """Mixin giving fit_ls to a model whose predicted reflectance is AFFINE in its mean
    parameters -- true of the whole constant-mean family, where delta_soiled_area is
    sum_k theta_k * basis_k and compute_soiling_factor only accumulates it and adds a
    theta-independent initial condition.

    That makes least squares a linear problem: one bounded solve, no starting point, no
    local optima. It does not hold for a model whose parameter enters through the
    deposition physics (SemiPhysical), which fits its mean by nonlinear least squares
    instead.

    Mix in alongside CommonFittingMethods, which supplies everything else the fit needs.
    """

    def mean_design_matrix(self, simulation_inputs, reflectance_data, residuals="level"):
        """The linear least-squares problem `X @ theta ~= y` for this model's mean
        parameters theta (self._mean_param_names, in order).

        Because the prediction is affine in theta, the whole forward model is

            offset + sum_k theta_k * (prediction at theta = e_k  -  offset),

        so the design matrix is obtained EXACTLY -- no finite differencing, and no second
        copy of the reflectance algebra to drift out of sync -- by evaluating the existing
        forward model once at theta = 0 and once per mean parameter at a unit basis vector.

        Args:
            residuals: "level" or "increment" (see CommonFittingMethods._ls_stack).

        Returns:
            (X, y): X has shape (n_observations, n_mean_parameters). Rows carrying a
            non-finite measurement (a gap in reflectance_data.average) are dropped, so a
            missing measurement costs that row rather than the whole fit.
        """
        _check_keys(simulation_inputs, reflectance_data)
        files = list(reflectance_data.average.keys())
        mean_names, sigma_names = list(self._mean_param_names), list(self._sigma_param_names)

        saved = {name: getattr(self, name) for name in mean_names + sigma_names}
        try:
            # Sigmas off while probing: the design matrix is a mean-only object, and leaving
            # them set would make every probe also build the (discarded) variance.
            for name in sigma_names:
                setattr(self, name, None)
            for name in mean_names:
                setattr(self, name, 0.0)
            offset = self._predicted_reflectance(simulation_inputs, reflectance_data)

            columns = []
            for name in mean_names:
                setattr(self, name, 1.0)
                probe = self._predicted_reflectance(simulation_inputs, reflectance_data)
                setattr(self, name, 0.0)
                columns.append(self._ls_stack({f: probe[f] - offset[f] for f in files}, files, residuals))
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

        X = np.column_stack(columns)
        y = self._ls_stack({f: np.asarray(reflectance_data.average[f], dtype=float) - offset[f] for f in files}, files, residuals)

        keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if not keep.any():
            raise ValueError("No usable observations for the least-squares fit: every residual row is non-finite.")
        return X[keep, :], y[keep]

    def fit_ls(self, simulation_inputs, reflectance_data, verbose=True, residuals="level", transform_to_original_scale=False):
        """Least-squares counterpart of fit_mle: fits the MEAN parameters by one bounded
        linear solve on mean_design_matrix.

        Parameters the model carries in logs are bounded below at 0 (a negative value has no
        transformed representation); the rest are left free, since a wind coefficient may
        legitimately be negative. The estimate minimizes the sum of squared reflectance
        errors, so every observation is weighted equally rather than down-weighted where the
        model calls itself noisy.

        No noise parameters are fitted -- see CommonFittingMethods' least-squares section for
        what that means for the caller.

        Returns:
            (x_hat, x_hat_cov) over self.mean_parameter_names, on the same scale convention
            as fit_mle. The covariance is the ordinary least-squares s^2 (X'X)^-1.
        """
        _check_keys(simulation_inputs, reflectance_data)

        X, y = self.mean_design_matrix(simulation_inputs, reflectance_data, residuals=residuals)
        n_obs, n_mean = X.shape
        is_log_mean = np.asarray(self._ls_lower_bounded(), dtype=bool)

        # Column scaling: a deposition coefficient's basis (accumulated cos(tilt) x dust
        # loading) and a wind coefficient's (that times wind speed and a geometry factor)
        # differ by orders of magnitude, and a normal-equation solve inherits the square of
        # that spread. The bounds scale with the columns, and being 0/+-inf are unchanged.
        scale = np.linalg.norm(X, axis=0)
        # An all-zero column is a mechanism this data cannot see at all (e.g.
        # impaction_retention when every mirror is horizontal): unidentifiable, not a
        # division by zero. Scaling it by 1 leaves the solver to return 0 for it.
        scale[scale == 0.0] = 1.0
        X_scaled = X / scale

        _print_if(f"Fitting {n_mean} mean parameter(s) by linear least squares on {n_obs} observations ...", verbose)
        ls = lsq_linear(X_scaled, y, bounds=(np.where(is_log_mean, 0.0, -np.inf), np.inf))
        theta = ls.x / scale
        # A log-transformed parameter driven onto its lower bound would be log(0) = -inf on
        # the fitted scale, and an infinite entry in the returned estimate.
        theta = np.where(is_log_mean & (theta <= 0.0), smb.tol, theta)

        residual_vector = X @ theta - y
        sse = float(residual_vector @ residual_vector)
        s2 = sse / max(n_obs - n_mean, 1)
        # OLS covariance, undone back onto the parameters' own scale. pinv, not inv: a
        # mechanism the data cannot separate leaves X'X singular, which should surface as a
        # huge (or NaN once square-rooted) standard error, not a LinAlgError mid-fold.
        inv_scale = np.diag(1.0 / scale)
        cov_theta = s2 * (inv_scale @ np.linalg.pinv(X_scaled.T @ X_scaled) @ inv_scale)
        _print_if(f"... done! SSE = {sse:.4e}", verbose)

        return self._finish_least_squares(theta, cov_theta, transform_to_original_scale, verbose)


class SemiPhysical(smb.PhysicalBase, CommonFittingMethods):
    # hrz0 enters through the deposition-velocity physics, so the prediction is NOT affine in
    # it and AffineMeanLeastSquares does not apply; fit_ls below is a nonlinear fit instead.
    _mean_param_names = ("hrz0",)
    _sigma_param_names = ("sigma_dep",)
    # hrz0 is a roughness ratio greater than one, so it is carried as log(log(hrz0)).
    _mean_transform_codes = (TRANSFORM_LOGLOG,)

    def __init__(self, file_params, verbose=True):
        table = pd.read_excel(file_params, index_col="Parameter")
        super().__init__()
        self.import_site_data_and_constants(file_params, verbose=verbose)
        self.helios.hamaker = float(table.loc["hamaker_glass"].Value)
        self.helios.poisson = float(table.loc["poisson_glass"].Value)
        self.helios.youngs_modulus = float(table.loc["youngs_modulus_glass"].Value)
        self.helios.nominal_reflectance = float(table.loc["nominal_reflectance"].Value)
        if not (isinstance(self.helios.stow_tilt, float)) and not (isinstance(self.helios.stow_tilt, int)):
            self.helios.stow_tilt = None
        self.verbose = verbose

    def helios_angles(
        self,
        simulation_inputs: smb.SimulationInputs,
        reflectance_data: smb.ReflectanceMeasurements,
        verbose: bool = True,
        second_surface: bool = True,
    ) -> None:
        sim_in = simulation_inputs
        ref_dat = reflectance_data
        files = list(sim_in.time.keys())
        N_experiments = len(files)

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(sim_in, ref_dat)

        _print_if("Setting tilts for " + str(N_experiments) + " experiments", verbose)
        helios = self.helios
        helios.tilt = {f: None for f in files}  # clear the existing tilts
        helios.acceptance_angles = {f: None for f in files}  # clear the existing acceptance angles
        for ii in range(N_experiments):
            f = files[ii]
            # start_idx = ref_dat.prediction_indices[f][0]        # Define the start index
            # end_idx = ref_dat.prediction_indices[f][-1]         # Define the start last index
            # tilts = ref_dat.tilts[f][:, start_idx:end_idx + 1]  # Extract the subset of tilts (end_idx + 1 include the last index)
            tilts = ref_dat.tilts[
                f
            ]  # THIS CANNOT MANAGE THE TRANSFORMATION OF SIM_IN WITH DAILY AVERAGE WHEN CHANGING START TIME (IF APPLIED AT THE BEGINNING)
            N_times = len(sim_in.time[f])
            N_helios = tilts.shape[0]
            self.helios.acceptance_angles[f] = [ref_dat.reflectometer_acceptance_angle[ii]] * N_helios
            self.helios.extinction_weighting[f] = []  # reset extinction weighting since heliostats are "new" - WHY?? THIS DEPENDS ONLY ON DUST! NOT?

            helios.tilt[f] = np.zeros((0, N_times))
            for jj in range(N_helios):
                row_mask = np.ones((1, N_times))
                helios.tilt[f] = np.vstack((helios.tilt[f], tilts[jj] * row_mask))

            helios.elevation[f] = 90 - helios.tilt[f]
            helios.incidence_angle[f] = reflectance_data.reflectometer_incidence_angle[f]

            if not second_surface:
                helios.inc_ref_factor[f] = (1 + np.sin(rad(helios.incidence_angle[f]))) / np.cos(rad(helios.incidence_angle[f]))  # first surface
                helios.aoi_model = "first_surface"
                _print_if("First surface model", verbose)
            elif second_surface:
                helios.inc_ref_factor[f] = 2 / np.cos(rad(helios.incidence_angle[f]))  # second surface model
                helios.aoi_model = "second_surface"
                _print_if("Second surface model", verbose)
            else:
                _print_if("Choose either first or second surface model", verbose)

        self.helios = helios

    def predict_soiling_factor(self, simulation_inputs, reflectance_data=None, hrz0=None, sigma_dep=None, verbose=True) -> None:
        # Uses simulation inputs and model parameters to predict the soiling
        # factor and the prediction variance (stored in
        # helios.soiling_factor and helios.soiling_factor_prediction_variance,
        # respectively).

        self.deposition_flux(simulation_inputs, hrz0=hrz0, verbose=verbose)
        self.adhesion_removal(simulation_inputs, verbose=verbose)
        self.calculate_delta_soiled_area(simulation_inputs, sigma_dep=sigma_dep, verbose=verbose)
        self.compute_soiling_factor(reflectance_data=reflectance_data)

        # prediction variance
        if self.sigma_dep is not None:
            for f in self.helios.soiling_factor.keys():
                inc_factor = self.helios.inc_ref_factor[f]
                dsav = self.helios.delta_soiled_area_variance[f]
                self.helios.soiling_factor_prediction_variance[f] = inc_factor**2 * np.cumsum(dsav, axis=1)
        else:
            self.helios.soiling_factor_prediction_variance = {}

    def fit_mle(self, simulation_inputs, reflectance_data, verbose=True, x0=None, transform_to_original_scale=False, **optim_kwargs):
        p_hat, p_cov = super().fit_mle(
            simulation_inputs, reflectance_data, verbose=True, x0=x0, transform_to_original_scale=transform_to_original_scale, **optim_kwargs
        )

        # print estimates and confidence intervals
        fmtCI = "95% confidence interval for {0:s}: [{1:.2e}, {2:.2e}]"
        if transform_to_original_scale:
            fmtE = "hrz0 = {0:.2e}, sigma_dep = {1:.2e}"
            _print_if(fmtE.format(p_hat[0], p_hat[1]), verbose)

            # print confidence intervals
            s = _std_errors_from_cov(p_cov)
            x_ci = p_hat + 1.96 * s * np.array([[-1], [1]])
            _print_if(fmtCI.format("hrz0", x_ci[0, 0], x_ci[1, 0]), verbose)
            _print_if(fmtCI.format("sigma_dep", x_ci[0, 1], x_ci[1, 1]), verbose)
        else:
            fmtE = "log(log(hrz0)) = {0:.2e}, sigma_dep = {1:.2e}"
            _print_if(fmtE.format(p_hat[0], p_hat[1]), verbose)
            s = _std_errors_from_cov(p_cov)
            y_ci = p_hat + 1.96 * s * np.array([[-1], [1]])
            _print_if(fmtCI.format("log(log(hrz0))", y_ci[0, 0], y_ci[1, 0]), verbose)
            _print_if(fmtCI.format("log(sigma_dep)", y_ci[0, 1], y_ci[1, 1]), verbose)

        return p_hat, p_cov

    def fit_ls(self, simulation_inputs, reflectance_data, verbose=True, transform_to_original_scale=False):
        """Least-squares counterpart of fit_mle: fits hrz0 alone, by NONLINEAR least squares.

        The predicted reflectance is not affine in hrz0 -- it enters through
        deposition_velocity and the wind profile -- so there is no design matrix to solve in
        one shot, as there is for the constant-mean family (AffineMeanLeastSquares). With a
        single parameter, though, the bounded scalar search over _sse that fit_least_squares
        already performs is the whole fit; what is added here is the covariance and the
        (estimate, covariance) contract fit_mle follows.

        Fits the reflectance level, the objective _sse defines. No noise parameter is fitted
        -- see CommonFittingMethods' least-squares section for what that means for callers.

        Returns:
            (x_hat, x_hat_cov) over ["hrz0"], on the same scale convention as fit_mle.
        """
        _check_keys(simulation_inputs, reflectance_data)

        _print_if("Fitting hrz0 by nonlinear least squares (bounded scalar search) ...", verbose)
        hrz0, _sse_at_optimum = self.fit_least_squares(simulation_inputs, reflectance_data, verbose=False)
        self._warn_if_at_scalar_bound("hrz0", hrz0)

        # Covariance from the local linearization, s^2 (J'J)^-1 with J = d(prediction)/d(hrz0)
        # taken by a central difference on the same forward model the objective uses. hrz0
        # must stay above 1 (the model represents it as log(log(hrz0))), so the backward step
        # is clipped to stay inside that domain when the fit lands near its lower bound.
        files = list(reflectance_data.average.keys())
        step = max(min(1e-4 * hrz0, 0.5 * (hrz0 - 1.0)), 1e-9)

        def level_prediction(value):
            self.update_model_parameters(value)  # scalar: sets hrz0 and clears sigma_dep
            return self._ls_stack(self._predicted_reflectance(simulation_inputs, reflectance_data), files, "level")

        jac = (level_prediction(hrz0 + step) - level_prediction(hrz0 - step)) / (2.0 * step)
        predicted = level_prediction(hrz0)
        measured = self._ls_stack({f: np.asarray(reflectance_data.average[f], dtype=float) for f in files}, files, "level")

        keep = np.isfinite(predicted) & np.isfinite(measured) & np.isfinite(jac)
        if not keep.any():
            raise ValueError("No usable observations for the least-squares fit: every residual row is non-finite.")

        residual = predicted[keep] - measured[keep]
        sse = float(residual @ residual)
        s2 = sse / max(int(keep.sum()) - 1, 1)
        jtj = float(jac[keep] @ jac[keep])
        # jtj == 0 means the prediction does not respond to hrz0 at all here: unidentifiable,
        # which an infinite variance (NaN standard error) reports rather than a divide error.
        cov_theta = np.array([[s2 / jtj if jtj > 0.0 else np.inf]])
        _print_if(f"... done! SSE = {sse:.4e}", verbose)

        return self._finish_least_squares(np.array([hrz0]), cov_theta, transform_to_original_scale, verbose)

    def update_model_parameters(self, x):
        """
        Updates the model parameters `hrz0` and `sigma_dep` based on the input `x`.

        If `x` is a list or NumPy array, the first element is assigned to `hrz0` and the second element (if present) is assigned to `sigma_dep`.

        If `x` is a single value, it is assigned to `hrz0` and `sigma_dep` is set to `None`.
        """
        if isinstance(x, list) or isinstance(x, np.ndarray):
            core, kappas = self._split_parameters(x)
            self.hrz0 = core[0]
            if len(core) > 1:
                self.sigma_dep = core[1]
            self._update_kappa_parameters(kappas)
        else:
            self.hrz0 = x
            self.sigma_dep = None

    def save(self, file_name, log_p_hat=None, log_p_hat_cov=None, training_simulation_data=None, training_reflectance_data=None):
        """
        Saves the model and associated data to a file using pickle.

        Args:
            file_name (str): The file path to save the model and data to.
            log_p_hat (numpy.ndarray, optional): The transformed model parameters.
            log_p_hat_cov (numpy.ndarray, optional): The covariance of the transformed model parameters.
            training_simulation_data (dict, optional): The simulation data used for training the model.
            training_reflectance_data (dict, optional): The reflectance data used for training the model.

        Returns:
            None
        """
        with open(file_name, "wb") as f:
            save_data = {"model": self, "type": "semi-physical"}
            if log_p_hat is not None:
                save_data["transformed_parameters"] = log_p_hat
            if log_p_hat_cov is not None:
                save_data["transformed_parameter_covariance"] = log_p_hat_cov
            if training_simulation_data is not None:
                save_data["simulation_data"] = training_simulation_data
            if training_reflectance_data is not None:
                save_data["reflectance_data"] = training_reflectance_data

            pickle.dump(save_data, f)


class ConstantMeanDeposition(smb.ConstantMeanBase, AffineMeanLeastSquares, CommonFittingMethods):
    simulation_inputs: smb.SimulationInputs
    reflectance_data: smb.ReflectanceMeasurements

    # delta_soiled_area is alpha * cos(tilt) * mu_tilde -- linear in the one mean parameter --
    # so AffineMeanLeastSquares' one-shot linear fit_ls applies directly.
    _mean_param_names = ("mu_tilde",)
    _sigma_param_names = ("sigma_dep",)
    _mean_transform_codes = (TRANSFORM_LOG,)

    def __init__(self, file_params, verbose=True):
        super().__init__()
        self.import_site_data_and_constants(file_params, verbose=verbose)
        table = pd.read_excel(file_params, index_col="Parameter")
        self.helios.nominal_reflectance = float(table.loc["nominal_reflectance"].Value)

    def helios_angles(
        self, simulation_inputs: smb.SimulationInputs, reflectance_data: smb.ReflectanceMeasurements, verbose=True, second_surface=True
    ):
        sim_in = simulation_inputs
        ref_dat = reflectance_data
        files = list(sim_in.time.keys())
        N_experiments = len(files)

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(sim_in, ref_dat)

        _print_if("Setting tilts for " + str(N_experiments) + " experiments", verbose)
        helios = self.helios
        helios.tilt = {f: None for f in files}  # clear the existing tilts
        helios.acceptance_angles = {f: None for f in files}  # clear the existing tilts
        for ii in range(N_experiments):
            f = files[ii]
            # start_idx = ref_dat.prediction_indices[f][0]        # Define the start index
            # end_idx = ref_dat.prediction_indices[f][-1]         # Define the start last index
            # tilts = ref_dat.tilts[f][:, start_idx:end_idx + 1]  # Extract the subset of tilts (end_idx + 1 include the last index)
            tilts = ref_dat.tilts[f]  # THIS COULD NOT MANAGE THE TRANSFORMATION OF SIM_IN WITH DAILY AVERAGE WHEN CHANGING START TIME
            N_times = len(sim_in.time[f])
            N_helios = tilts.shape[0]
            self.helios.acceptance_angles[f] = [ref_dat.reflectometer_acceptance_angle[ii]] * N_helios
            self.helios.extinction_weighting[f] = []  # reset extinction weighting since heliostats are "new"

            helios.tilt[f] = np.zeros((0, N_times))
            for jj in range(N_helios):
                row_mask = np.ones((1, N_times))
                helios.tilt[f] = np.vstack((helios.tilt[f], tilts[jj] * row_mask))

            helios.elevation[f] = 90 - helios.tilt[f]
            helios.incidence_angle[f] = reflectance_data.reflectometer_incidence_angle[f]

            if not second_surface:
                helios.inc_ref_factor[f] = (1 + np.sin(rad(helios.incidence_angle[f]))) / np.cos(rad(helios.incidence_angle[f]))  # first surface
                helios.aoi_model = "first_surface"
                _print_if("First surface model", verbose)
            elif second_surface:
                helios.inc_ref_factor[f] = 2 / np.cos(rad(helios.incidence_angle[f]))  # second surface model
                helios.aoi_model = "second_surface"
                _print_if("Second surface model", verbose)
            else:
                _print_if("Choose either first or second surface model", verbose)

        self.helios = helios

    def predict_soiling_factor(self, simulation_inputs: smb.SimulationInputs, reflectance_data=None, mu_tilde=None, sigma_dep=None, verbose=True):
        sim_in = simulation_inputs
        self.calculate_delta_soiled_area(sim_in, mu_tilde=mu_tilde, sigma_dep=sigma_dep, verbose=verbose)
        self.compute_soiling_factor(reflectance_data=reflectance_data)

        # prediction variance
        if self.sigma_dep is not None:
            for f in self.helios.soiling_factor.keys():
                inc_factor = self.helios.inc_ref_factor[f]
                dsav = self.helios.delta_soiled_area_variance[f]
                self.helios.soiling_factor_prediction_variance[f] = inc_factor**2 * np.cumsum(dsav, axis=1)
        else:
            self.helios.soiling_factor_prediction_variance = {}

    def fit_map(self, simulation_inputs, reflectance_data, priors, verbose=True, x0=None, transform_to_original_scale=False, save_file=None):
        _print_if("Getting MAP estimates ... ", verbose)
        y, y_cov = super().fit_map(simulation_inputs, reflectance_data, priors, verbose=False, x0=x0, transform_to_original_scale=False)

        _print_if("========== MAP Estimates ======== ", verbose)
        if transform_to_original_scale:
            x_hat, x_hat_cov = self.transform_scale(y, y_cov)

            # print estimates
            fmt = "mu_tilde = {0:.2e}, sigma_dep = {1:.2e}"
            _print_if(fmt.format(x_hat[0], x_hat[1]), verbose)

            # print confidence intervals
            s = _std_errors_from_cov(x_hat_cov)
            x_ci = x_hat + 1.96 * s * np.array([[-1], [1]])
            fmt = "95% confidence interval for {0:s}: [{1:.2e}, {2:.2e}]"
            _print_if(fmt.format("mu_tilde", x_ci[0, 0], x_ci[1, 0]), verbose)
            _print_if(fmt.format("sigma_dep", x_ci[0, 1], x_ci[1, 1]), verbose)

        else:
            x_hat = y
            x_hat_cov = y_cov

            # print estimates
            fmt = "log(mu_tilde) = {0:.2e}, log(sigma_dep) = {1:.2e} "
            _print_if(fmt.format(x_hat[0], x_hat[1]), verbose)

            # print confidence intervals
            s = _std_errors_from_cov(x_hat_cov)
            x_ci = x_hat + 1.96 * s * np.array([[-1], [1]])
            fmt = "95% confidence interval for {0:s}: [{1:.2e}, {2:.2e}]"
            _print_if(fmt.format("log(mu_tilde)", x_ci[0, 0], x_ci[1, 0]), verbose)
            _print_if(fmt.format("log(sigma_dep)", x_ci[0, 1], x_ci[1, 1]), verbose)

        return x_hat, x_hat_cov

    def fit_mle(self, simulation_inputs, reflectance_data, verbose=True, x0=None, transform_to_original_scale=False, save_file=None):
        _print_if("Getting MLE estimates ... ", verbose)
        y, y_cov = super().fit_mle(simulation_inputs, reflectance_data, verbose=False, x0=x0, transform_to_original_scale=False)

        _print_if("========== MLE Estimates ======== ", verbose)
        if transform_to_original_scale:
            x_hat = self.transform_scale(y)
            # Transform the covariance directly rather than inverting it to a Hessian,
            # transforming that, and inverting again.
            jacobian = self.natural_scale_jacobian(y)
            x_hat_cov = (jacobian[:, None] * y_cov) * jacobian[None, :]

            # print estimates
            fmt = "mu_tilde = {0:.2e}, sigma_dep = {1:.2e}"
            _print_if(fmt.format(x_hat[0], x_hat[1]), verbose)

            # print confidence intervals
            s = _std_errors_from_cov(x_hat_cov)
            x_ci = x_hat + 1.96 * s * np.array([[-1], [1]])
            fmt = "95% confidence interval for {0:s}: [{1:.2e}, {2:.2e}]"
            _print_if(fmt.format("mu_tilde", x_ci[0, 0], x_ci[1, 0]), verbose)
            _print_if(fmt.format("sigma_dep", x_ci[0, 1], x_ci[1, 1]), verbose)

        else:
            x_hat = y
            x_hat_cov = y_cov

            # print estimates
            fmt = "log(mu_tilde) = {0:.2e}, log(sigma_dep) = {1:.2e} "
            _print_if(fmt.format(x_hat[0], x_hat[1]), verbose)

            # print confidence intervals
            s = _std_errors_from_cov(x_hat_cov)
            x_ci = x_hat + 1.96 * s * np.array([[-1], [1]])
            fmt = "95% confidence interval for {0:s}: [{1:.2e}, {2:.2e}]"
            _print_if(fmt.format("log(mu_tilde)", x_ci[0, 0], x_ci[1, 0]), verbose)
            _print_if(fmt.format("log(sigma_dep)", x_ci[0, 1], x_ci[1, 1]), verbose)

        return x_hat, x_hat_cov

    def update_model_parameters(self, x):
        if isinstance(x, list) or isinstance(x, np.ndarray):
            core, kappas = self._split_parameters(x)
            self.mu_tilde = core[0]
            if len(core) > 1:
                self.sigma_dep = core[1]
            self._update_kappa_parameters(kappas)
        else:
            self.mu_tilde = x

    def save(self, file_name, log_p_hat=None, log_p_hat_cov=None, training_simulation_data=None, training_reflectance_data=None):
        """
        Saves the soiling model and associated data to a file.

        Args:
            file_name (str): The name of the file to save the data to.
            log_p_hat (numpy.ndarray, optional): The transformed model parameters.
            log_p_hat_cov (numpy.ndarray, optional): The covariance of the transformed model parameters.
            training_simulation_data (object, optional): The simulation data used for training the model.
            training_reflectance_data (object, optional): The reflectance data used for training the model.
        """
        with open(file_name, "wb") as f:
            save_data = {"model": self, "type": "constant-mean"}
            if log_p_hat is not None:
                save_data["transformed_parameters"] = log_p_hat
            if log_p_hat_cov is not None:
                save_data["transformed_parameter_covariance"] = log_p_hat_cov
            if training_simulation_data is not None:
                save_data["simulation_data"] = training_simulation_data
            if training_reflectance_data is not None:
                save_data["reflectance_data"] = training_reflectance_data

            pickle.dump(save_data, f)
