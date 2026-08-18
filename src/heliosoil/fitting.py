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
import numdifftools as ndt
import pickle


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

        Each model transforms its parameters differently (log, log-log, or not at all), and
        the delta method needs that derivative to carry a least-squares covariance onto the
        fitted scale. Declared next to each class's own transform_scale so the two cannot
        disagree."""
        raise NotImplementedError(f"{type(self).__name__} does not support least-squares fitting.")

    def _fitted_scale_label(self, name, i):
        """How mean parameter `i` is spelled on the fitted scale, for reporting -- "log(x)",
        "log(log(x))", or the bare name where the parameter is not transformed."""
        return f"transformed({name})"

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

        if np.all(x0 is None):  # intialize using least squares and 1D MLE
            _print_if("Getting initial deposition parameter guess via least squares", verbose)
            p0, sse = self.fit_least_squares(simulation_inputs, reflectance_data, verbose=False)

            _print_if("Getting initial sigma_dep guess via MLE (at least-squares value for deposition parameters)", verbose)

            def nloglike1D(y):
                return self._negative_log_likelihood([p0, y], sim_in, ref_dat)

            s0 = minimize_scalar(nloglike1D, bounds=(smb.tol, sse), method="Bounded")  # use bounded to prevent evaluation at values <=1
            x0 = np.array([p0, s0.x])
            _print_if("x0 = [" + str(x0[0]) + ", " + str(x0[1]) + "]", verbose)

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

    def _ls_lower_bounded(self):
        """Per-mean-parameter flag: True where the parameter must stay positive because the
        model represents it in logs. Defaults to every mean parameter (the constant-mean
        deposition coefficient is a rate and cannot be negative); the wind model overrides it,
        since its omegas are carried linearly and may fit negative."""
        return np.ones(len(self._mean_param_names), dtype=bool)


class SemiPhysical(smb.PhysicalBase, CommonFittingMethods):
    # hrz0 enters through the deposition-velocity physics, so the prediction is NOT affine in
    # it and AffineMeanLeastSquares does not apply; fit_ls below is a nonlinear fit instead.
    _mean_param_names = ("hrz0",)
    _sigma_param_names = ("sigma_dep",)

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

    def transform_scale(self, x, likelihood_hessian=None, direction="inverse"):
        # direction is either "forward" (to log-scaled space) or "inverse" (back to original scale)
        x = np.array(x)
        # A length-1 x is hrz0 alone -- what fit_ls estimates, since least squares fits no
        # sigma. hrz0 comes first in the parameter vector, so every expression below simply
        # stops after its first entry; a length-2 x is unchanged.
        if len(x) not in (1, 2):
            raise ValueError(f"Cannot transform a vector of {len(x)} parameter(s): expected 1 (hrz0) or 2 (hrz0 and sigma_dep).")
        if direction == "inverse":
            # Double-exp can overflow for large log-params; inf is handled downstream.
            with np.errstate(over="ignore"):
                z = np.array([np.exp(np.exp(x[0])), *(np.exp(x[1:]))])
        elif direction == "forward":
            z = np.array([np.log(np.log(x[0])), *(np.log(x[1:]))])
        else:
            raise ValueError("Transformation direction not recognized.")

        if not isinstance(likelihood_hessian, np.ndarray):  # can't use likelihood_hessian is None because it is an array if supplied
            return z
        else:
            # Jacobian for transformation. See Reparameterization at https://en.wikipedia.org/wiki/Fisher_information
            with np.errstate(over="ignore"):
                J = np.array([[np.exp(x[0] + np.exp(x[0])), 0], [0, np.exp(x[1])]])

            if direction == "inverse":
                Ji = inv(J)
                H = Ji.transpose() @ likelihood_hessian @ Ji
            elif direction == "forward":
                H = J.transpose() @ likelihood_hessian @ J

            return z, H

    def _fitted_scale_jacobian(self, x):
        # forward transform is log(log(hrz0)), so the derivative is 1 / (hrz0 * log(hrz0))
        x = np.asarray(x, dtype=float)
        return 1.0 / (x * np.log(x))

    def _fitted_scale_label(self, name, i):
        return f"log(log({name}))"

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
            self.hrz0 = x[0]
            if len(x) > 1:
                self.sigma_dep = x[1]
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
        H_log = np.linalg.inv(y_cov)

        _print_if("========== MLE Estimates ======== ", verbose)
        if transform_to_original_scale:
            x_hat, H = self.transform_scale(y, H_log)
            x_hat_cov = np.linalg.inv(H)

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

    def _fitted_scale_jacobian(self, x):
        # forward transform is log(mu_tilde), so the derivative is 1 / mu_tilde
        return 1.0 / np.asarray(x, dtype=float)

    def _fitted_scale_label(self, name, i):
        return f"log({name})"

    def transform_scale(self, x, likelihood_hessian=None, direction="inverse"):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = np.array(x)
            # Both parameters are log-transformed, so one elementwise map covers either the
            # full [mu_tilde, sigma_dep] vector or mu_tilde alone -- which is what fit_ls
            # returns, since least squares fits no sigma.
            if len(x) not in (1, 2):
                raise ValueError(f"Cannot transform a vector of {len(x)} parameter(s): expected 1 (mu_tilde) or 2 (mu_tilde and sigma_dep).")
            if direction == "inverse":
                z = np.exp(x)
            elif direction == "forward":
                z = np.log(x)
            else:
                raise ValueError("Transformation direction not recognized.")

            if not isinstance(likelihood_hessian, np.ndarray):  # can't use likelihood_hessian is None because it is an array if supplied
                return z
            else:
                # Jacobian for transformation. See Reparameterization at https://en.wikipedia.org/wiki/Fisher_information
                J = np.diag(np.exp(x))

                if direction == "inverse":
                    Ji = inv(J)
                    H = Ji.transpose() @ likelihood_hessian @ Ji
                elif direction == "forward":
                    H = J.transpose() @ likelihood_hessian @ J

                return z, H

        # elif isinstance(x,az.data.inference_data.InferenceData):
        #     if likelihood_hessian != None:
        #         print("Warning: You have supplied an Arviz infrenceData object. The supplied likelihood Hessian will be ignored.")

        #     p = x.posterior
        #     if direction == "inverse":
        #         p2 = {  'mu_tilde': np.exp(p.log_mu_tilde),\
        #                 'sigma_dep':np.exp(p.log_sigma_dep)
        #             }
        #     elif direction == "forward":
        #         p2 = {  'log_mu_tilde': np.log(p.mu_tilde),\
        #                 'log_sigma_dep':np.log(p.sigma_dep)
        #             }
        #     p2 = az.convert_to_inference_data(p2)
        #     return p2

    def update_model_parameters(self, x):
        if isinstance(x, list) or isinstance(x, np.ndarray):
            self.mu_tilde = x[0]
            if len(x) > 1:
                self.sigma_dep = x[1]
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
