import heliosoil.base_models as smb
from heliosoil.utilities import _print_if, _check_keys, _parse_dust_str, cosd
import numpy as np
from numpy import radians as rad
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize_scalar, minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, logit
import numdifftools as ndt
import pickle
import warnings


def _experiment_value(value, f):
    """Value for experiment ``f``, from a per-experiment mapping or a shared scalar."""
    if isinstance(value, dict):
        return value[f]
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[f]
    return value


class CommonFittingMethods:

    # Deposition noise model. "scalar" is the historical two-parameter model;
    # "components" splits the deposition variance into a site-common part and a
    # mirror-specific part and takes a third parameter. Declared at class level so that
    # models built before this option existed (e.g. unpickled) behave as before.
    variance_model = "scalar"
    _endpoint_correction = None

    # The variance split (common_variance_fraction, sigma_c, sigma_m) is declared on
    # SoilingBase, next to sigma_dep; update_model_parameters populates it here.

    # Mean parameter: how it is transformed for fitting, and what to call it. Set by
    # each model class.
    _mean_parameter_transform = "log"
    _mean_parameter_name = "mu_tilde"

    def set_variance_model(self, variance_model="scalar", endpoint_correction=None):
        """
        Choose how the deposition noise is modelled.

        The "scalar" model gives every mirror independent deposition noise of standard
        deviation ``sigma_dep``. The "components" model splits that variance into a part
        common to all mirrors at the site during an interval and a mirror-specific part,
        adding a third parameter ``kappa``, the common fraction. Separating the two
        requires at least two mirrors measured together.

        Args:
            variance_model (str): "scalar" or "components".
            endpoint_correction (bool, optional): Whether to model the correlation
                between consecutive reflectance differences created by the measurement
                they share. Defaults to following ``variance_model``, so that "scalar"
                reproduces earlier fits exactly.

        Raises:
            ValueError: If ``variance_model`` is not recognised.
        """
        if variance_model not in ("scalar", "components"):
            raise ValueError(
                f"variance_model must be 'scalar' or 'components', got {variance_model!r}."
            )
        self.variance_model = variance_model
        self._endpoint_correction = endpoint_correction

    @property
    def endpoint_correction(self):
        """Whether consecutive differences share their endpoint measurement noise."""
        if self._endpoint_correction is None:
            return self.variance_model == "components"
        return bool(self._endpoint_correction)

    @property
    def n_parameters(self):
        """Number of fitted parameters under the current variance model."""
        return 3 if self.variance_model == "components" else 2

    @property
    def parameter_names(self):
        """Parameter names on the natural scale."""
        names = [self._mean_parameter_name, "sigma_dep"]
        if self.variance_model == "components":
            names.append("kappa")
        return names

    @property
    def transformed_parameter_names(self):
        """Parameter names on the unconstrained fitting scale."""
        if self._mean_parameter_transform == "log_log":
            mean = f"log(log({self._mean_parameter_name}))"
        else:
            mean = f"log({self._mean_parameter_name})"
        names = [mean, "log(sigma_dep)"]
        if self.variance_model == "components":
            names.append("logit(kappa)")
        return names

    def transform_scale(self, x, likelihood_hessian=None, direction="inverse"):
        """
        Map parameters between their natural scale and the unconstrained fitting scale.

        The fitting scale is ``log(log(hrz0))`` or ``log(mu_tilde)`` for the mean
        parameter, ``log(sigma_dep)``, and for the components model ``logit(kappa)``.
        Optimising there enforces ``hrz0 > 1``, ``sigma_dep > 0`` and ``0 < kappa < 1``
        without bounds.

        Args:
            x: Parameter vector, on the fitting scale when ``direction`` is "inverse"
                and on the natural scale when it is "forward".
            likelihood_hessian (numpy.ndarray, optional): If supplied, the Hessian is
                transformed to the other scale and returned alongside the parameters.
            direction (str): "inverse" to the natural scale, "forward" to the fitting
                scale.

        Returns:
            numpy.ndarray: The transformed parameters, or a tuple of those and the
            transformed Hessian if one was supplied.
        """
        x = np.asarray(x, dtype=float)
        log_log_mean = self._mean_parameter_transform == "log_log"

        if direction == "inverse":
            values = [np.exp(np.exp(x[0])) if log_log_mean else np.exp(x[0]), np.exp(x[1])]
            if x.size > 2:
                values.append(expit(x[2]))
        elif direction == "forward":
            values = [np.log(np.log(x[0])) if log_log_mean else np.log(x[0]), np.log(x[1])]
            if x.size > 2:
                values.append(logit(x[2]))
        else:
            raise ValueError("Transformation direction not recognized.")
        z = np.array(values)

        if not isinstance(
            likelihood_hessian, np.ndarray
        ):  # can't use likelihood_hessian is None because it is an array if supplied
            return z

        # Jacobian d(natural)/d(fitting), evaluated at the point on the fitting scale.
        # See Reparameterization at https://en.wikipedia.org/wiki/Fisher_information
        y = x if direction == "inverse" else z
        derivatives = [
            np.exp(y[0] + np.exp(y[0])) if log_log_mean else np.exp(y[0]),
            np.exp(y[1]),
        ]
        if x.size > 2:
            fraction = expit(y[2])
            derivatives.append(fraction * (1.0 - fraction))
        J = np.diag(derivatives)

        if direction == "inverse":
            Ji = inv(J)
            H = Ji.transpose() @ likelihood_hessian @ Ji
        elif direction == "forward":
            H = J.transpose() @ likelihood_hessian @ J

        return z, H

    def _set_variance_components(self, kappa):
        """
        Record the common/mirror-specific split of the deposition variance.

        ``sigma_dep`` keeps its meaning as the total, so anything downstream that only
        needs the per-mirror predictive variance is unaffected.

        Args:
            kappa (float): Fraction of the deposition variance common to all mirrors.

        Raises:
            ValueError: If ``kappa`` is outside [0, 1].
        """
        kappa = float(kappa)
        if not 0.0 <= kappa <= 1.0:
            raise ValueError(f"The common variance fraction must be in [0, 1], got {kappa}.")
        self.common_variance_fraction = kappa
        if self.sigma_dep is not None:
            self.sigma_c = self.sigma_dep * np.sqrt(kappa)
            self.sigma_m = self.sigma_dep * np.sqrt(1.0 - kappa)

    def _check_variance_components_identifiable(self, reflectance_data):
        """
        Confirm some experiment has enough mirrors to separate the variance components.

        The across-mirror scatter identifies the mirror-specific component and the
        common movement of all mirrors identifies the common one, so a single mirror
        determines only their sum.

        Raises:
            ValueError: If no experiment has two or more mirrors.
        """
        mirrors = [reflectance_data.average[f].shape[1] for f in reflectance_data.average]
        if max(mirrors, default=0) < 2:
            raise ValueError(
                "Separating the common and mirror-specific deposition variances needs at "
                "least two mirrors measured together, but every experiment has one. Use "
                "set_variance_model('scalar') to fit the total deposition variance."
            )

    def _variance_component_summary(self, y_hat, y_cov):
        """
        Point estimates and 95% intervals for sigma_c and sigma_m.

        Uses the delta method on the log scale, which keeps the intervals positive.
        With ``u = log(sigma_dep)`` and ``v = logit(kappa)``, ``log(sigma_c) = u +
        log(kappa)/2`` and ``log(sigma_m) = u + log(1-kappa)/2``.

        Args:
            y_hat (numpy.ndarray): Estimates on the fitting scale.
            y_cov (numpy.ndarray): Their covariance on the fitting scale.

        Returns:
            dict: Maps "sigma_c" and "sigma_m" to (estimate, lower, upper).
        """
        kappa = expit(y_hat[2])
        gradients = {
            "sigma_c": np.array([0.0, 1.0, 0.5 * (1.0 - kappa)]),
            "sigma_m": np.array([0.0, 1.0, -0.5 * kappa]),
        }
        logs = {
            "sigma_c": y_hat[1] + 0.5 * np.log(kappa),
            "sigma_m": y_hat[1] + 0.5 * np.log1p(-kappa),
        }

        summary = {}
        for name, gradient in gradients.items():
            sd = np.sqrt(max(gradient @ y_cov @ gradient, 0.0))
            with np.errstate(over="ignore"):
                summary[name] = (
                    np.exp(logs[name]),
                    np.exp(logs[name] - 1.96 * sd),
                    np.exp(logs[name] + 1.96 * sd),
                )
        return summary

    def compute_soiling_factor(self, rho0=None):
        # Converts helios.delta_soiled_area into an accumulated area loss and
        # populates helios.soiling_factor.

        files = list(self.helios.tilt.keys())
        helios = self.helios
        helios.soiling_factor = {f: None for f in files}  # clear the soiling factor

        for f in files:
            if rho0 is None:
                N_helios = helios.tilt[f].shape[0]
                cumulative_soil0 = np.zeros(N_helios)  # start from clean
            else:
                inc_factor = self.helios.inc_ref_factor[f].squeeze()
                cumulative_soil0 = (
                    1 - rho0[f] / self.helios.nominal_reflectance
                ) / inc_factor  # back-calculated soiled area from measurement

            cumulative_soil = np.c_[cumulative_soil0, helios.delta_soiled_area[f]]
            cumulative_soil = np.cumsum(cumulative_soil, axis=1)  # accumulate soiling
            helios.soiling_factor[f] = (
                1 - cumulative_soil[:, 1::] * helios.inc_ref_factor[f]
            )  # soiling factor, still to be multiplied by rho0

        self.helios = helios

    def _predicted_reflectance(self, f, reflectance_data):
        """
        Predicted reflectance at the measurement times of experiment ``f``.

        Requires ``predict_soiling_factor(..., rho0=reflectance_data.rho0)`` to have
        been called first: the per-mirror initial reflectance is carried by the
        soiling factor, so the scaling here is the nominal reflectance, matching the
        ``b`` used in ``_compute_variance_of_measurements``.

        Args:
            f: Experiment (file) key.
            reflectance_data (ReflectanceMeasurements): Supplies ``prediction_indices``.

        Returns:
            numpy.ndarray: Predicted reflectance, shape (n_measurements, n_mirrors).
        """
        pi = reflectance_data.prediction_indices[f]
        sf = self.helios.soiling_factor[f]
        return self.helios.nominal_reflectance * sf[:, pi].transpose()

    def _reflectance_loss_factor(self, f):
        """
        Reflectance lost per unit soiled area, ``b = nominal_reflectance * inc_ref_factor``.

        Assumes a fixed reflectometer incidence angle within an experiment, so that ``b``
        is the same at both ends of a reflectance difference.

        Args:
            f: Experiment (file) key.

        Returns:
            float: The loss factor ``b``.

        Raises:
            ValueError: If ``inc_ref_factor[f]`` is not a single value.
        """
        inc = np.asarray(self.helios.inc_ref_factor[f])
        if inc.size != 1:
            raise ValueError(
                f"Experiment {f} has {inc.size} incidence-reflectance factors. The fitting "
                "likelihoods assume a fixed reflectometer incidence angle per experiment."
            )
        return self.helios.nominal_reflectance * float(inc.reshape(-1)[0])

    def _loading_matrix(self, f, simulation_inputs):
        """
        Deposition loading ``alpha_j * cos(tilt_j)`` on the simulation grid.

        ``alpha`` is the airborne dust concentration relative to the prototype
        distribution, and is a site quantity shared by all mirrors; the tilt history is
        per mirror.

        Args:
            f: Experiment (file) key.
            simulation_inputs (SimulationInputs): Supplies dust concentration and type.

        Returns:
            numpy.ndarray: Loading, shape (n_times, n_mirrors).
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

        alpha = sim_in.dust_concentration[f] / den[f]
        return (alpha[None, :] * cosd(self.helios.tilt[f])).transpose()

    def _difference_windows(self, f, reflectance_data):
        """
        Simulation-grid slices spanned by each reflectance difference.

        Difference ``i`` covers intervals ``k_{i-1}+1 ... k_i``, matching the inclusive
        cumulative sum in ``compute_soiling_factor``. Because all mirrors in an
        experiment share a measurement grid, these windows partition the timeline.

        Args:
            f: Experiment (file) key.
            reflectance_data (ReflectanceMeasurements): Supplies ``prediction_indices``.

        Returns:
            list[slice]: One slice per difference.
        """
        pi = reflectance_data.prediction_indices[f]
        return [slice(pi[i] + 1, pi[i + 1] + 1) for i in range(len(pi) - 1)]

    def _deposition_cross_products(self, f, simulation_inputs, reflectance_data):
        """
        Between-mirror deposition cross-products for each reflectance difference.

        Entry ``[i, p, q]`` is ``sum_j a_ij_p * a_ij_q`` over the intervals of difference
        ``i``, with ``a_ij_p = b * alpha_j * cos(tilt_j_p)``. Scaled by the variance
        components this gives the deposition part of the difference covariance.

        Args:
            f: Experiment (file) key.
            simulation_inputs (SimulationInputs): Supplies the loading matrix.
            reflectance_data (ReflectanceMeasurements): Supplies the difference windows.

        Returns:
            numpy.ndarray: Cross-products, shape (n_differences, n_mirrors, n_mirrors).
        """
        m = self._loading_matrix(f, simulation_inputs)
        b = self._reflectance_loss_factor(f)
        windows = self._difference_windows(f, reflectance_data)

        cross_products = np.empty((len(windows), m.shape[1], m.shape[1]))
        for i, window in enumerate(windows):
            loading = m[window, :]
            cross_products[i] = b**2 * (loading.transpose() @ loading)
        return cross_products

    def _differenced_measurement_covariance(self, f, reflectance_data, endpoint_correction=True):
        """
        Covariance of the differenced reflectometer noise, per mirror.

        Successive differences share the measurement at their common endpoint, which
        makes the covariance tridiagonal. Setting ``endpoint_correction`` False keeps
        only the diagonal, i.e. treats the differences as independent.

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

    def _experiment_covariance(
        self,
        f,
        sigma_dep,
        kappa,
        simulation_inputs,
        reflectance_data,
        endpoint_correction=True,
    ):
        """
        Covariance of all reflectance differences in one experiment.

        Deposition noise is split into a common component shared by every mirror during
        an interval, ``sigma_c**2 = kappa * sigma_dep**2``, and a mirror-specific component,
        ``sigma_m**2 = (1 - kappa) * sigma_dep**2``. The common component couples mirrors;
        the mirror-specific component and the measurement noise do not.

        Entries are ordered mirror-major: difference ``i`` of mirror ``p`` is at index
        ``p * n_differences + i``.

        Args:
            f: Experiment (file) key.
            sigma_dep (float): Total deposition noise standard deviation.
            kappa (float): Fraction of the deposition variance that is common to all
                mirrors, in [0, 1].
            simulation_inputs (SimulationInputs): Simulation inputs.
            reflectance_data (ReflectanceMeasurements): Measurement data.
            endpoint_correction (bool): Passed to
                ``_differenced_measurement_covariance``.

        Returns:
            numpy.ndarray: Covariance, shape (n_mirrors * n_differences, same).
        """
        cross_products = self._deposition_cross_products(f, simulation_inputs, reflectance_data)
        meas_cov = self._differenced_measurement_covariance(
            f, reflectance_data, endpoint_correction=endpoint_correction
        )
        n_diff, n_mirrors = cross_products.shape[0], cross_products.shape[1]

        s2_common = kappa * sigma_dep**2
        s2_mirror = (1.0 - kappa) * sigma_dep**2

        # Assembled as (mirror, difference, mirror, difference) and then flattened, which
        # keeps the mirror-major ordering while avoiding a Python loop over mirror pairs.
        # This runs on every likelihood evaluation, so it is worth vectorising.
        cov = np.zeros((n_mirrors, n_diff, n_mirrors, n_diff))
        diag = np.arange(n_diff)
        mirrors = np.arange(n_mirrors)

        # Deposition is diagonal in the difference index because the windows partition
        # the timeline; the common component couples mirrors, the mirror-specific one
        # only adds to the diagonal.
        weights = s2_common * np.ones((n_mirrors, n_mirrors)) + s2_mirror * np.eye(n_mirrors)
        cov[:, diag, :, diag] = weights[None, :, :] * cross_products

        # Measurement noise acts within a single mirror.
        cov[mirrors, :, mirrors, :] += meas_cov

        return cov.reshape(n_mirrors * n_diff, n_mirrors * n_diff)

    def _negative_log_likelihood_components(
        self,
        params,
        simulation_inputs,
        reflectance_data,
        endpoint_correction=True,
    ):
        """
        Negative log-likelihood with the deposition noise split into variance components.

        Generalises ``_negative_log_likelihood`` to mirrors observed simultaneously at one
        site, whose deposition noise is partly shared. Each experiment contributes one
        multivariate normal over its stacked reflectance differences. Differences with a
        missing endpoint measurement are dropped, which marginalises them exactly.

        Separating the two components requires at least two mirrors in some experiment;
        with one mirror the likelihood depends only on ``sigma_dep``.

        Args:
            params: ``(mean_parameter, sigma_dep, kappa)``, where ``mean_parameter`` is
                ``hrz0`` or ``mu_tilde`` depending on the model, ``sigma_dep**2`` is the
                total deposition variance and ``kappa`` is the common fraction.
            simulation_inputs (SimulationInputs): Simulation inputs.
            reflectance_data (ReflectanceMeasurements): Measurement data.
            endpoint_correction (bool): Whether to model the correlation between
                consecutive differences induced by their shared measurement.

        Returns:
            float: The negative log-likelihood.
        """
        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        sigma_dep, kappa = params[1], params[2]
        self.update_model_parameters(params[0:2])
        self.predict_soiling_factor(simulation_inputs, rho0=reflectance_data.rho0, verbose=False)

        nll = 0.0
        for f in list(reflectance_data.times.keys()):
            observed = np.diff(reflectance_data.average[f], axis=0)
            predicted = np.diff(self._predicted_reflectance(f, reflectance_data), axis=0)

            # mirror-major ordering, matching _experiment_covariance
            residual = (observed - predicted).transpose().ravel()
            cov = self._experiment_covariance(
                f,
                sigma_dep,
                kappa,
                simulation_inputs,
                reflectance_data,
                endpoint_correction=endpoint_correction,
            )

            observed_mask = np.isfinite(residual)
            if not observed_mask.all():
                residual = residual[observed_mask]
                cov = cov[np.ix_(observed_mask, observed_mask)]

            factor = cho_factor(cov, lower=True)
            nll += 0.5 * (
                residual.size * np.log(2 * np.pi)
                + 2 * np.sum(np.log(np.diag(factor[0])))
                + residual @ cho_solve(factor, residual)
            )

        return nll

    def simulate_reflectance_data(
        self,
        simulation_inputs,
        params,
        measurement_indices,
        measurement_sigma,
        rho0=None,
        number_of_measurements=1.0,
        reflectometer_incidence_angle=None,
        rng=None,
        missing_fraction=0.0,
    ):
        """
        Simulates a reflectance dataset from the model, ready to be fitted.

        Draws deposition from the generative process via
        ``SoilingBase.random_delta_soiled_area``, accumulates it into a reflectance path,
        samples that path at the requested times and adds measurement noise. The result
        is a ``ReflectanceMeasurements`` that ``fit_mle`` accepts.

        Args:
            simulation_inputs (SimulationInputs): Simulation inputs. Tilts and
                ``inc_ref_factor`` must already be set on ``helios``, as
                ``helios_angles`` does.
            params: Parameter vector, ``(mean_parameter, sigma_dep)`` plus ``kappa`` for
                the components model.
            measurement_indices: Indices into the simulation grid at which the mirrors
                are measured, either one sequence applied to every experiment or a dict
                keyed by experiment.
            measurement_sigma: Standard deviation of a single reflectometer reading,
                scalar or per experiment. The noise actually added has standard deviation
                ``measurement_sigma / sqrt(number_of_measurements)``, matching the
                ``sigma_of_the_mean`` the likelihood uses.
            rho0 (dict, optional): Initial reflectance per mirror. Defaults to the
                nominal reflectance for every mirror.
            number_of_measurements (float): Readings behind each reported mean.
            reflectometer_incidence_angle (float, optional): Recorded on the result.
                Defaults to ``helios.incidence_angle`` when available, else zero.
            rng (numpy.random.Generator, optional): Source of randomness.
            missing_fraction (float): Fraction of measurements to replace with NaN, for
                exercising the missing-data path. Defaults to none.

        Returns:
            ReflectanceMeasurements: The simulated dataset.
        """
        if rng is None:
            rng = np.random.default_rng()
        if not 0.0 <= missing_fraction < 1.0:
            raise ValueError(f"missing_fraction must be in [0, 1), got {missing_fraction}.")

        sim_in = simulation_inputs
        files = list(self.helios.tilt.keys())

        if not isinstance(measurement_indices, dict):
            measurement_indices = {f: list(measurement_indices) for f in files}
        if rho0 is None:
            rho0 = {
                f: np.full(self.helios.tilt[f].shape[0], self.helios.nominal_reflectance)
                for f in files
            }

        self.update_model_parameters(params)

        # Populates delta_soiled_area_variance, and for the semi-physical model the
        # deposition flux and adhesion removal that the draw depends on.
        self.predict_soiling_factor(sim_in, rho0=rho0, verbose=False)
        deposited = self.random_delta_soiled_area(sim_in, rng=rng, verbose=False)

        times, average, sigma, tilts, angles = [], [], [], [], []
        for f in files:
            indices = measurement_indices[f]
            b = self._reflectance_loss_factor(f)

            # Same relation compute_soiling_factor inverts: an inclusive cumulative sum,
            # so index k already includes the deposition during interval k.
            path = np.asarray(rho0[f])[:, None] - b * np.cumsum(deposited[f], axis=1)
            clean = path[:, indices].transpose()

            sigma_f = np.full(clean.shape, _experiment_value(measurement_sigma, f))
            noise_sd = sigma_f / np.sqrt(_experiment_value(number_of_measurements, f))
            measured = clean + noise_sd * rng.standard_normal(clean.shape)

            if missing_fraction > 0.0:
                measured = measured.copy()
                measured[rng.random(measured.shape) < missing_fraction] = np.nan

            if reflectometer_incidence_angle is not None:
                angle = _experiment_value(reflectometer_incidence_angle, f)
            else:
                angle = float(np.asarray(self.helios.incidence_angle.get(f, 0.0)).reshape(-1)[0])

            times.append(np.asarray(sim_in.time[f])[indices])
            average.append(measured)
            sigma.append(sigma_f)
            tilts.append(self.helios.tilt[f])
            angles.append(angle)

        # Carry the simulation inputs' labels so that _check_keys passes and the result
        # can be handed straight to the likelihood.
        source_names = getattr(sim_in, "files", None)
        names = [source_names[f] for f in files] if source_names else None

        return smb.ReflectanceMeasurements.from_arrays(
            times=times,
            average=average,
            sigma=sigma,
            time_grids=[np.asarray(sim_in.time[f]) for f in files],
            tilts=tilts,
            number_of_measurements=[
                _experiment_value(number_of_measurements, f) for f in files
            ],
            reflectometer_incidence_angle=angles,
            names=names,
        )

    def _compute_variance_of_measurements(
        self, sigma_dep, simulation_inputs, reflectance_data=None
    ):
        """ "
        Computes the total variance of the reflectance measurements, including both the measurement error and the variance due to the soiling model parameters.

        The function takes in the standard deviation of the model parameters (`sigma_dep`), the simulation inputs (`simulation_inputs`), and optionally
        the reflectance data (`reflectance_data`). It first checks that the keys in the simulation inputs and reflectance data match. Then, it computes
        the total variance for each file, taking into account the measurement error and the variance due to the soiling model parameters.

        The function returns a dictionary `s2total` that contains the total variance for each file.
        """
        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        sim_in = simulation_inputs
        files = list(sim_in.time.keys())

        s2_dep = sigma_dep**2
        s2total = dict.fromkeys(files)
        for f in files:

            if reflectance_data is None:
                pif = range(0, len(sim_in.time[f]))
            else:
                pif = reflectance_data.prediction_indices[f]

            b = self._reflectance_loss_factor(f)
            m = self._loading_matrix(f, sim_in)

            if reflectance_data is None:
                meas_sig = np.zeros(self.helios.tilt[f].shape).transpose()
            else:
                meas_sig = reflectance_data.sigma_of_the_mean[f]

            # Difference i spans simulation intervals k_{i-1}+1 ... k_i, matching the
            # inclusive cumulative sum in compute_soiling_factor that forms the mean.
            c2t = np.cumsum(m**2, axis=0)
            ind1 = pif[0:-1]
            ind2 = pif[1::]
            s2total[f] = (
                s2_dep * b**2 * (c2t[ind2, :] - c2t[ind1, :])
                + meas_sig[0:-1, :] ** 2
                + meas_sig[1::, :] ** 2
            )

        return s2total

    def _sse(self, params, simulation_inputs, reflectance_data):
        # Computes the sum of squared errors between a soiling model and
        # the reflectance measurements.

        meas = reflectance_data.average

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        sse = 0
        self.update_model_parameters(params)
        self.predict_soiling_factor(simulation_inputs, rho0=reflectance_data.rho0, verbose=False)
        files = list(self.helios.soiling_factor.keys())
        for f in files:
            rho_prediction = self._predicted_reflectance(f, reflectance_data)
            sse += np.sum((rho_prediction - meas[f]) ** 2)
        return sse

    def _negative_log_likelihood(self, params, simulation_inputs, reflectance_data):
        """
        Negative log-likelihood under the currently selected variance model.

        Args:
            params: ``(mean_parameter, sigma_dep)``, plus ``kappa`` for the components
                model.
            simulation_inputs (SimulationInputs): Simulation inputs.
            reflectance_data (ReflectanceMeasurements): Measurement data.

        Returns:
            float: The negative log-likelihood.
        """
        if self.variance_model == "components":
            return self._negative_log_likelihood_components(
                params,
                simulation_inputs,
                reflectance_data,
                endpoint_correction=self.endpoint_correction,
            )
        return self._negative_log_likelihood_scalar(
            params, simulation_inputs, reflectance_data
        )

    def _negative_log_likelihood_scalar(self, params, simulation_inputs, reflectance_data):

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        sim_in = simulation_inputs
        files = list(reflectance_data.times.keys())
        meas = reflectance_data.average

        # define optimization objective function (negative log likelihood)
        sigma_dep = params[1]
        self.update_model_parameters(params)
        self.predict_soiling_factor(simulation_inputs, rho0=reflectance_data.rho0, verbose=False)

        # Compute variance in reflectance, not soiling factor
        s2total = self._compute_variance_of_measurements(
            sigma_dep, sim_in, reflectance_data=reflectance_data
        )

        # One Gaussian term per reflectance difference, per mirror, per experiment.
        n_terms = sum(s2total[f].size for f in files)
        loglike = -0.5 * n_terms * np.log(2 * np.pi)

        for f in files:
            delta_r = np.diff(meas[f], axis=0)
            rho_prediction = self._predicted_reflectance(f, reflectance_data)
            mu_delta_r = np.diff(rho_prediction, axis=0)
            loglike += np.sum(
                -0.5 * np.log(s2total[f]) - (delta_r - mu_delta_r) ** 2 / (2 * s2total[f])
            )

        return -loglike

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

    def _mean_parameter_bounds(self, simulation_inputs, reflectance_data):
        """
        Bracket for the one-dimensional least-squares search over the mean parameter.

        The bracket is model specific because the mean parameter is: `hrz0` is bounded
        below by one, while `mu_tilde` is a small positive number.

        Args:
            simulation_inputs (SimulationInputs): Simulation inputs.
            reflectance_data (ReflectanceMeasurements): Measurement data.

        Returns:
            tuple: Lower and upper bounds.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define _mean_parameter_bounds so that "
            "fit_least_squares searches a range appropriate to its mean parameter."
        )

    def fit_least_squares(self, simulation_inputs, reflectance_data, verbose=True):

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        def fun(x):
            return self._sse(x, simulation_inputs, reflectance_data)

        _print_if("Fitting parameters with least squares ...", verbose)
        xL, xU = self._mean_parameter_bounds(simulation_inputs, reflectance_data)
        width = xU - xL

        # Resolve the optimum far more finely than the edge test below. The default
        # absolute tolerance is 1e-5 regardless of the bracket, which on a narrow
        # bracket is coarser than the edge test and would hide a pinned result.
        res = minimize_scalar(
            fun, bounds=(xL, xU), method="Bounded", options={"xatol": 1e-9 * width}
        )

        # A bounded search stops near an endpoint when the optimum lies outside the
        # bracket, which is otherwise indistinguishable from a converged fit.
        if min(res.x - xL, xU - res.x) <= 1e-6 * width:
            warnings.warn(
                f"Least-squares estimate of {self._mean_parameter_name} is at the edge of "
                f"the search bracket [{xL:.3e}, {xU:.3e}]. The optimum probably lies "
                "outside it, and the value returned is the bound rather than a fit.",
                RuntimeWarning,
            )

        _print_if("... done! \n estimated parameter is = " + str(res.x), verbose)
        return res.x, res.fun

    def fit_mle(
        self,
        simulation_inputs,
        reflectance_data,
        verbose=True,
        x0=None,
        transform_to_original_scale=False,
        save_file=None,
        **optim_kwargs,
    ):
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

        if self.variance_model == "components":
            self._check_variance_components_identifiable(ref_dat)

        if np.all(x0 is None):  # intialize using least squares and 1D MLE
            _print_if("Getting initial deposition parameter guess via least squares", verbose)
            p0, sse = self.fit_least_squares(simulation_inputs, reflectance_data, verbose=False)

            _print_if(
                "Getting initial sigma_dep guess via MLE (at least-squares value for deposition parameters)",
                verbose,
            )

            # Start from an even split of the deposition variance, which is interior to
            # the admissible range of kappa and so avoids starting on a boundary.
            def nloglike1D(y):
                return self._negative_log_likelihood(
                    [p0, y, 0.5][0 : self.n_parameters], sim_in, ref_dat
                )

            s0 = minimize_scalar(
                nloglike1D, bounds=(smb.tol, sse), method="Bounded"
            )  # use bounded to prevent evaluation at values <=1
            x0 = np.array([p0, s0.x, 0.5][0 : self.n_parameters])
            _print_if("x0 = [" + ", ".join(f"{v}" for v in x0) + "]", verbose)

        # MLE. Transform to logs to ensure parameters are positive
        _print_if("Maximizing likelihood ...", verbose)
        y0 = self.transform_scale(x0, direction="forward")

        def nloglike(y):
            return self._negative_log_likelihood(self.transform_scale(y), sim_in, ref_dat)

        res = minimize(nloglike, y0, **optim_kwargs)
        y = res.x
        _print_if("  " + res.message, verbose)

        _print_if(
            "Estimating parameter covariance using numerical approximation of Hessian ... ",
            verbose,
        )
        H_log = ndt.Hessian(nloglike)(y)  # Hessian is in the log transformed space

        try:
            y_cov = np.linalg.inv(H_log)  # Parameter covariance in the log space
        except np.linalg.LinAlgError:
            if np.linalg.det(H_log) == 0:
                y_cov = np.linalg.pinv(H_log)  # Use pseudoinverse if determinant is zero
            else:
                raise  # Re-raise the exception if it's not due to zero determinant

        if transform_to_original_scale:
            # Get standard errors using observed information
            p_hat, H = self.transform_scale(y, likelihood_hessian=H_log)
            p_cov = np.linalg.inv(H)
            names = self.parameter_names
        else:
            p_hat, p_cov = y, y_cov
            names = self.transformed_parameter_names

        _print_if("... done!", verbose)
        standard_errors = np.sqrt(np.diag(p_cov))
        ci = p_hat + 1.96 * standard_errors * np.array([[-1], [1]])
        fmt = "  {0:s} = {1:.3e}, 95% confidence interval: [{2:.3e}, {3:.3e}]"
        for ii, name in enumerate(names):
            _print_if(fmt.format(name, p_hat[ii], ci[0, ii], ci[1, ii]), verbose)

        if verbose and self.variance_model == "components":
            # Reported on the natural scale in both cases: the split is what the
            # components model exists to estimate, and it is not one of the coordinates.
            for name, (estimate, lower, upper) in self._variance_component_summary(
                y, y_cov
            ).items():
                _print_if(fmt.format(name, estimate, lower, upper), verbose)

        _print_if("", verbose)
        return p_hat, p_cov

    def save_data(
        self,
        log_p_hat=None,
        log_p_hat_cov=None,
        training_simulation_data=None,
        training_reflectance_data=None,
    ):
        save_data = {
            "model": self,
            "type": None,
            "variance_model": self.variance_model,
            "endpoint_correction": self.endpoint_correction,
        }
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
            self.predict_soiling_factor(simulation_inputs, rho0=reflectance_data.rho0)
        else:
            self.predict_soiling_factor(simulation_inputs)

        sim_in = simulation_inputs
        samples = posterior_predictive_distribution_samples
        files = list(sim_in.time.keys())
        N_mirrors = np.array([self.helios.tilt[f].shape[0] for f in files])
        if np.all(N_mirrors == N_mirrors[0]):
            N_mirrors = N_mirrors[0]
        else:
            raise ValueError(
                "Number of mirrors must be the same for each experiment to use this function."
            )

        if reflectance_data is not None:
            # check to ensure that reflectance_data and simulation_input keys correspond to the same files
            _check_keys(sim_in, reflectance_data)

        N_experiments = sim_in.N_simulations
        ws_max = max(
            [max(sim_in.wind_speed[f]) for f in files]
        )  # max wind speed for setting y-axes
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
                    r0 = self.helios.nominal_reflectance

                    if reflectance_std == "measurements":
                        s = reflectance_data.sigma[f][:, jj]
                    elif reflectance_std == "mean":
                        s = reflectance_data.sigma_of_the_mean[f][:, jj]
                    else:
                        raise ValueError(
                            "reflectance_std="
                            + reflectance_std
                            + ' not recognized. Must be either "measurements" or "mean" '
                        )

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
                        a.plot(
                            sim_in.time[f],
                            ym,
                            label="Reflectance Prediction (Bayesian)",
                            color="red",
                        )

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
                elif (
                    samples is not None
                ):  # use percentiles of posterior predictive samples for confidence intervals
                    Lp = np.percentile(y, 2.5, axis=1)
                    Up = np.percentile(y, 97.5, axis=1)
                    a.fill_between(ts, Lp, Up, color="red", alpha=0.1, label=r"95% Bayesian CI")

                a.xaxis.set_major_locator(
                    mdates.DayLocator(interval=1)
                )  # sets x ticks to day interval

                if (
                    reflectance_data is not None
                ):  # reflectance is computed at reflectometer incidence angle
                    if repeat_y_labels or (ii == 0):
                        ang = reflectance_data.reflectometer_incidence_angle[f]
                        s = a.set_ylabel(r"$\rho(t)$ at " + str(ang) + r"$^{\circ}$")
                    else:
                        a.set_yticklabels([])

                else:  # reflectance is computed at heliostat incidence angle. Put average incidence angle on axis label
                    if repeat_y_labels or (ii > 0):
                        ang = np.mean(self.helios.incidence_angle[f])
                        s = a.set_ylabel(
                            r"soiling factor at " + str(ang) + r"$^{{\circ}}$ \n (average)"
                        )
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
            a2.xaxis.set_major_locator(
                mdates.DayLocator(interval=1)
            )  # sets x ticks to day interval
            myFmt = mdates.DateFormatter("%d-%m-%Y")
            a2.xaxis.set_major_formatter(myFmt)
            a2.tick_params(axis="y", labelcolor="blue")

            a2a = a2.twinx()
            a2a.plot(
                ts, ws, color="green", label="Wind Speed (mean = {0:.2f} m/s)".format(ws.mean())
            )
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

            a2.set_title(
                label_str.format(dust_conc.mean())
                + ", \n Wind Speed (mean = {0:.2f} m/s)".format(ws.mean()),
                fontsize=20,
            )

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
        if save_path is not None:
            fig.savefig(save_path)

        fig.suptitle(fig_title, fontsize=16)
        fig.tight_layout()
        if return_handles:
            return fig, ax, mean_predictions, CI_lower_predictions, CI_upper_predictions
        else:
            return mean_predictions, CI_lower_predictions, CI_upper_predictions


class SemiPhysical(smb.PhysicalBase, CommonFittingMethods):

    _mean_parameter_transform = "log_log"
    _mean_parameter_name = "hrz0"

    def __init__(self, file_params, variance_model="scalar", endpoint_correction=None):
        table = pd.read_excel(file_params, index_col="Parameter")
        super().__init__()
        self.set_variance_model(variance_model, endpoint_correction=endpoint_correction)
        self.import_site_data_and_constants(file_params)
        self.helios.hamaker = float(table.loc["hamaker_glass"].Value)
        self.helios.poisson = float(table.loc["poisson_glass"].Value)
        self.helios.youngs_modulus = float(table.loc["youngs_modulus_glass"].Value)
        self.helios.nominal_reflectance = float(table.loc["nominal_reflectance"].Value)
        if not (isinstance(self.helios.stow_tilt, float)) and not (
            isinstance(self.helios.stow_tilt, int)
        ):
            self.helios.stow_tilt = None

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
            self.helios.acceptance_angles[f] = [
                ref_dat.reflectometer_acceptance_angle[ii]
            ] * N_helios
            self.helios.extinction_weighting[f] = (
                []
            )  # reset extinction weighting since heliostats are "new" - WHY?? THIS DEPENDS ONLY ON DUST! NOT?

            helios.tilt[f] = np.zeros((0, N_times))
            for jj in range(N_helios):
                row_mask = np.ones((1, N_times))
                helios.tilt[f] = np.vstack((helios.tilt[f], tilts[jj] * row_mask))

            helios.elevation[f] = 90 - helios.tilt[f]
            helios.incidence_angle[f] = reflectance_data.reflectometer_incidence_angle[f]

            if not second_surface:
                helios.inc_ref_factor[f] = (1 + np.sin(rad(helios.incidence_angle[f]))) / np.cos(
                    rad(helios.incidence_angle[f])
                )  # first surface
                helios.aoi_model = "first_surface"
                _print_if("First surface model", verbose)
            elif second_surface:
                helios.inc_ref_factor[f] = 2 / np.cos(
                    rad(helios.incidence_angle[f])
                )  # second surface model
                helios.aoi_model = "second_surface"
                _print_if("Second surface model", verbose)
            else:
                _print_if("Choose either first or second surface model", verbose)

        self.helios = helios

    def predict_soiling_factor(
        self, simulation_inputs, rho0=None, hrz0=None, sigma_dep=None, verbose=True
    ) -> None:
        # Uses simulation inputs and model parameters to predict the soiling
        # factor and the prediction variance (stored in
        # helios.soiling_factor and helios.soiling_factor_prediction_variance,
        # respectively).

        self.deposition_flux(simulation_inputs, hrz0=hrz0, verbose=verbose)
        self.adhesion_removal(simulation_inputs, verbose=verbose)
        self.calculate_delta_soiled_area(simulation_inputs, sigma_dep=sigma_dep, verbose=verbose)
        self.compute_soiling_factor(rho0=rho0)

        # prediction variance
        if self.sigma_dep is not None:
            for f in self.helios.soiling_factor.keys():
                inc_factor = self.helios.inc_ref_factor[f]
                dsav = self.helios.delta_soiled_area_variance[f]
                self.helios.soiling_factor_prediction_variance[f] = inc_factor**2 * np.cumsum(
                    dsav, axis=1
                )
        else:
            self.helios.soiling_factor_prediction_variance = {}

    def _mean_parameter_bounds(self, simulation_inputs, reflectance_data):
        """
        Bracket for the least-squares search over `hrz0`.

        `hrz0` is a ratio of the boundary layer height to the surface roughness length,
        so it exceeds one; the `log(log(hrz0))` fitting transform requires it too. The
        upper bound is a generous fixed value.

        Returns:
            tuple: Lower and upper bounds.
        """
        return 1.0 + 1e-6, 1000.0

    def update_model_parameters(self, x):
        """
        Updates the model parameters from a parameter vector.

        The first element is `hrz0`, the second (if present) `sigma_dep`, and the third
        (if present) the fraction of the deposition variance common to all mirrors. A
        bare scalar sets `hrz0` and clears `sigma_dep`.
        """
        if isinstance(x, list) or isinstance(x, np.ndarray):
            self.hrz0 = x[0]
            if len(x) > 1:
                self.sigma_dep = x[1]
            if len(x) > 2:
                self._set_variance_components(x[2])
        else:
            self.hrz0 = x
            self.sigma_dep = None

    def save(
        self,
        file_name,
        log_p_hat=None,
        log_p_hat_cov=None,
        training_simulation_data=None,
        training_reflectance_data=None,
    ):
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

            save_data = {
                "model": self,
                "type": "semi-physical",
                "variance_model": self.variance_model,
                "endpoint_correction": self.endpoint_correction,
            }
            if log_p_hat is not None:
                save_data["transformed_parameters"] = log_p_hat
            if log_p_hat_cov is not None:
                save_data["transformed_parameter_covariance"] = log_p_hat_cov
            if training_simulation_data is not None:
                save_data["simulation_data"] = training_simulation_data
            if training_reflectance_data is not None:
                save_data["reflectance_data"] = training_reflectance_data

            pickle.dump(save_data, f)


class ConstantMeanDeposition(smb.ConstantMeanBase, CommonFittingMethods):
    simulation_inputs: smb.SimulationInputs
    reflectance_data: smb.ReflectanceMeasurements

    _mean_parameter_transform = "log"
    _mean_parameter_name = "mu_tilde"

    def __init__(self, file_params, variance_model="scalar", endpoint_correction=None):
        super().__init__()
        self.set_variance_model(variance_model, endpoint_correction=endpoint_correction)
        self.import_site_data_and_constants(file_params)
        table = pd.read_excel(file_params, index_col="Parameter")
        self.helios.nominal_reflectance = float(table.loc["nominal_reflectance"].Value)

    def helios_angles(
        self,
        simulation_inputs: smb.SimulationInputs,
        reflectance_data: smb.ReflectanceMeasurements,
        verbose=True,
        second_surface=True,
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
            tilts = ref_dat.tilts[
                f
            ]  # THIS COULD NOT MANAGE THE TRANSFORMATION OF SIM_IN WITH DAILY AVERAGE WHEN CHANGING START TIME
            N_times = len(sim_in.time[f])
            N_helios = tilts.shape[0]
            self.helios.acceptance_angles[f] = [
                ref_dat.reflectometer_acceptance_angle[ii]
            ] * N_helios
            self.helios.extinction_weighting[f] = (
                []
            )  # reset extinction weighting since heliostats are "new"

            helios.tilt[f] = np.zeros((0, N_times))
            for jj in range(N_helios):
                row_mask = np.ones((1, N_times))
                helios.tilt[f] = np.vstack((helios.tilt[f], tilts[jj] * row_mask))

            helios.elevation[f] = 90 - helios.tilt[f]
            helios.incidence_angle[f] = reflectance_data.reflectometer_incidence_angle[f]

            if not second_surface:
                helios.inc_ref_factor[f] = (1 + np.sin(rad(helios.incidence_angle[f]))) / np.cos(
                    rad(helios.incidence_angle[f])
                )  # first surface
                helios.aoi_model = "first_surface"
                _print_if("First surface model", verbose)
            elif second_surface:
                helios.inc_ref_factor[f] = 2 / np.cos(
                    rad(helios.incidence_angle[f])
                )  # second surface model
                helios.aoi_model = "second_surface"
                _print_if("Second surface model", verbose)
            else:
                _print_if("Choose either first or second surface model", verbose)

        self.helios = helios

    def predict_soiling_factor(
        self,
        simulation_inputs: smb.SimulationInputs,
        rho0=None,
        mu_tilde=None,
        sigma_dep=None,
        verbose=True,
    ):

        sim_in = simulation_inputs
        self.calculate_delta_soiled_area(
            sim_in, mu_tilde=mu_tilde, sigma_dep=sigma_dep, verbose=verbose
        )
        self.compute_soiling_factor(rho0=rho0)

        # prediction variance
        if self.sigma_dep is not None:
            for f in self.helios.soiling_factor.keys():
                inc_factor = self.helios.inc_ref_factor[f]
                dsav = self.helios.delta_soiled_area_variance[f]
                self.helios.soiling_factor_prediction_variance[f] = inc_factor**2 * np.cumsum(
                    dsav, axis=1
                )
        else:
            self.helios.soiling_factor_prediction_variance = {}

    def fit_map(
        self,
        simulation_inputs,
        reflectance_data,
        priors,
        verbose=True,
        x0=None,
        transform_to_original_scale=False,
        save_file=None,
    ):

        _print_if("Getting MAP estimates ... ", verbose)
        y, y_cov = super().fit_map(
            simulation_inputs,
            reflectance_data,
            priors,
            verbose=False,
            x0=x0,
            transform_to_original_scale=False,
        )

        _print_if("========== MAP Estimates ======== ", verbose)
        if transform_to_original_scale:
            x_hat, x_hat_cov = self.transform_scale(y, y_cov)

            # print estimates
            fmt = "mu_tilde = {0:.2e}, sigma_dep = {1:.2e}"
            _print_if(fmt.format(x_hat[0], x_hat[1]), verbose)

            # print confidence intervals
            s = np.sqrt(np.diag(x_hat_cov))
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
            s = np.sqrt(np.diag(x_hat_cov))
            x_ci = x_hat + 1.96 * s * np.array([[-1], [1]])
            fmt = "95% confidence interval for {0:s}: [{1:.2e}, {2:.2e}]"
            _print_if(fmt.format("log(mu_tilde)", x_ci[0, 0], x_ci[1, 0]), verbose)
            _print_if(fmt.format("log(sigma_dep)", x_ci[0, 1], x_ci[1, 1]), verbose)

        return x_hat, x_hat_cov

    def _mean_parameter_bounds(self, simulation_inputs, reflectance_data):
        """
        Bracket for the least-squares search over `mu_tilde`.

        `mu_tilde` scales the soiled area directly: mirror `p` accumulates
        `mu_tilde * sum_j alpha_j cos(tilt_j_p)` over the measured span, and the
        predicted reflectance is `rho0` minus `b` times that. Requiring the prediction
        to stay positive gives

            mu_tilde < min_p rho0_p / (b * sum_j alpha_j cos(tilt_j_p))

        which adapts to the dust loading and tilt history of the data rather than
        assuming a scale. Falls back to one if the loading is degenerate.

        Args:
            simulation_inputs (SimulationInputs): Supplies the dust loading.
            reflectance_data (ReflectanceMeasurements): Supplies `rho0` and the
                measurement span.

        Returns:
            tuple: Lower and upper bounds.
        """
        upper = np.inf
        for f in reflectance_data.average:
            loading = self._loading_matrix(f, simulation_inputs)
            last = reflectance_data.prediction_indices[f][-1]
            accumulated = np.sum(loading[: last + 1, :], axis=0)  # per mirror
            b = self._reflectance_loss_factor(f)
            rho0 = np.asarray(reflectance_data.rho0[f], dtype=float)

            positive = accumulated > 0
            if not np.any(positive):
                continue
            upper = min(upper, float(np.min(rho0[positive] / (b * accumulated[positive]))))

        if not np.isfinite(upper) or upper <= 0:
            upper = 1.0
        return smb.tol, upper

    def update_model_parameters(self, x):
        """
        Updates the model parameters from a parameter vector.

        The first element is `mu_tilde`, the second (if present) `sigma_dep`, and the
        third (if present) the fraction of the deposition variance common to all
        mirrors.
        """
        if isinstance(x, list) or isinstance(x, np.ndarray):
            self.mu_tilde = x[0]
            if len(x) > 1:
                self.sigma_dep = x[1]
            if len(x) > 2:
                self._set_variance_components(x[2])
        else:
            self.mu_tilde = x

    def save(
        self,
        file_name,
        log_p_hat=None,
        log_p_hat_cov=None,
        training_simulation_data=None,
        training_reflectance_data=None,
    ):
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
            save_data = {
                "model": self,
                "type": "constant-mean",
                "variance_model": self.variance_model,
                "endpoint_correction": self.endpoint_correction,
            }
            if log_p_hat is not None:
                save_data["transformed_parameters"] = log_p_hat
            if log_p_hat_cov is not None:
                save_data["transformed_parameter_covariance"] = log_p_hat_cov
            if training_simulation_data is not None:
                save_data["simulation_data"] = training_simulation_data
            if training_reflectance_data is not None:
                save_data["reflectance_data"] = training_reflectance_data

            pickle.dump(save_data, f)
