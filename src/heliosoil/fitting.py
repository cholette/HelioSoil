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
from numpy import radians as rad
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import lsq_linear, minimize_scalar, minimize
import numdifftools as ndt
import pickle


class CommonFittingMethods:
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

    def _compute_variance_of_measurements(self, sigma_dep, simulation_inputs, reflectance_data=None):
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

            nom_ref_anchor = _nominal_reflectance_anchor(reflectance_data, f, self.helios.nominal_reflectance)
            b = nom_ref_anchor * self.helios.inc_ref_factor[f]  # fixed for fitting experiments at reflectometer incidence angle
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

            if reflectance_data is None:
                meas_sig = np.zeros(self.helios.tilt[f].shape).transpose()
            else:
                meas_sig = reflectance_data.sigma_of_the_mean[f]

            c2t = np.cumsum(alpha**2 * gravitational_settling_factor(self.helios.tilt[f]) ** 2, axis=1).transpose()
            ind1 = pif[0:-1]
            ind2 = [x - 1 for x in pif[1::]]
            s2total[f] = s2_dep * b**2 * (c2t[ind2, :] - c2t[ind1, :]) + meas_sig[0:-1, :] ** 2 + meas_sig[1::, :] ** 2

        return s2total

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
        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs, reflectance_data)

        sim_in = simulation_inputs
        files = list(reflectance_data.times.keys())
        pi = reflectance_data.prediction_indices
        meas = reflectance_data.average
        NL = [reflectance_data.average[f].shape[0] for f in files]

        # define optimization objective function (negative log likelihood)
        sigma_dep = params[1]
        loglike = -0.5 * np.sum(NL) * np.log(2 * np.pi)
        self.update_model_parameters(params)
        self.predict_soiling_factor(simulation_inputs, reflectance_data=reflectance_data, verbose=False)
        sf = self.helios.soiling_factor  # soiling factor to be multiplied by clean reflectance

        # Compute variance in reflectance, not soiling factor
        s2total = self._compute_variance_of_measurements(sigma_dep, sim_in, reflectance_data=reflectance_data)

        for f in files:
            delta_r = np.diff(meas[f], axis=0)
            r0 = _nominal_reflectance_series(reflectance_data, f, self.helios.nominal_reflectance)  # nominal clean reflectance
            rho_prediction = r0 * sf[f][:, pi[f]].transpose()
            mu_delta_r = np.diff(rho_prediction, axis=0)
            loglike += np.sum(-0.5 * np.log(s2total[f]) - (delta_r - mu_delta_r) ** 2 / (2 * s2total[f]))

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
