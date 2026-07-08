"""
Constant-mean soiling model extended with a horizontal, wind-driven impaction term.

The standard constant-mean model (heliosoil.base_models.ConstantMeanBase) accumulates
soiled area as a function of dust loading and the vertical projection of the mirror
only (cos(tilt)). This module adds a second mechanism: dust carried horizontally by
wind that impacts the mirror face, which depends on wind speed and the relative angle
between the wind direction and the mirror-normal azimuth. The windward and leeward
faces of a tilted mirror are given independent fitted impaction coefficients since
there is no a priori reason to expect them to be equal.

    delta_A_ij = alpha_j * [ cos(theta_ij) * (mu_tilde + eps_j)
                           + U_j * ( p_windward_ij * (omega_windward + eps_gamma_j)
                                   + p_leeward_ij  * (omega_leeward  + eps_gamma_j) ) ]

where p_windward = sin(theta) * max(0, cos(Delta_gamma)),
      p_leeward  = sin(theta) * max(0, -cos(Delta_gamma)),
      Delta_gamma = azimuth (mirror normal) - wind_direction (meteorological, "from").

Only one of p_windward/p_leeward is nonzero at any (heliostat, time), so a single
shared noise term eps_gamma ~ N(0, sigma_dep_gamma^2) is used for both faces.
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
                f"Cannot parse orientation from mirror name '{name}': expected a "
                f"leading 'O' + cardinal-direction token (e.g. 'ON_M1_T00')."
            )
        cardinal = token[1:]
        try:
            azimuths[i] = cardinal_to_azimuth(cardinal)
        except ValueError as e:
            raise ValueError(f"Cannot parse orientation from mirror name '{name}': {e}") from e
    return azimuths


class ConstantMeanWindBase(smb.ConstantMeanBase):
    """
    Constant-mean deposition model with an added horizontal wind-driven impaction term.
    See module docstring for the governing equation.
    """

    def __init__(self):
        # Explicit (non-cooperative) call: a bare super().__init__() would resolve
        # via the instance's MRO and land on ConstantMeanDeposition.__init__ (which
        # requires file_params) when this runs as part of ConstantMeanWindDeposition.
        smb.ConstantMeanBase.__init__(self)
        self.omega_windward = None
        self.omega_leeward = None
        self.sigma_dep_gamma = None

    def import_site_data_and_constants(self, file_params, verbose=True):
        super().import_site_data_and_constants(file_params, verbose=verbose)
        table = pd.read_excel(file_params, index_col="Parameter")
        try:
            self.omega_windward = float(table.loc["omega_windward"].Value)
        except Exception:
            self.omega_windward = None
            _print_if(f"No omega_windward model defined in {file_params}.", verbose)
        try:
            self.omega_leeward = float(table.loc["omega_leeward"].Value)
        except Exception:
            self.omega_leeward = None
            _print_if(f"No omega_leeward model defined in {file_params}.", verbose)
        try:
            self.sigma_dep_gamma = float(table.loc["sigma_dep_gamma"].Value)
        except Exception:
            self.sigma_dep_gamma = None
            _print_if(f"No sigma_dep_gamma model defined in {file_params}.", verbose)

    def calculate_delta_soiled_area(
        self,
        simulation_inputs,
        mu_tilde=None,
        sigma_dep=None,
        omega_windward=None,
        omega_leeward=None,
        sigma_dep_gamma=None,
        verbose=True,
    ):
        _print_if("Calculating soil deposited in a timestep [m^2/m^2]", verbose)

        sim_in = simulation_inputs
        helios = self.helios
        dust = sim_in.dust

        if mu_tilde is None:
            mu_tilde = self.mu_tilde
        else:
            _print_if("Using supplied value for mu_tilde = " + str(mu_tilde), verbose)

        if sigma_dep is not None or self.sigma_dep is not None:
            sigma_dep = self.sigma_dep if sigma_dep is None else sigma_dep
            if sigma_dep is not None:
                _print_if("Using supplied value for sigma_dep = " + str(sigma_dep), verbose)

        if omega_windward is None:
            omega_windward = self.omega_windward
        else:
            _print_if("Using supplied value for omega_windward = " + str(omega_windward), verbose)

        if omega_leeward is None:
            omega_leeward = self.omega_leeward
        else:
            _print_if("Using supplied value for omega_leeward = " + str(omega_leeward), verbose)

        if sigma_dep_gamma is not None or self.sigma_dep_gamma is not None:
            sigma_dep_gamma = self.sigma_dep_gamma if sigma_dep_gamma is None else sigma_dep_gamma
            if sigma_dep_gamma is not None:
                _print_if(
                    "Using supplied value for sigma_dep_gamma = " + str(sigma_dep_gamma), verbose
                )

        files = list(sim_in.time.keys())
        for f in files:
            tilt = helios.tilt[f]

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

            try:
                attr = _parse_dust_str(sim_in.dust_type[f])
                den = getattr(dust, attr)
            except Exception:
                raise ValueError(
                    "Dust measurement = "
                    + sim_in.dust_type[f]
                    + " not present in dust class. Use dust_type="
                    + sim_in.dust_type[f]
                    + " option when initializing the model"
                )

            alpha = sim_in.dust_concentration[f] / den[f]

            p_windward, p_leeward = wind_projection_factors(tilt, azimuth, wind_dir)

            wind_term = wind_speed[None, :] * (
                p_windward * omega_windward + p_leeward * omega_leeward
            )
            helios.delta_soiled_area[f] = alpha[None, :] * (cosd(tilt) * mu_tilde + wind_term)

            if sigma_dep is not None or sigma_dep_gamma is not None:
                s2_dep = sigma_dep**2 if sigma_dep is not None else 0.0
                s2_gamma = sigma_dep_gamma**2 if sigma_dep_gamma is not None else 0.0
                dsav = alpha**2 * (
                    s2_dep * cosd(tilt) ** 2
                    + s2_gamma * wind_speed[None, :] ** 2 * (p_windward + p_leeward) ** 2
                )
                helios.delta_soiled_area_variance[f] = dsav

        self.helios = helios

    def random_delta_soiled_area(
        self,
        simulation_inputs,
        mu_tilde=None,
        sigma_dep=None,
        omega_windward=None,
        omega_leeward=None,
        sigma_dep_gamma=None,
        verbose=True,
    ):
        """
        Simulates the delta soiled area with randomness in the deposition velocity and
        horizontal impaction rate. The airborne dust loading is treated as a constant.

        Overrides ConstantMeanBase.random_delta_soiled_area, which calls
        calculate_delta_soiled_area positionally as (sim_in, mu_tilde, sigma_dep,
        verbose) -- on this class that would silently pass `verbose` into the
        omega_windward slot instead.
        """
        self.calculate_delta_soiled_area(
            simulation_inputs,
            mu_tilde=mu_tilde,
            sigma_dep=sigma_dep,
            omega_windward=omega_windward,
            omega_leeward=omega_leeward,
            sigma_dep_gamma=sigma_dep_gamma,
            verbose=verbose,
        )
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
    machinery as heliosoil.fitting.ConstantMeanDeposition but with a 5-parameter
    vector [mu_tilde, omega_windward, omega_leeward, sigma_dep, sigma_dep_gamma].
    """

    # Parameter vector layout: which entries are log-transformed (must be positive)
    # vs. left in linear space (omega_windward/omega_leeward may be ~0 or negative).
    _LOG_TRANSFORM = np.array([True, False, False, True, True])
    _PARAM_NAMES = ["mu_tilde", "omega_windward", "omega_leeward", "sigma_dep", "sigma_dep_gamma"]

    def __init__(self, file_params):
        ConstantMeanWindBase.__init__(self)
        self.import_site_data_and_constants(file_params)
        table = pd.read_excel(file_params, index_col="Parameter")
        self.helios.nominal_reflectance = float(table.loc["nominal_reflectance"].Value)

    def helios_angles(
        self,
        simulation_inputs,
        reflectance_data,
        verbose=True,
        second_surface=True,
        orientations=None,
    ):
        """
        Sets helios.tilt (as in ConstantMeanDeposition.helios_angles) and additionally
        populates helios.azimuth[f], shape (N_helios, N_times), from mirror-name
        parsing or an explicit override.

        Args:
            orientations: optional override for the mirror-normal azimuths [deg].
                - None (default): parsed from reflectance_data.mirror_names[f] via
                  parse_orientation_names (expects names like 'ON_M1_T00').
                - dict[file -> array-like of length N_helios]: explicit azimuths per file.
                - array-like of length N_helios: same azimuths applied to every file.
        """
        ConstantMeanDeposition.helios_angles(
            self,
            simulation_inputs,
            reflectance_data,
            verbose=verbose,
            second_surface=second_surface,
        )

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
                raise ValueError(
                    f"Number of orientations ({az.shape[0]}) does not match the number "
                    f"of heliostats ({N_helios}) for file {f}."
                )

            helios.azimuth[f] = np.tile(az[:, None], (1, N_times))

        self.helios = helios

    def predict_soiling_factor(
        self,
        simulation_inputs,
        rho0=None,
        mu_tilde=None,
        sigma_dep=None,
        omega_windward=None,
        omega_leeward=None,
        sigma_dep_gamma=None,
        verbose=True,
    ):
        sim_in = simulation_inputs
        self.calculate_delta_soiled_area(
            sim_in,
            mu_tilde=mu_tilde,
            sigma_dep=sigma_dep,
            omega_windward=omega_windward,
            omega_leeward=omega_leeward,
            sigma_dep_gamma=sigma_dep_gamma,
            verbose=verbose,
        )
        self.compute_soiling_factor(rho0=rho0)

        if self.sigma_dep is not None or self.sigma_dep_gamma is not None:
            for f in self.helios.soiling_factor.keys():
                inc_factor = self.helios.inc_ref_factor[f]
                dsav = self.helios.delta_soiled_area_variance[f]
                self.helios.soiling_factor_prediction_variance[f] = inc_factor**2 * np.cumsum(
                    dsav, axis=1
                )
        else:
            self.helios.soiling_factor_prediction_variance = {}

    def update_model_parameters(self, x):
        if isinstance(x, (list, np.ndarray)):
            self.mu_tilde = x[0]
            if len(x) > 1:
                self.omega_windward = x[1]
                self.omega_leeward = x[2]
            if len(x) > 3:
                self.sigma_dep = x[3]
                self.sigma_dep_gamma = x[4]
        else:
            self.mu_tilde = x

    def _negative_log_likelihood(self, params, simulation_inputs, reflectance_data):
        # Identical to CommonFittingMethods._negative_log_likelihood except that
        # sigma_dep is read from params[3], matching this class's 5-parameter
        # layout [mu_tilde, omega_windward, omega_leeward, sigma_dep, sigma_dep_gamma]
        # instead of the base class's 2-parameter [x, sigma_dep] layout.
        _check_keys(simulation_inputs, reflectance_data)

        sim_in = simulation_inputs
        files = list(reflectance_data.times.keys())
        pi = reflectance_data.prediction_indices
        meas = reflectance_data.average
        r0 = self.helios.nominal_reflectance
        NL = [reflectance_data.average[f].shape[0] for f in files]

        sigma_dep = params[3]
        loglike = -0.5 * np.sum(NL) * np.log(2 * np.pi)
        self.update_model_parameters(params)
        self.predict_soiling_factor(simulation_inputs, rho0=reflectance_data.rho0, verbose=False)
        sf = self.helios.soiling_factor

        s2total = self._compute_variance_of_measurements(
            sigma_dep, sim_in, reflectance_data=reflectance_data
        )

        for f in files:
            delta_r = np.diff(meas[f], axis=0)
            rho_prediction = r0 * sf[f][:, pi[f]].transpose()
            mu_delta_r = np.diff(rho_prediction, axis=0)
            loglike += np.sum(
                -0.5 * np.log(s2total[f]) - (delta_r - mu_delta_r) ** 2 / (2 * s2total[f])
            )

        return -loglike

    def _compute_variance_of_measurements(
        self, sigma_dep, simulation_inputs, reflectance_data=None
    ):
        # `sigma_dep` is passed positionally by the shared likelihood machinery
        # (heliosoil.fitting.CommonFittingMethods); sigma_dep_gamma is read from
        # self, which update_model_parameters keeps in sync immediately before
        # this is called.
        _check_keys(simulation_inputs, reflectance_data)

        sim_in = simulation_inputs
        files = list(sim_in.time.keys())

        s2_dep = sigma_dep**2 if sigma_dep is not None else 0.0
        s2_gamma = self.sigma_dep_gamma**2 if self.sigma_dep_gamma is not None else 0.0
        s2total = dict.fromkeys(files)
        for f in files:
            if reflectance_data is None:
                pif = range(0, len(sim_in.time[f]))
            else:
                pif = reflectance_data.prediction_indices[f]

            b = self.helios.nominal_reflectance * self.helios.inc_ref_factor[f]

            try:
                attr = _parse_dust_str(sim_in.dust_type[f])
                den = getattr(sim_in.dust, attr)
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

            tilt = self.helios.tilt[f]
            azimuth = self.helios.azimuth[f]
            wind_dir = sim_in.wind_direction[f]
            wind_speed = sim_in.wind_speed[f]
            p_windward, p_leeward = wind_projection_factors(tilt, azimuth, wind_dir)

            c2t_dep = np.cumsum(alpha**2 * cosd(tilt) ** 2, axis=1).transpose()
            c2t_gamma = np.cumsum(
                alpha**2 * wind_speed[None, :] ** 2 * (p_windward + p_leeward) ** 2, axis=1
            ).transpose()

            ind1 = pif[0:-1]
            ind2 = [x - 1 for x in pif[1::]]
            s2total[f] = (
                b**2
                * (
                    s2_dep * (c2t_dep[ind2, :] - c2t_dep[ind1, :])
                    + s2_gamma * (c2t_gamma[ind2, :] - c2t_gamma[ind1, :])
                )
                + meas_sig[0:-1, :] ** 2
                + meas_sig[1::, :] ** 2
            )

        return s2total

    def transform_scale(self, x, likelihood_hessian=None, direction="inverse"):
        if isinstance(x, (np.ndarray, list)):
            x = np.array(x, dtype=float)
            mask = self._LOG_TRANSFORM
            if direction == "inverse":
                z = np.where(mask, np.exp(x), x)
            elif direction == "forward":
                # np.where evaluates both branches elementwise, so np.log(x) would
                # otherwise be computed (and warn) even at unmasked entries, which
                # may legitimately be zero or negative (omega_windward/leeward).
                safe_x = np.where(mask, x, 1.0)
                z = np.where(mask, np.log(safe_x), x)
            else:
                raise ValueError("Transformation direction not recognized.")

            if not isinstance(likelihood_hessian, np.ndarray):
                return z

            # Jacobian of the (partially log) reparameterization.
            jac_diag = np.where(mask, np.exp(x), 1.0)
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
        _check_keys(simulation_inputs, reflectance_data)

        if x0 is None:
            _print_if(
                "Getting initial mu_tilde/sigma_dep guess via the plain constant-mean "
                "fit (omega_windward = omega_leeward = 0) ...",
                verbose,
            )
            omega_windward_saved, omega_leeward_saved = self.omega_windward, self.omega_leeward
            self.omega_windward, self.omega_leeward = 0.0, 0.0
            p0, sse = self.fit_least_squares(simulation_inputs, reflectance_data, verbose=False)
            self.omega_windward, self.omega_leeward = omega_windward_saved, omega_leeward_saved

            def nloglike1D(s):
                return self._negative_log_likelihood(
                    [p0, 0.0, 0.0, s, s], simulation_inputs, reflectance_data
                )

            s0 = minimize_scalar(nloglike1D, bounds=(smb.tol, sse), method="Bounded")
            x0 = np.array([p0, 0.0, 0.0, s0.x, s0.x])
            _print_if("x0 = " + str(x0), verbose)

        _print_if("Getting MLE estimates ... ", verbose)
        y, y_cov = CommonFittingMethods.fit_mle(
            self,
            simulation_inputs,
            reflectance_data,
            verbose=False,
            x0=x0,
            transform_to_original_scale=False,
            **optim_kwargs,
        )
        H_log = np.linalg.inv(y_cov)

        _print_if("========== MLE Estimates ======== ", verbose)
        if transform_to_original_scale:
            x_hat, H = self.transform_scale(y, H_log)
            x_hat_cov = np.linalg.inv(H)
        else:
            x_hat = y
            x_hat_cov = y_cov

        s = np.sqrt(np.diag(x_hat_cov))
        x_ci = x_hat + 1.96 * s * np.array([[-1], [1]])
        for i, name in enumerate(self._PARAM_NAMES):
            label = (
                name
                if transform_to_original_scale
                else f"log({name})" if self._LOG_TRANSFORM[i] else name
            )
            _print_if(f"{label} = {x_hat[i]:.3e}", verbose)
            _print_if(
                f"95% confidence interval for {label}: [{x_ci[0, i]:.3e}, {x_ci[1, i]:.3e}]",
                verbose,
            )

        return x_hat, x_hat_cov

    def save(
        self,
        file_name,
        log_p_hat=None,
        log_p_hat_cov=None,
        training_simulation_data=None,
        training_reflectance_data=None,
    ):
        with open(file_name, "wb") as f:
            save_data = {"model": self, "type": "constant-mean-wind"}
            if log_p_hat is not None:
                save_data["transformed_parameters"] = log_p_hat
            if log_p_hat_cov is not None:
                save_data["transformed_parameter_covariance"] = log_p_hat_cov
            if training_simulation_data is not None:
                save_data["simulation_data"] = training_simulation_data
            if training_reflectance_data is not None:
                save_data["reflectance_data"] = training_reflectance_data

            pickle.dump(save_data, f)
