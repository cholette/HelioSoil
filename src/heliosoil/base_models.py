import numpy as np
from numpy import matlib
from numpy import radians as rad
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import copy
from sklearn.cluster import KMeans
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple
from textwrap import dedent
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
import pytz
from pysolar import solar, radiation
import json
from scipy.interpolate import RegularGridInterpolator
from heliosoil.utilities import (
    _print_if,
    _ensure_list,
    _extinction_function,
    _same_ext_coeff,
    _import_option_helper,
    _parse_dust_str,
    _to_dict_of_lists,
    get_project_root,
)

tol = np.finfo(float).eps  # machine floating point precision


class SoilingBase:
    def __init__(self):
        """
        Initializes the base model class with parameters from a file.

        Args:
            file_params (str): Path to the Excel file containing the model parameters.

        Attributes:
            latitude (float): Latitude of the site in degrees.
            longitude (float): Longitude of the site in degrees.
            timezone_offset (float): Timezone offset from GMT in hours.
            hrz0 (float): Site roughness height ratio.
            loss_model (str): Either "geometry" or "mie" to specify the loss model.
            constants (constants): An instance of the constants class.
            helios (helios): An instance of the helios class.
            sigma_dep (float): Standard deviation of the deposition velocity.
            Ra (float): Aerodynamic resistance.
        """
        self.latitude = None  # latitude in degrees of site
        self.longitude = None  # longitude in degrees of site
        self.timezone_offset = None  # [hrs from GMT] timezone of site
        self.constants = Constants()  # a subclass for constant
        self.helios = Heliostats()  # a subclass containing information about the heliostats
        self.sigma_dep = None  # standard deviation for deposition velocity
        self.loss_model = None  # either "geometry" or "mie"

    def import_site_data_and_constants(self, file_params, verbose=True):

        _print_if(f"\nLoading data from {file_params} ... ", verbose)
        table = pd.read_excel(file_params, index_col="Parameter")

        # optional parameter imports
        try:
            self.latitude = float(table.loc["latitude"].Value)  # latitude in degrees of site
            self.longitude = float(table.loc["longitude"].Value)  # longitude in degrees of site
            self.timezone_offset = float(
                table.loc["timezone_offset"].Value
            )  # [hrs from GMT] timezone of site
        except Exception:
            _print_if(
                dedent(
                    f"""\
            You are missing at least one of (lat,lon,timezone_offset) in:
            {file_params}
            Field performance cannot be simulated until all of these are defined. """
                ),
                verbose,
            )
            self.latitude = None
            self.longitude = None
            self.timezone_offset = None

        self.constants.import_constants(file_params, verbose=verbose)


class PhysicalBase(SoilingBase):
    def __init__(self):
        super().__init__()
        self.hrz0 = None  # [-] site roughness height ratio

    def import_site_data_and_constants(self, file_params, verbose=True):
        super().import_site_data_and_constants(file_params)
        table = pd.read_excel(file_params, index_col="Parameter")

        try:
            self.loss_model = table.loc["loss_model"].Value  # either "geometry" or "mie"
        except Exception:
            _print_if(
                f"No loss model defined in {file_params}. You will need to define this before simulating",
                verbose,
            )

        try:
            self.hrz0 = float(table.loc["hr_z0"].Value)  # [-] site roughness height ratio
        except Exception:
            _print_if(
                f"No hrz0 model defined in {file_params}. You will need to define this before simulating",
                verbose,
            )

    def deposition_velocity(
        self, dust, wind_speed=None, air_temp=None, hrz0=None, verbose=True, Ra=True
    ):
        dust = dust

        # unpack constants
        constants = self.constants
        κ = constants.k_von_Karman
        A_slip = constants.A_slip
        λ_air_p = constants.air_lambda_p
        μ_air, ν_air = constants.air_mu, constants.air_nu
        ρ_air = constants.air_rho
        g = constants.g
        Re_Limit = constants.Re_Limit
        kB = constants.k_Boltzman
        β_EIM = constants.beta_EIM

        if hrz0 is None:  # hrz0 from constants file
            hrz0 = self.hrz0
            _print_if(
                "No value for hrz0 supplied. Using value in self.hrz0 = " + str(self.hrz0) + ".",
                verbose,
            )
        else:
            _print_if(
                "Value for hrz0 = " + str(hrz0) + " supplied. Value in self.hrz0 ignored.",
                verbose,
            )

        # N_sims = sim_in.N_simulations
        # _print_if("Calculating deposition velocity for each of the "+str(N_sims)+" simulations",verbose)

        D_meters = dust.D[0] * 1e-6  # µm --> m
        Ntimes = len(wind_speed)  # .shape[0]

        Cc = 1 + 2 * (λ_air_p / D_meters) * (
            A_slip[0] + A_slip[1] * np.exp(-A_slip[2] * D_meters / λ_air_p)
        )  # slip correction factor

        # computation of the gravitational settling velocity
        vg = (g * (D_meters**2) * Cc * (dust.rho[0])) / (18 * μ_air)
        # terminal velocity [m/s] if Re<0.1
        Re = ρ_air * vg * D_meters / μ_air  # Reynolds number for vg(Re<0.1)
        for ii in range(constants.N_iter):
            vnew = vg.copy()  # initialize vnew with vg
            Cd_g = 24 / Re
            Cd_g[Re > Re_Limit[0]] = (
                24
                / Re[Re > Re_Limit[0]]
                * (
                    1
                    + 3 / 16 * Re[Re > Re_Limit[0]]
                    + 9 / 160 * (Re[Re > Re_Limit[0]] ** 2) * np.log(2 * Re[Re > Re_Limit[0]])
                )
            )
            Cd_g[Re > Re_Limit[1]] = (
                24 / Re[Re > Re_Limit[1]] * (1 + 0.15 * Re[Re > Re_Limit[1]] ** 0.687)
            )
            Cd_g[Re > Re_Limit[2]] = 0.44
            vg_high_re = np.sqrt(4 * g * D_meters * Cc * dust.rho[0] / (3 * Cd_g * ρ_air))
            vnew[Re_Limit[0] >= Re] = vg_high_re[
                Re_Limit[0] >= Re
            ]  # replace vg with vg_high_re for Re>Re_Limit[0]
            if max(abs(vnew - vg) / vnew) < constants.tol:
                vg = vnew
                break
            vg = vnew
            Re = ρ_air * vg * D_meters / μ_air
        if ii == constants.N_iter:
            _print_if(
                "Max iter reached in Reynolds calculation for gravitational settling velocity",
                verbose,
            )

        # computation of the settling velocity due to inertia and diffusion
        u_friction = κ * wind_speed / np.log(hrz0)  # [m/s] friction velocity
        diffusivity = (
            kB
            / (3 * np.pi * μ_air)
            * np.transpose(matlib.repmat(air_temp + 273.15, len(D_meters), 1))
            * matlib.repmat(Cc / D_meters, Ntimes, 1)
        )  # [m^2/s] brownian diffusivity (Stokes-Einstein expression)
        Schmidt_number = ν_air / diffusivity  # Schmidt number
        Stokes_number = (
            np.transpose(matlib.repmat((u_friction**2), len(D_meters), 1)) * vg / ν_air / g
        )  # Stokes number
        Cd_momentum = κ**2 / ((np.log(hrz0)) ** 2)  # drag coefficient for momentum
        E_brownian = Schmidt_number ** (-2 / 3)  # Brownian factor
        E_impaction = (Stokes_number**β_EIM) / (
            constants.alpha_EIM + Stokes_number**β_EIM
        )  # Impaction factor (Giorgi, 1986)
        E_interception = 0  # Interception factor (=0 in this model)
        R1 = np.exp(
            -np.sqrt(Stokes_number)
        )  # 'stick' factor for boundary layer resistance computation
        R1[R1 <= tol] = tol  # to avoid division by 0
        if Ra:
            aerodynamic_resistance = 1 / (Cd_momentum * wind_speed)
            _print_if("Aerodynamic resistance is considered", verbose)  # [s/m]
        elif not Ra:
            aerodynamic_resistance = 0
            _print_if("Aerodynamic resistance is neglected", verbose)
        else:
            _print_if("Choose whether or not considering the aerodynamic resistance", verbose)

        boundary_layer_resistance = 1 / (
            constants.eps0
            * np.transpose(matlib.repmat((u_friction), len(D_meters), 1))
            * R1
            * (E_brownian + E_impaction + E_interception)
        )  # [s/m]

        # Rt = np.transpose(matlib.repmat(aerodynamic_resistance,len(D_meters),1))+boundary_layer_resistance

        vt = 1 / (
            np.transpose(matlib.repmat(aerodynamic_resistance, len(D_meters), 1))
            + boundary_layer_resistance
        )  # [m/s]

        vz = (vg + vt).transpose()  # [m/s]

        return (aerodynamic_resistance, boundary_layer_resistance, vg, vt, vz)

    def deposition_flux(self, simulation_inputs, hrz0=None, verbose=True, Ra=True):
        sim_in = simulation_inputs
        helios = self.helios
        dust = sim_in.dust

        # unpack constants
        constants = self.constants
        κ = constants.k_von_Karman
        A_slip = constants.A_slip
        λ_air_p = constants.air_lambda_p
        μ_air, ν_air = constants.air_mu, constants.air_nu
        ρ_air = constants.air_rho
        g = constants.g
        Re_Limit = constants.Re_Limit
        kB = constants.k_Boltzman
        β_EIM = constants.beta_EIM

        if hrz0 is None:  # hrz0 from constants file
            hrz0 = self.hrz0
            _print_if(
                "No value for hrz0 supplied. Using value in self.hrz0 = " + str(self.hrz0) + ".",
                verbose,
            )
        else:
            _print_if(
                "Value for hrz0 = " + str(hrz0) + " supplied. Value in self.hrz0 ignored.",
                verbose,
            )

        N_sims = sim_in.N_simulations
        _print_if(
            "Calculating deposition velocity for each of the " + str(N_sims) + " simulations",
            verbose,
        )

        files = list(sim_in.wind_speed.keys())
        for f in list(files):
            D_meters = dust.D[f] * 1e-6  # µm --> m
            Ntimes = len(sim_in.wind_speed[f])  # .shape[0]
            Nhelios = helios.tilt[f].shape[0]
            Nd = D_meters.shape[0]

            Cc = 1 + 2 * (λ_air_p / D_meters) * (
                A_slip[0] + A_slip[1] * np.exp(-A_slip[2] * D_meters / λ_air_p)
            )  # slip correction factor

            # computation of the gravitational settling velocity
            vg = (g * (D_meters**2) * Cc * (dust.rho[f])) / (18 * μ_air)
            # terminal velocity [m/s] if Re<0.1
            Re = ρ_air * vg * D_meters / μ_air  # Reynolds number for vg(Re<0.1)
            for ii in range(constants.N_iter):
                vnew = vg.copy()  # initialize vnew with vg
                Cd_g = 24 / Re
                Cd_g[Re > Re_Limit[0]] = (
                    24
                    / Re[Re > Re_Limit[0]]
                    * (
                        1
                        + 3 / 16 * Re[Re > Re_Limit[0]]
                        + 9 / 160 * (Re[Re > Re_Limit[0]] ** 2) * np.log(2 * Re[Re > Re_Limit[0]])
                    )
                )
                Cd_g[Re > Re_Limit[1]] = (
                    24 / Re[Re > Re_Limit[1]] * (1 + 0.15 * Re[Re > Re_Limit[1]] ** 0.687)
                )
                Cd_g[Re > Re_Limit[2]] = 0.44
                vg_high_re = np.sqrt(4 * g * D_meters * Cc * dust.rho[f] / (3 * Cd_g * ρ_air))
                vnew[Re_Limit[0] >= Re] = vg_high_re[
                    Re_Limit[0] >= Re
                ]  # replace vg with vg_high_re for Re>Re_Limit[0]
                if max(abs(vnew - vg) / vnew) < constants.tol:
                    vg = vnew
                    break
                vg = vnew
                Re = ρ_air * vg * D_meters / μ_air
            if ii == constants.N_iter:
                _print_if(
                    "Max iter reached in Reynolds calculation for gravitational settling velocity",
                    verbose,
                )

            # computation of the settling velocity due to inertia and diffusion
            u_friction = κ * sim_in.wind_speed[f] / np.log(hrz0)  # [m/s] friction velocity
            diffusivity = (
                kB
                / (3 * np.pi * μ_air)
                * np.transpose(matlib.repmat(sim_in.air_temp[f] + 273.15, len(D_meters), 1))
                * matlib.repmat(Cc / D_meters, Ntimes, 1)
            )  # [m^2/s] brownian diffusivity (Stokes-Einstein expression)
            Schmidt_number = ν_air / diffusivity  # Schmidt number
            Stokes_number = (
                np.transpose(matlib.repmat((u_friction**2), len(D_meters), 1)) * vg / ν_air / g
            )  # Stokes number
            Cd_momentum = κ**2 / ((np.log(hrz0)) ** 2)  # drag coefficient for momentum
            E_brownian = Schmidt_number ** (-2 / 3)  # Brownian factor
            E_impaction = (Stokes_number**β_EIM) / (
                constants.alpha_EIM + Stokes_number**β_EIM
            )  # Impaction factor (Giorgi, 1986)
            E_interception = 0  # Interception factor (=0 in this model)
            R1 = np.exp(
                -np.sqrt(Stokes_number)
            )  # 'stick' factor for boundary layer resistance computation
            R1[R1 <= tol] = tol  # to avoid division by 0
            if Ra:
                aerodynamic_resistance = 1 / (Cd_momentum * sim_in.wind_speed[f])
                _print_if("Aerodynamic resistance is considered", verbose)  # [s/m]
            elif not Ra:
                aerodynamic_resistance = 0
                _print_if("Aerodynamic resistance is neglected", verbose)
            else:
                _print_if(
                    "Choose whether or not considering the aerodynamic resistance",
                    verbose,
                )

            boundary_layer_resistance = 1 / (
                constants.eps0
                * np.transpose(matlib.repmat((u_friction), len(D_meters), 1))
                * R1
                * (E_brownian + E_impaction + E_interception)
            )  # [s/m]

            # Rt = np.transpose(matlib.repmat(aerodynamic_resistance,len(D_meters),1))+boundary_layer_resistance

            vt = 1 / (
                np.transpose(matlib.repmat(aerodynamic_resistance, len(D_meters), 1))
                + boundary_layer_resistance
            )  # [m/s]

            # computation of vertical deposition velocity
            vz = (vg + vt).transpose()  # [m/s]

            helios.pdfqN[f] = np.empty((Nhelios, Ntimes, Nd))
            for idx in range(helios.tilt[f].shape[0]):
                Fd = (
                    np.cos(rad(helios.tilt[f][idx, :])) * vz
                )  # Flux per unit concentration at each time, for each heliostat [m/s] (Eq. 28 in [1] without Cd)
                if Fd.min() < 0:
                    warnings.warn(
                        "Deposition velocity is negative (min value: "
                        + str(Fd.min())
                        + "). Setting negative components to zero."
                    )
                    Fd[Fd < 0] = 0
                helios.pdfqN[f][idx, :, :] = (
                    Fd.transpose() * dust.pdfN[f]
                )  # Dust flux pdf, i.e. [dq[particles/(s*m^2)]/dLog_{10}(D[µm]) ] deposited on 1m2.

        self.helios = helios

    def adhesion_removal(self, simulation_inputs, verbose=True):
        _print_if("Calculating adhesion/removal balance", verbose)
        helios = self.helios
        dust = simulation_inputs.dust
        dt = simulation_inputs.dt
        constants = self.constants
        g = constants.g
        files = list(simulation_inputs.time.keys())

        for f in files:
            D_meters = dust.D[f] * 1e-6  # Change to µm
            youngs_modulus_composite = (
                4
                / 3
                * (
                    (1 - dust.poisson[f] ** 2) / dust.youngs_modulus[f]
                    + (1 - helios.poisson**2) / helios.youngs_modulus
                )
                ** (-1)
            )
            # [N/m2] composite Young modulus
            hamaker_system = np.sqrt(
                dust.hamaker[f] * helios.hamaker
            )  # [J] system Hamaker constant (Israelachvili)
            work_adh = hamaker_system / (12 * np.pi * constants.D0**2)  # [J/m^2] work of adhesion
            radius_sep = (
                (3 * np.pi * work_adh * D_meters**2) / (8 * youngs_modulus_composite)
            ) ** (
                1 / 3
            )  # [m] contact radius at separation (JKR model)
            F_adhesion = (
                3 / 4 * np.pi * work_adh * D_meters
            )  # [N] van der Waals adhesion force (JKR model)
            F_gravity = dust.rho[f] * np.pi / 6 * g * D_meters**3  # [N] weight force

            if (
                helios.stow_tilt is None
            ):  # No common stow angle supplied. Need to use raw tilts to compute removal moments
                _print_if(
                    "  No common stow_tilt. Use values in helios.tilt to compute removal moments. This might take some time.",
                    verbose,
                )
                Nhelios = helios.tilt[f].shape[0]
                Ntimes = helios.tilt[f].shape[1]
                helios.pdfqN[f] = cumulative_trapezoid(
                    y=helios.pdfqN[f], dx=dt[f], axis=1, initial=0
                )  # Accumulate in time so that we ensure we remove all dust present on mirror if removal condition is satisfied at a particular time
                for h in range(Nhelios):
                    for k in range(Ntimes):
                        mom_removal = (
                            np.sin(rad(helios.tilt[f][h, k]))
                            * F_gravity
                            * np.sqrt((D_meters**2) / 4 - radius_sep**2)
                        )  # [Nm] removal moment exerted by gravity at each tilt for each diameter
                        mom_adhesion = (
                            F_adhesion + F_gravity * np.cos(rad(helios.tilt[f][h, k]))
                        ) * radius_sep  # [Nm] adhesion moment
                        helios.pdfqN[f][
                            h, k:, mom_adhesion < mom_removal
                        ] = 0  # ALL dust desposited at this diameter up to this point falls off
                        # if any(mom_adhesion<mom_removal):
                        #     _print_if("Some dust is removed",verbose)

                helios.pdfqN[f] = np.gradient(
                    helios.pdfqN[f], dt[f], axis=1
                )  # Take derivative so that pdfqN is the rate at wich dust is deposited at each diameter

            else:  # common stow angle at night for all heliostats. Assumes tilt at night is close to vertical at night.
                # Since the heliostats are stowed at a large tilt angle at night, we assume that any dust that falls off at this stow
                # is never deposited. This introduces a small error since the dust deposited during the day never affects the reflectance, but faster computation.
                _print_if(
                    "  Using common stow_tilt. Assumes all heliostats are stored at helios.stow_tilt at night.",
                    verbose,
                )
                mom_removal = (
                    np.sin(rad(helios.stow_tilt))
                    * F_gravity
                    * np.sqrt((D_meters**2) / 4 - radius_sep**2)
                )  # [Nm] removal moment exerted by gravity
                mom_adhesion = (
                    F_adhesion + F_gravity * np.cos(rad(helios.stow_tilt))
                ) * radius_sep  # [Nm] adhesion moment
                helios.pdfqN[f][
                    :, :, mom_adhesion < mom_removal
                ] = 0  # Remove this diameter from consideration

        self.helios = helios

    def calculate_delta_soiled_area(self, simulation_inputs, sigma_dep=None, verbose=True):

        # info and error checking
        _print_if("Calculating soil deposited in a timestep [m^2/m^2]", verbose)

        sim_in = simulation_inputs
        helios = self.helios
        dust = sim_in.dust
        extinction_weighting = helios.extinction_weighting

        files = list(sim_in.wind_speed.keys())
        for f in files:
            D_meters = dust.D[f] * 1e-6
            helios.delta_soiled_area[f] = np.empty(
                (helios.tilt[f].shape[0], helios.tilt[f].shape[1])
            )

            if sigma_dep is not None or self.sigma_dep is not None:
                helios.delta_soiled_area_variance[f] = np.empty(
                    (helios.tilt[f].shape[0], helios.tilt[f].shape[1])
                )

            # compute alpha
            try:
                attr = _parse_dust_str(sim_in.dust_type[f])
                den = getattr(dust, attr)  # dust.(sim_in.dust_type[f])
            except Exception:
                raise ValueError(
                    "Dust measurement = "
                    + sim_in.dust_type[f]
                    + " not present in dust class. Use dust_type="
                    + sim_in.dust_type[f]
                    + " option when loading the simulation data."
                )
            alpha = sim_in.dust_concentration[f] / den[f]

            # Compute the area coverage by dust at each time step
            N_helios = helios.tilt[f].shape[0]
            N_times = helios.tilt[f].shape[1]
            for ii in range(N_helios):
                for jj in range(N_times):

                    # if loss_model == 'geometry':
                    #     # The below two integrals are equivalent, but the version with the log10(D)
                    #     # as the independent variable is used due to the log spacing of the diameter grid
                    #     #
                    #     # helios.delta_soiled_area[f][ii,jj] = alpha[jj] * np.trapezoid(helios.pdfqN[f][ii,jj,:]*\
                    #     #     (np.pi/4*D_meters**2)*sim_in.dt[f]/dust.D[f]/np.log(10),dust.D[f])
                    #
                    #     helios.delta_soiled_area[f][ii,jj] = alpha[jj] * np.pi/4 *np.trapezoid(helios.pdfqN[f][ii,jj,:]*\
                    #         (D_meters**2)*sim_in.dt[f],np.log10(dust.D[f]))
                    # else: # loss_model == "mie"
                    helios.delta_soiled_area[f][ii, jj] = (
                        alpha[jj]
                        * np.pi
                        / 4
                        * np.trapezoid(
                            helios.pdfqN[f][ii, jj, :]
                            * (D_meters**2)
                            * sim_in.dt[f]
                            * extinction_weighting[f][ii, :],
                            np.log10(dust.D[f]),
                        )
                    )  # pdfqN includes cos(tilt)

            # variance of noise for each measurement
            if sigma_dep is not None:
                theta = np.radians(self.helios.tilt[f])
                helios.delta_soiled_area_variance[f] = sigma_dep**2 * (
                    alpha**2 * np.cos(theta) ** 2
                )
                # sigma_dep**2*helios.inc_ref_factor[f]*np.cumsum(alpha**2*np.cos(theta)**2,axis=1)
            elif self.sigma_dep is not None:
                theta = np.radians(self.helios.tilt[f])
                helios.delta_soiled_area_variance[f] = self.sigma_dep**2 * (
                    alpha**2 * np.cos(theta) ** 2
                )

        self.helios = helios

    def plot_area_flux(
        self,
        sim_data,
        exp_idx,
        hel_id,
        air_temp,
        wind_speed,
        tilt=0.0,
        hrz0=None,
        constants=None,
        ax=None,
        Ra=True,
        verbose=True,
    ):

        dummy_sim = SimulationInputs()
        dummy_sim.dust = Dust()

        for att_name in sim_data.dust.__dict__.keys():
            val = {0: getattr(sim_data.dust, att_name)[exp_idx]}
            setattr(dummy_sim.dust, att_name, val)

        # dummy_sim.dust.import_dust(dust_file,verbose=False,dust_measurement_types="PM10")
        dummy_sim.air_temp = {0: np.array([air_temp])}
        dummy_sim.wind_speed = {0: np.array([wind_speed])}
        dummy_sim.dt = {0: 1.0}
        dummy_sim.dust_type = {0: "PM10"}  # this doesn't matter for this function
        dummy_sim.dust_concentration = {0: np.array([dummy_sim.dust.PM10[0]])}  # makes alpha = 1
        dummy_sim.N_simulations = 1

        if self.loss_model == "mie":
            dummy_sim.source_normalized_intensity = {
                0: sim_data.source_normalized_intensity[exp_idx]
            }
            dummy_sim.source_wavelength = {0: sim_data.source_wavelength[exp_idx]}
            acceptance_angle = self.helios.acceptance_angles[exp_idx][hel_id]
            _print_if("Loss model is " "mie" " ", verbose)
        else:
            _print_if(
                "Loss model is " "geometry" ". Extinction weights are unity for all diameters.",
                verbose,
            )
            acceptance_angle = np.nan

        dummy_model = copy.deepcopy(self)
        dummy_model.helios = Heliostats()
        dummy_model.helios.tilt = {0: np.array([[tilt]])}
        dummy_model.sigma_dep = None
        dummy_model.loss_model = self.loss_model
        # dummy_model.helios.acceptance_angles = [acceptance_angle]
        # dummy_model.helios.extinction_weighting = {0:np.atleast_2d(self.helios.extinction_weighting[exp_idx][0,:])}
        dummy_model.helios.extinction_weighting = {
            0: np.atleast_2d(self.helios.extinction_weighting[exp_idx][hel_id, :])
        }

        fmt = "Setting constants.{0:s} to {1:s} (was {2:s})"
        if constants is not None:
            for kk in constants.keys():
                temp = str(getattr(dummy_model.constants, kk))
                print(fmt.format(str(kk), str(constants[kk]), temp))
                setattr(dummy_model.constants, kk, constants[kk])

        if hrz0 is None:
            hrz0 = dummy_model.hrz0
            dummy_model.deposition_flux(dummy_sim, Ra=Ra)
        else:
            dummy_model.deposition_flux(dummy_sim, hrz0=hrz0, Ra=Ra)

        dummy_model.calculate_delta_soiled_area(dummy_sim)

        if ax is None:
            _, ax1 = plt.subplots()
        else:
            ax1 = ax

        title = f"""
            Area loss rate for given dust distribution at acceptance angle {acceptance_angle * 1e3:.2f} mrad,
            wind_speed= {wind_speed:.1f} m/s, air_temperature={air_temp:.1f} C
            (total area loss is {dummy_model.helios.delta_soiled_area[0][0, 0]:.2e} m$^2$/(s$\\cdot$m$^2$))
        """
        area_loss_rate = (
            dummy_model.helios.pdfqN[0][0, 0, :]
            * np.pi
            / 4
            * dummy_sim.dust.D[0] ** 2
            * 1e-12
            * dummy_model.helios.extinction_weighting[0][0, :]
        )
        ax1.plot(dummy_sim.dust.D[0], area_loss_rate)
        ax1.set_title(
            title.format(
                wind_speed,
                air_temp,
            )
        )
        ax1.set_xlabel(r"D [$\mu$m]")
        ax1.set_ylabel(r"$\frac{dA [m^2/m^2/s] }{dLog(D \;[\mu m])}$", color="black", size=20)
        plt.xscale("log")
        ax1.set_xticks([0.001, 0.01, 0.1, 1, 2.5, 4, 10, 20, 100])


class ConstantMeanBase(SoilingBase):
    def __init__(self):
        super().__init__()
        self.mu_tilde = None

    def import_site_data_and_constants(self, file_params, verbose=True):
        super().import_site_data_and_constants(file_params)
        table = pd.read_excel(file_params, index_col="Parameter")
        try:
            self.mu_tilde = float(table.loc["mu_tilde"].Value)  # [-] constant average deposition
        except Exception:
            self.mu_tilde = None
            _print_if(
                f"No mu_tilde model defined in {file_params}. You will need to define this before simulating",
                verbose,
            )
        try:
            self.sigma_dep = float(table.loc["sigma_dep"].Value)
        except Exception:
            self.sigma_dep = None
            _print_if(f"No sigma_dep model defined in {file_params}.", verbose)

    def calculate_delta_soiled_area(
        self, simulation_inputs, mu_tilde=None, sigma_dep=None, verbose=True
    ):

        _print_if("Calculating soil deposited in a timestep [m^2/m^2]", verbose)

        sim_in = simulation_inputs
        helios = self.helios
        dust = sim_in.dust

        if mu_tilde is None:  # use value in self
            mu_tilde = self.mu_tilde
        else:
            mu_tilde = mu_tilde
            _print_if("Using supplied value for mu_tilde = " + str(mu_tilde), verbose)

        if sigma_dep is not None or self.sigma_dep is not None:
            if sigma_dep is None:  # use value in self
                sigma_dep = self.sigma_dep
            else:
                sigma_dep = sigma_dep
                _print_if("Using supplied value for sigma_dep = " + str(sigma_dep), verbose)

        files = list(sim_in.time.keys())
        for f in files:
            helios.delta_soiled_area[f] = np.empty(
                (helios.tilt[f].shape[0], helios.tilt[f].shape[1])
            )

            # compute alpha
            try:
                attr = _parse_dust_str(sim_in.dust_type[f])
                den = getattr(dust, attr)  # dust.(sim_in.dust_type[f])
            except Exception:
                raise ValueError(
                    "Dust measurement = "
                    + sim_in.dust_type[f]
                    + " not present in dust class. Use dust_type="
                    + sim_in.dust_type[f]
                    + " option when initializing the model"
                )

            alpha = sim_in.dust_concentration[f] / den[f]

            # Compute the area coverage by dust at each time step
            N_helios = helios.tilt[f].shape[0]
            N_times = helios.tilt[f].shape[1]
            for ii in range(N_helios):
                for jj in range(N_times):
                    helios.delta_soiled_area[f][ii, jj] = (
                        alpha[jj] * np.cos(rad(helios.tilt[f][ii, jj])) * mu_tilde
                    )

            # Predict confidence interval if sigma_dep is defined. Fixed tilt assumed in this class.
            if sigma_dep is not None:
                theta = np.radians(self.helios.tilt[f])
                inc_factor = self.helios.inc_ref_factor[f]
                dsav = sigma_dep**2 * (alpha**2 * np.cos(theta) ** 2)

                helios.delta_soiled_area_variance[f] = dsav
                self.helios.soiling_factor_prediction_variance[f] = np.cumsum(
                    inc_factor**2 * dsav, axis=1
                )

        self.helios = helios


@dataclass
class SimulationInputs:
    """
    Manage input data for soiling model simulations.

    Parses weather, dust, and source intensity data from Excel files for each experiment.

    Args:
        files (Optional[List[str]]): Paths to simulation input files.
        k_factors (Optional[List[float]]): Dust concentration multipliers per file.
        dust_type (Optional[List[str]]): Dust measurement types (e.g., 'TSP', 'PM10').
        verbose (bool): Whether to display progress messages.
    """

    files: List[Union[str, Path]] = field(default=None)
    k_factors: Optional[List[float]] = field(default=None)
    dust_type: Optional[List[str]] = field(
        default=None,
        metadata={"description": "Dust measurement type (TSP, PMX, PMX.Y)"},
    )
    verbose: bool = field(default=True)

    dt: Dict[int, float] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "simulation time step", "units": "seconds"},
    )
    time: Dict[int, pd.Series] = field(
        init=False,
        default_factory=dict,
        metadata={
            "description": "absolute time (taken from first entry)",
            "units": "n/a",
        },
    )
    time_diff: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "delta_time since start date", "units": "days"},
    )
    start_datetime: Dict[int, np.datetime64] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "start datetime of simulation", "units": "datetime64"},
    )
    end_datetime: Dict[int, np.datetime64] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "end datetime of simulation", "units": "datetime64"},
    )
    air_temp: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "air temperature", "units": "degC"},
    )
    wind_speed: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "wind speed", "units": "m/s"},
    )
    wind_speed_mov_avg: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "wind speed hourly moving average", "units": "m/s"},
    )
    wind_direction: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "wind direction", "units": "degrees"},
    )
    dust_concentration: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "PM10 or TSP concentration in air", "units": "µg/m³"},
    )
    dust_conc_mov_avg: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={
            "description": "hourly moving average of dust concentration",
            "units": "µg/m³",
        },
    )
    rain_intensity: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "rain intensity", "units": "mm/hr"},
    )
    dni: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "Direct Normal Irradiance", "units": "W/m^2"},
    )
    relative_humidity: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "relative humidity", "units": "%"},
    )
    source_normalized_intensity: Dict[int, Optional[np.ndarray]] = field(
        init=False,
        default_factory=dict,
        metadata={
            "description": "Source intensity (normalized to integrate to 1)",
            "units": "1/m^2/nm",
        },
    )
    source_wavelength: Dict[int, Optional[np.ndarray]] = field(
        init=False,
        default_factory=dict,
        metadata={
            "description": "source wavelengths corresponding to intensity",
            "units": "nm",
        },
    )
    dust: "Dust" = field(
        init=False,
        metadata={"description": "Dust properties per experiment", "units": "n/a"},
    )
    weather_variables: List[str] = field(
        init=False,
        default_factory=list,
        metadata={"description": "list of imported weather variables", "units": "n/a"},
    )
    N_simulations: int = field(
        init=False,
        default=0,
        metadata={"description": "number of simulations", "units": "count"},
    )
    k_factors_dict: Dict[int, float] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "k-factors per experiment", "units": "dimensionless"},
    )

    smallest_windspeed: float = field(
        default=1e-6,
        metadata={
            "description": "smallest wind speed to set zero values",
            "units": "m/s",
        },
    )

    def __post_init__(self) -> None:
        if self.files is not None:
            self.files = _ensure_list(self.files)
            self.dust_type = _import_option_helper(self.files, self.dust_type)
            self.N_simulations = len(self.files)

            # Prepare k_factors list
            if self.k_factors is None:
                k_list = [1.0] * self.N_simulations
            elif self.k_factors == "import":
                k_list = [
                    pd.read_excel(f, sheet_name="Dust", index_col="Parameter")
                    .loc["k_factor"]
                    .values[0]
                    for f in self.files
                ]
            else:
                k_list = _import_option_helper(self.files, self.k_factors)
                if len(k_list) != self.N_simulations:
                    raise ValueError("Please specify a k-factor for each weather file")

            self.k_factors_dict = {ii: k_list[ii] for ii in range(self.N_simulations)}

            # Import data
            self.import_weather()
            self.dust = Dust(self.files)
            self.dust.import_dust(verbose=self.verbose)
            self.import_source_intensity()

    def import_source_intensity(self) -> None:
        for ii, f in enumerate(self.files):
            xl = pd.ExcelFile(f)
            if "Source_Intensity" in xl.sheet_names:
                _print_if(f"Loading source (normalized) intensity from {f}", self.verbose)
                intensity = xl.parse("Source_Intensity")
                self.source_wavelength[ii] = intensity["Wavelength (nm)"].to_numpy()
                self.source_normalized_intensity[ii] = intensity[
                    "Source Intensity (W/m^2 nm)"
                ].to_numpy()
                norm = np.trapezoid(
                    y=self.source_normalized_intensity[ii], x=self.source_wavelength[ii]
                )
                self.source_normalized_intensity[ii] /= norm
            else:
                self.source_normalized_intensity[ii] = None
            xl.close()

    def import_weather(self) -> None:
        weather_variables_map = {
            "air_temp": ["airtemp", "temperature", "temp", "ambt", "t1"],
            "wind_speed": ["windspeed", "ws", "wind_speed"],
            "dni": ["dni", "directnormalirradiance"],
            "rain_intensity": ["rainintensity", "precipitation"],
            "relative_humidity": ["rh", "relativehumidity", "rhx"],
            "wind_direction": ["wd", "winddirection"],
        }
        dust_names = {
            "pm_tot": ["pm_tot", "pmtot", "pmt", "pm20"],
            "tsp": ["tsp"],
            "pm10": ["pm10"],
            "pm2p5": ["pm2_5", "pm2p5", "pm2.5"],
            "pm1": ["pm1"],
            "pm4": ["pm4"],
        }

        for ii, file in enumerate(self.files):
            weather = pd.read_excel(file, sheet_name="Weather")

            # Identify time column
            time_col = next(
                (
                    col
                    for col in weather.columns
                    if col.lower() in ["time", "timestamp", "date", "datetime", "date time"]
                ),
                None,
            )
            if time_col is None:
                raise ValueError(f"No time column found in file {file}.")

            # Parse time
            weather[time_col] = pd.to_datetime(weather[time_col]).dt.round("min")
            time = pd.to_datetime(weather[time_col])
            self.start_datetime[ii] = time.iloc[0]
            self.end_datetime[ii] = time.iloc[-1]

            _print_if(
                f"Importing site data from {file}, dust_type={self.dust_type[ii]}, length={(self.end_datetime[ii] - self.start_datetime[ii]).days} days",
                self.verbose,
            )

            self.time[ii] = time
            self.dt[ii] = (time.iloc[1] - time.iloc[0]).total_seconds()
            self.time_diff[ii] = (
                (self.time[ii].values - self.time[ii].values.astype("datetime64[D]"))
                .astype("timedelta64[h]")
                .astype("int")
            )

            # Load weather variables
            for (
                attr_name,
                column_names,
            ) in (
                weather_variables_map.items()
            ):  # Search for weather variables inside the weather file and save them to self
                for column in column_names:
                    if column in [col.lower() for col in weather.columns]:
                        (setattr(self, attr_name, {}) if not hasattr(self, attr_name) else None)
                        col_match = [col for col in weather.columns if col.lower() == column][0]
                        getattr(self, attr_name)[ii] = np.array(weather.loc[:, col_match])
                        _print_if(
                            f"Importing {col_match} data as {attr_name}...",
                            self.verbose,
                        )
                        if attr_name not in self.weather_variables:
                            self.weather_variables.append(attr_name)
                        break

            # Correct zero wind speeds
            if ii in self.wind_speed:
                idx_low = np.where(self.wind_speed[ii] == 0)[0]
                if len(idx_low) > 0:
                    self.wind_speed[ii][idx_low] = self.smallest_windspeed
                    _print_if(
                        f"Warning: zero windspeeds set to {self.smallest_windspeed}",
                        self.verbose,
                    )

            self.wind_speed_mov_avg[ii] = (
                pd.Series(self.wind_speed[ii])
                .rolling(window=int(60.0 / (self.dt[ii] / 60)), min_periods=1)
                .mean()
                .to_numpy()
            )

            # Dust concentration
            self.dust_concentration[ii] = (
                self.k_factors_dict[ii] * weather[self.dust_type[ii]].to_numpy()
            )
            if "dust_concentration" not in self.weather_variables:
                self.weather_variables.append("dust_concentration")

            # Load all dust measurements
            for (
                dust_key,
                dust_aliases,
            ) in dust_names.items():  # Load all dust concentration data inside weather file
                for alias in dust_aliases:
                    if alias in [col.lower() for col in weather.columns]:
                        col_match = [col for col in weather.columns if col.lower() == alias][0]
                        dust_value = np.array(weather.loc[:, col_match])
                        if not hasattr(self, dust_key.lower()):
                            setattr(self, dust_key.lower(), {})
                        getattr(self, dust_key.lower())[ii] = dust_value
                        _print_if(f"Importing {dust_key} data...", self.verbose)
                        if dust_key.lower() not in self.weather_variables:
                            self.weather_variables.append(dust_key.lower())
                        break

            self.dust_conc_mov_avg[ii] = (
                pd.Series(self.dust_concentration[ii])
                .rolling(window=int(60.0 / (self.dt[ii] / 60)), min_periods=1)
                .mean()
                .to_numpy()
            )

            if self.verbose:
                days = (self.end_datetime[ii] - self.start_datetime[ii]).days
                _print_if(f"Simulation length for {file}: {days} days", self.verbose)

    def get_experiment_subset(self, idx: Union[int, List[int]]) -> "SimulationInputs":
        copy_self = copy.deepcopy(self)
        idxs = [idx] if isinstance(idx, int) else list(idx)
        for attr in vars(copy_self):
            val = getattr(copy_self, attr)
            if isinstance(val, dict):
                for key in list(val.keys()):
                    if key not in idxs:
                        del val[key]
        return copy_self


@dataclass
class Dust:
    """
    Data class for dust properties loaded from experiment files.

    Attributes are keyed by experiment index.
    """

    files: Optional[List[Union[str, Path]]] = field(default=None)

    D: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "µm", "description": "Dust particle diameters"},
    )
    rho: Dict[int, float] = field(
        init=False,
        default_factory=dict,
        metadata={
            "units": "kg/m^3",
            "description": "Particle material density (assummed constant)",
        },
    )
    m: Dict[int, complex] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "-", "description": "Complex refractive index"},
    )
    pdfN: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={
            "units": "[1/m³]/log10([µm])",
            "description": "Number distribution dN/d(log10(D))",
        },
    )
    pdfM: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={
            "units": "[µg/m³]/dLog10(D[µm])",
            "description": "Mass distribution dm/dLog10(D)",
        },
    )
    pdfA: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={
            "units": "[m2/m³]/dLog10([µm])",
            "description": "Area distribution dA/dLog10(D)",
        },
    )
    hamaker: Dict[int, float] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "J", "description": "Hamaker constant of dust"},
    )
    poisson: Dict[int, float] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "-", "description": "Poisson's ratio of dust"},
    )
    youngs_modulus: Dict[int, float] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "Pa", "description": "Young's modulus of dust"},
    )
    PM10: Dict[int, float] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "µg/m³", "description": "PM10 concentration"},
    )
    TSP: Dict[int, float] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "µg/m³", "description": "TSP concentration"},
    )
    PMT: Dict[int, float] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "µg/m³", "description": "PMT concentration"},
    )
    Nd: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "1/cm³", "description": "Number concentration components"},
    )
    log10_mu: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "log10(µm)", "description": "Log10 of mean diameters"},
    )
    log10_sig: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "log10(µm)", "description": "Log10 of distribution widths"},
    )

    def import_dust(self, verbose=True, dust_measurement_type=None):

        _print_if("Importing dust properties for each experiment", verbose)
        experiment_files = self.files
        dust_measurement_type = _import_option_helper(experiment_files, dust_measurement_type)

        for ii, f in enumerate(experiment_files):
            table = pd.read_excel(f, sheet_name="Dust", index_col="Parameter")
            rhoii = float(table.loc["rho"].Value)
            self.rho[ii] = rhoii
            self.m[ii] = (
                table.loc["refractive_index_real_part"].Value
                - table.loc["refractive_index_imaginary_part"].Value * 1j
            )

            # definition of parameters to compute the dust size distribution
            diameter_grid_info = np.array(table.loc["D"].Value.split(";"))  # [µm]
            diameter_end_points = np.log10(diameter_grid_info[0:2].astype("float"))
            spacing = diameter_grid_info[2].astype("int")
            Dii = np.logspace(diameter_end_points[0], diameter_end_points[1], num=spacing)
            self.D[ii] = Dii

            if isinstance(
                table.loc["Nd"].Value, str
            ):  # if this is imported as a string, we need to split it.
                self.Nd[ii] = np.array(table.loc["Nd"].Value.split(";"), dtype=float)
                self.log10_mu[ii] = np.log10(
                    np.array(table.loc["mu"].Value.split(";"), dtype=float)
                )
                self.log10_sig[ii] = np.log10(
                    np.array(table.loc["sigma"].Value.split(";"), dtype=float)
                )
            elif isinstance(table.loc["Nd"].Value, float):  # handle single-component case
                self.Nd[ii] = np.array([table.loc["Nd"].Value])
                self.log10_mu[ii] = np.log10([np.array(table.loc["mu"].Value)])
                self.log10_sig[ii] = np.log10([np.array(table.loc["sigma"].Value)])
            else:
                raise ValueError(
                    "Format of dust distribution components is not recognized in file {0:s}".format(
                        f
                    )
                )

            # computation of the dust size distribution
            N_components = len(self.Nd[ii])
            nNd = np.zeros((len(Dii), N_components))
            for jj in range(N_components):
                Ndjj = self.Nd[ii][jj]
                lsjj = self.log10_sig[ii][jj]
                lmjj = self.log10_mu[ii][jj]
                nNd[:, jj] = (
                    Ndjj
                    / (np.sqrt(2 * np.pi) * lsjj)
                    * np.exp(-((np.log10(Dii) - lmjj) ** 2) / (2 * lsjj**2))
                )

            pdfNii = (
                np.sum(nNd, axis=1) * 1e6
            )  # pdfN (number) distribution dN[m^-3]/dLog10(D[µm]), 1e6 factor from { V(cm^3->m^3) 1e6 }
            self.pdfN[ii] = pdfNii
            self.pdfA[ii] = (
                pdfNii * (np.pi / 4 * Dii**2) * 1e-12
            )  # pdfA (area) dA[m^2/m^3]/dLog10(D[µm]), 1e-12 factor from { D^2(µm^2->m^2) 1e-12}
            self.pdfM[ii] = (
                pdfNii * (rhoii * np.pi / 6 * Dii**3) * 1e-9
            )  # pdfm (mass) dm[µg/m^3]/dLog10(D[µm]), 1e-9 factor from { D^3(µm^3->m^3) 1e-18 , m(kg->µg) 1e9}
            self.TSP[ii] = np.trapezoid(self.pdfM[ii], np.log10(Dii))
            self.PMT[ii] = self.TSP[ii]
            self.PM10[ii] = np.trapezoid(
                self.pdfM[ii][Dii <= 10], np.log10(Dii[Dii <= 10])
            )  # PM10 = np.trapezoid(self.pdfM[self.D<=10],dx=np.log10(self.D[self.D<=10]))

            self.hamaker[ii] = float(table.loc["hamaker_dust"].Value)
            self.poisson[ii] = float(table.loc["poisson_dust"].Value)
            self.youngs_modulus[ii] = float(table.loc["youngs_modulus_dust"].Value)

        # add dust measurements if they are PMX
        for dt in dust_measurement_type:
            if dt not in [
                None,
                "TSP",
                "PMT",
            ]:  # another concentration is of interest (possibly because we have PMX measurements)
                X = dt[2::]
                if len(X) in [1, 2]:  # integer, e.g. PM20
                    X = int(X)
                    att = "PM{0:d}".format(X)
                elif len(X) == 3:  # decimal, e.g. PM2.5
                    att = "PM" + "_".join(X.split("."))
                    X = float(X)

                new_meas = {f: None for f, _ in enumerate(experiment_files)}
                for ii, _ in enumerate(experiment_files):
                    new_meas[ii] = np.trapezoid(self.pdfM[ii][Dii <= X], np.log10(Dii[Dii <= X]))

                setattr(self, att, new_meas)
                _print_if(
                    "Added " + att + " attribute to dust class to all experiment dust classes",
                    verbose,
                )

    def plot_distributions(
        self, figsize: Tuple[float, float] = (5, 5)
    ) -> Tuple[plt.Figure, Any, List[Any]]:
        """
        Plot number and mass PDFs on a shared log-scale diameter axis.

        Args:
            figsize: Tuple of figure width and height in inches.

        Returns:
            fig: Figure containing the subplots.
            axes1: Array of primary axes (number PDFs).
            axes2: List of secondary axes (mass PDFs).
        """
        N = len(self.D)
        fig, axes1 = plt.subplots(nrows=N, sharex=True, squeeze=False, figsize=figsize)
        axes2: List[Any] = []

        for i in range(N):
            d = self.D[i]
            n_pdf = self.pdfN[i]
            m_pdf = self.pdfM[i]

            ax1 = axes1[i, 0]
            ax1.set_xscale("log")
            ax1.set_xlabel(r"Diameter $D$ [$\mu$m]")
            ax1.set_ylabel(r"dN/dlog$D$ [# m$^{-3}$]", color="tab:red")
            ax1.plot(d, n_pdf, color="tab:red")
            ax1.tick_params(axis="y", labelcolor="tab:red")
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.set_ylabel(r"dm/dlog$D$ [µg m$^{-3}$]", color="tab:blue")
            ax2.plot(d, m_pdf, color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            axes2.append(ax2)

        plt.tight_layout()
        fig.suptitle("Number and Mass PDFs", y=1.02)
        return fig, axes1, axes2

    def plot_area_distribution(self, figsize: Tuple[float, float] = (5, 5)) -> List[Any]:
        """
        Plot area PDF on a log-scale diameter axis.

        Args:
            figsize: Tuple of figure width and height in inches.

        Returns:
            axes: List of axes objects for each experiment.
        """
        N = len(self.D)
        fig, axes = plt.subplots(nrows=N, sharex=True, squeeze=False, figsize=figsize)

        for i in range(N):
            d = self.D[i]
            a_pdf = self.pdfA[i]
            ax = axes[i, 0]
            ax.set_xscale("log")
            ax.set_xlabel(r"Diameter $D$ [$\mu$m]")
            ax.set_ylabel(r"dA/dlog$D$ [m$^2$ m$^{-3}$]", color="black")
            ax.plot(d, a_pdf, color="black")
            ax.tick_params(axis="y", labelcolor="black")
            ax.set_title("Area PDF")
            ax.grid(True)

        plt.tight_layout()
        return [axes[i, 0] for i in range(N)]


@dataclass
class TruckParameters:
    """Default parameters for cleaning truck configuration."""

    # Cost parameters
    cost_water: float = field(
        default=0.87, metadata={"units": "$/kL", "description": "Cost of water"}
    )
    usage_water: float = field(
        default=0.4,
        metadata={
            "units": "L/m²",
            "description": "Water usage per square meter cleaned",
        },
    )
    cost_fuel: float = field(default=2.0, metadata={"units": "$/L", "description": "Cost of fuel"})
    usage_fuel: float = field(
        default=25.0,
        metadata={"units": "L/hour", "description": "Fuel consumption rate"},
    )
    salary_operator: float = field(
        default=80e3, metadata={"units": "$/year", "description": "Operator salary"}
    )
    cost_purchase: float = field(
        default=150e3, metadata={"units": "$/truck", "description": "Cost of truck"}
    )
    cost_maintenance: float = field(
        default=15e3,
        metadata={"units": "$/year", "description": "Annual maintenance cost"},
    )
    useful_life: float = field(
        default=10.0, metadata={"units": "years", "description": "Useful life of truck"}
    )
    # Velocities
    velocity_cleaning: float = field(
        default=2.0,
        metadata={"units": "km/h", "description": "Truck velocity during cleaning"},
    )
    velocity_travel: float = field(
        default=20.0,
        metadata={"units": "km/h", "description": "Truck velocity during travel"},
    )
    # Times
    time_setup: float = field(
        default=30.0,
        metadata={
            "units": "seconds/heliostat",
            "description": "Setup time per heliostat",
        },
    )
    time_shift: float = field(
        default=8.0,
        metadata={"units": "hours", "description": "Duration of cleaning shift"},
    )
    # Distances and volumes
    distance_reload_station: float = field(
        default=750.0,
        metadata={"units": "m", "description": "Distance to reload station"},
    )
    truck_water_volume: float = field(
        default=15000.0, metadata={"units": "L", "description": "Water tank capacity"}
    )
    # Heliostat dimensions
    heliostat_width: float = field(
        default=None, metadata={"units": "m", "description": "Width of heliostat"}
    )
    heliostat_height: float = field(
        default=None, metadata={"units": "m", "description": "Height of heliostat"}
    )


class Truck:
    """Truck class with parameter management system that will automatically update cleaning sectors and cleaning rate with each update."""

    def __init__(self, config_path: Path = None):
        self._params = TruckParameters()
        self._solar_field = (
            None  # Solarfield ID and positions with respect to receiver (m) [ID, x-x, y-y]
        )
        self._cleaning_rate = None  # Number of heliostats cleaned per truck per shift
        self._sectors = None  # Number of cleaning sectors to create in the field
        self._n_sectors_per_truck = None  # Number of sectors cleaned per truck per shift
        self._consumable_costs = {"water": None, "fuel": None, "total": None}
        if config_path:
            self.load_config(Path(config_path))

    @property
    def consumable_costs(self) -> dict:
        """Get current cleaning costs per sector."""
        if self._consumable_costs["water"] is None or self._consumable_costs["fuel"] is None:
            self._calculate_costs()
        return self._consumable_costs

    def _calculate_costs(self) -> None:
        """Calculate water and fuel costs per cleaning sector."""
        if not all(
            [hasattr(self._params, attr) for attr in ["heliostat_width", "heliostat_height"]]
        ):
            raise ValueError("Must set heliostat dimensions before calculating costs")

        p = self._params

        # Calculate area per cleaning sector
        area_heliostat_cleaning_sector = (
            p.heliostat_width  # [m]
            * p.heliostat_height  # [m]
            * self.cleaning_rate  # [heliostats/shift]
            / self.n_sectors_per_truck  # [sectors/shift]
        )  # [m²/sector]

        # Calculate water cost per sector
        self._consumable_costs["water"] = (
            p.usage_water  # [L/m²]
            * p.cost_water
            / 1e3  # [$/L]
            * area_heliostat_cleaning_sector  # [m²/sector]
        )  # [$/cleaning sector]

        # Calculate fuel cost per sector
        self._consumable_costs["fuel"] = (
            p.usage_fuel  # [L/hour]
            * p.cost_fuel  # [$/L]
            * p.time_shift  # [hours/shift]
            / self.n_sectors_per_truck  # [sectors/shift]
        )  # [$/cleaning sector]

        # Calculate total cost
        self._consumable_costs["total"] = (
            self._consumable_costs["water"] + self._consumable_costs["fuel"]
        )

    @property
    def cleaning_rate(self) -> float:
        """Get current cleaning rate."""
        if self._cleaning_rate is None:
            raise ValueError("Must call calculate_cleaning_rate first")
        return self._cleaning_rate

    @cleaning_rate.setter
    def cleaning_rate(self, rate: float):
        """Set cleaning rate."""
        self._cleaning_rate = rate
        self._params.cleaning_rate = rate
        print(f"Updated cleaning rate to {rate:.1f} heliostats/shift")

    @property
    def sectors(self) -> tuple:
        """Get current sector configuration."""
        if self._sectors is None:
            raise ValueError("Must call calculate_cleaning_rate first")
        return self._sectors

    @sectors.setter
    def sectors(self, new_sectors: tuple):
        """Set sector configuration (n_rad, n_az)."""
        self._sectors = new_sectors
        print(
            f"Updated sectors to {new_sectors[0]} x {new_sectors[1]} = {new_sectors[0] * new_sectors[1]} total sectors"
        )

    @property
    def n_sectors_per_truck(self) -> int:
        """Get number of sectors cleaned per truck."""
        if self._n_sectors_per_truck is None:
            raise ValueError("Must call calculate_cleaning_rate first")
        return self._n_sectors_per_truck

    @n_sectors_per_truck.setter
    def n_sectors_per_truck(self, n: int):
        """Set number of sectors cleaned per truck."""
        self._n_sectors_per_truck = n
        self._params.n_sectors_cleaned_per_truck = n
        print(f"Updated sectors per truck to {n}")

    def update_parameters(self, **kwargs):
        """Update truck parameters with validation and recalculate cleaning rate.

        Args:
            **kwargs: Parameter names and values to update
        """
        old_rate = self.cleaning_rate if self._solar_field is not None else None

        for param_name, value in kwargs.items():
            if hasattr(self._params, param_name):
                setattr(self._params, param_name, value)
                print(f"Updated {param_name} to {value}")
            else:
                print(f"Warning: {param_name} is not a valid parameter")

        # Show cleaning rate change if field properties are set
        if self._solar_field is not None:
            new_rate = self.cleaning_rate
            print(
                f"Cleaning rate changed from {old_rate:.1f} to {new_rate:.1f} heliostats per shift"
            )
            print("Updated costs per sector:")
            print(f"  Total: {self._costs['total']:.2f} $/cleaning sector")

    def load_config(self, config_path: Path) -> None:
        """Load parameters from CSV file, using defaults for missing values."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            df = pd.read_excel(config_path, index_col="Parameter")

            # Update only parameters that exist in config file
            for param, value in df["Value"].items():
                if hasattr(self._params, param):
                    setattr(self._params, param, value)

        except Exception as e:
            print(f"Error loading truck properties: {e}")
            print("Using default parameters")

    def calculate_cleaning_rate(
        self,
        solar_field,
        cleaning_rate: float = None,
        num_sectors: Optional[Union[int, Tuple[int, int], str]] = None,
        tolerance: float = 0.05,
    ) -> tuple:
        """Calculate cleaning rate based on truck parameters or use provided rate.

        Args:
            solar_field (np.ndarray): Array containing heliostat positions
            cleaning_rate (float, optional): Manual override for cleaning rate
            tolerance (float, optional): Maximum allowed difference between calculated and discretized rates

        Returns:
            tuple: (cleaning_rate, (n_rad, n_az), n_sectors_per_truck)
        """
        # Calculate or use provided cleaning rate
        if cleaning_rate is None and (self._sectors or num_sectors) is None:
            hour_per_reload = self._hour_per_reload_consumables()
            spacing = self._heliostat_spacing(solar_field[:, 1], solar_field[:, 2])
            hour_per_clean = self._hour_per_heliostat_cleaning(
                heliostat_spacing=spacing,
                truck_velocity=self._params.velocity_travel,
                truck_cleaning_velocity=self._params.velocity_cleaning,
                cleaning_setup_seconds=self._params.time_setup,
            )
            target_rate = self._heliostats_cleaned_shift(
                shift_hours=self._params.time_shift,
                hour_per_heliostat_clean=hour_per_clean,
                hour_reloading_per_heliostat=hour_per_reload,
            )
            print(f"Calculated cleaning rate: {target_rate:.1f} heliostats/shift")
        elif cleaning_rate is not None:
            target_rate = cleaning_rate
            print(f"Using config specified cleaning rate: {target_rate:.1f} heliostats/shift")
        elif num_sectors is not None:
            if isinstance(num_sectors, int):
                n_rad = int(np.sqrt(num_sectors))
                n_az = int(np.ceil(num_sectors / n_rad))
                self._sectors = (n_rad, n_az)
                target_rate = len(solar_field) / (n_rad * n_az)
                print(f"Using manual sector configuration: {n_rad} x {n_az} sectors")
            elif isinstance(num_sectors, tuple) and len(num_sectors) == 2:
                self._sectors = num_sectors
                target_rate = len(solar_field) / (num_sectors[0] * num_sectors[1])
                print(
                    f"Using manual sector configuration: {num_sectors[0]} x {num_sectors[1]} sectors"
                )
            else:
                raise ValueError("num_sectors must be an int or a tuple of two ints")
        else:
            raise ValueError(
                "Must provide either:\n1. Manual cleaning rate: cleaning_rate only,\n2. Auto cleaning rate calculation: no num_sectors and no cleaning_rate.\n3. Manual sector configuration: num_sectors"
            )

        if target_rate > len(solar_field):
            target_rate = len(solar_field)
            cleaning_rate = len(solar_field)
            print(
                f"Warning: Target rate {target_rate} exceeds number of heliostats {len(solar_field)}. Setting to {len(solar_field)}"
            )
        # Calculate sectors based on target rate
        self._optimize_sectors(solar_field, target_rate, tolerance)

        # Update consumable costs
        self._calculate_costs()

    def _optimize_sectors(self, solar_field, target_rate: float, tolerance: float) -> tuple:
        """Calculate optimal sector configuration for given cleaning rate."""
        n_sectors_per_truck = 1
        best_error = float("inf")
        best_sectors = None

        if target_rate >= len(
            solar_field
        ):  # Increase field resoltuion if we are cleaning full field in one truck
            best_n_sectors = int(np.ceil(len(solar_field) / 50))
            best_sectors = (
                int(np.floor(np.sqrt(best_n_sectors))),
                int(np.ceil(np.sqrt(best_n_sectors))),
            )
            best_rate = len(solar_field) / (best_sectors[0] * best_sectors[1] / best_n_sectors)
            best_error = abs(best_rate - target_rate) / target_rate
            if best_rate < target_rate:
                raise ValueError("Target rate exceeds number of heliostats in field")
        else:
            while n_sectors_per_truck <= 10:
                n_sectors = len(solar_field) / target_rate
                n_sectors_scaled = n_sectors * n_sectors_per_truck

                n_rad = max(1, int(np.sqrt(n_sectors_scaled)))
                n_az = int(np.ceil(n_sectors_scaled / n_rad))

                while n_rad * n_az < n_sectors_scaled:
                    n_rad += 1
                    n_az = int(np.ceil(n_sectors_scaled / n_rad))

                actual_rate = len(solar_field) / (n_rad * n_az / n_sectors_per_truck)
                error = abs(actual_rate - target_rate) / target_rate

                if error < best_error:
                    best_error = error
                    best_sectors = (n_rad, n_az)
                    best_rate = actual_rate
                    best_n_sectors = n_sectors_per_truck

                if error <= tolerance:
                    break

                n_sectors_per_truck += 1

        # Store results
        self._cleaning_rate = best_rate
        self._sectors = best_sectors
        self._n_sectors_per_truck = best_n_sectors

        print(
            f"Grid size: {best_sectors[0]} x {best_sectors[1]} = {best_sectors[0] * best_sectors[1]} sectors"
        )
        print(f"Sectors per truck: {best_n_sectors}")
        print(f"Effective cleaning rate: {best_rate:.1f} heliostats/shift")
        print(f"Error from target: {best_error * 100:.1f}%")

    def _heliostats_cleaned_shift(
        self,
        shift_hours=None,
        hour_per_heliostat_clean=None,
        hour_reloading_per_heliostat=None,
    ) -> float:
        """Calculate heliostats cleaned per shift."""

        return shift_hours / (hour_per_heliostat_clean + hour_reloading_per_heliostat)

    def _hour_per_heliostat_cleaning(
        self,
        heliostat_spacing: float,
        truck_velocity: float = 10.0,
        truck_cleaning_velocity: float = 2.0,
        cleaning_setup_seconds: float = 30.0,
    ):
        """Calculates the time it takes to move to a heliostat and clean it."""
        return (
            heliostat_spacing / (truck_velocity * 1e3)
            + self._params.heliostat_width / (truck_cleaning_velocity * 1e3)
            + (cleaning_setup_seconds / 3600)
        )  # [hours] time it takes to move to a heliostat and clean it

    def _heliostat_spacing(self, positions_x: np.ndarray, positions_y: np.ndarray) -> float:
        """Calculate average spacing between heliostats."""
        n_heliostats = len(positions_x)
        min_distances = np.zeros(n_heliostats)
        for i in range(n_heliostats):
            distances = np.sqrt(
                (positions_x - positions_x[i]) ** 2 + (positions_y - positions_y[i]) ** 2
            )
            distances[distances == 0] = np.inf
            min_distances[i] = np.min(distances)
        return np.mean(min_distances) - self._params.heliostat_width

    def _hour_per_reload_consumables(self) -> float:
        """Calculate time required for consumable reloading."""
        p = self._params

        heliostat_area = p.heliostat_width * p.heliostat_height  # [m^2] area of heliostat
        cleaning_capacity_area = (
            p.truck_water_volume / p.usage_water
        )  # [m^2] cleaning capacity of water
        reload_occurence_rate = heliostat_area / cleaning_capacity_area  # []

        hour_travel_reload = (
            2 * p.distance_reload_station / (p.velocity_travel * 1e3)
        )  # [hours] travel time to reload station

        return reload_occurence_rate * (hour_travel_reload + p.time_shift)  # [hours]


@dataclass
class Sun:
    """
    Manage solar geometry and clearsky direct normal irradiance data.

    Stores solar irradiation, angles, and clearsky DNI per time index.
    """

    irradiation: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={
            "units": "W/m^2",
            "description": "Extraterrestrial nominal solar irradiation",
        },
    )
    elevation: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "degrees", "description": "Solar elevation angles"},
    )
    declination: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "degrees", "description": "Solar declination angles"},
    )
    azimuth: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "degrees", "description": "Solar azimuth angles"},
    )
    zenith: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "degrees", "description": "Solar zenith angles"},
    )
    hourly: Dict[int, Any] = field(
        init=False,
        default_factory=dict,
        metadata={"description": "Hourly solar parameters"},
    )
    time: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={"units": "datetime", "description": "Time vector for solar angles"},
    )
    DNI: Dict[int, np.ndarray] = field(
        init=False,
        default_factory=dict,
        metadata={
            "units": "W/m^2",
            "description": "Direct normal irradiance at ground",
        },
    )
    stow_angle: float = field(
        init=False,
        metadata={
            "units": "degrees",
            "description": "Minimum sun elevation angle where heliostat field operates",
        },
    )

    def import_sun(self, file_params: str) -> None:
        """
        Load stow angle from an Excel parameter file.

        Args:
            file_params (str): Path to the Excel file containing 'stowangle'.
        """
        table = pd.read_excel(file_params, index_col="Parameter")
        self.stow_angle = float(table.loc["stowangle"].Value)

    def angles_and_clearsky_dni(
        self, lat: float, lon: float, time_grid: pd.Series, tz_offset: float = 0.0
    ) -> None:
        """
        Compute solar angles and clearsky direct normal irradiance.

        Calculates azimuth, elevation, and clearsky DNI using pysolar for each
        datetime in the provided time grid, storing them keyed by the next index.

        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.
            time_grid (pd.Series): Series of datetimes for solar calculations.
            tz_offset (float, optional): UTC offset in hours. Defaults to 0.0.
        """
        timezone = pytz.FixedOffset(int(tz_offset * 60))
        tg = np.array(time_grid.dt.to_pydatetime())
        time_utc = [t.replace(tzinfo=timezone) for t in tg]

        solar_angles = np.array([solar.get_position(lat, lon, t) for t in time_utc])
        az = solar_angles[:, 0]
        el = solar_angles[:, 1]
        dni_vals = np.array(
            [
                radiation.get_radiation_direct(time, elevation) if elevation > 0 else 0.0
                for time, elevation in zip(time_utc, el)
            ]
        )

        idx = len(self.azimuth)
        self.azimuth[idx] = az
        self.elevation[idx] = el
        self.DNI[idx] = dni_vals


class Heliostats:
    def __init__(self):

        # Properties of heliostat (scalars, assumes identical heliostats)
        self.hamaker = []  # [J] hamaker constant of heliostat glass
        self.poisson = []  # [-] poisson ratio of heliostat glass
        self.youngs_modulus = []  # [Pa] young's modulus of glass
        self.nominal_reflectance = []  # [-] as clean reflectance of the heliostat
        self.height = []
        self.width = []
        self.num_radial_sectors = []
        self.num_theta_sectors = []

        # Properties of individual heliostats (1D array indexed by heliostat_index)
        self.x = []  # [m] x (east-west) position of representative heliostats
        self.y = []  # [m] y (north-south) position of representative heliostats
        self.rho = []  # [m] radius for polar coordinates of representative heliostats
        self.theta = (
            []
        )  # [deg] angle (from north) for polar coordinates of representative heliostats
        self.dist = []  # planar distance to tower
        self.elevation_angle_to_tower = []  # elevation angle from heliostats to tower
        self.sector_area = []  # [m**2] sector area
        self.full_field = {
            "rho": [],
            "theta": [],
            "x": [],
            "y": [],
            "z": [],
            "sector_id": [],
        }  # populated if representative heliostats are from a sectorization of a field

        self.acceptance_angles = {}  # acceptance angle for receiver

        # Mie extinction weighting (dict of 2D arrays indexed by heliostat index, dust diameter)
        self.extinction_weighting = {}

        # Movement properties (dicts of 2D arrays indexed by [heliostat_index, time] with weather file name keys )
        self.tilt = {}  # [deg] tilt angle of the heliostat
        self.azimuth = {}  # [deg] azimuth angle of the heliostat
        self.incidence_angle = {}  # [deg] incidence angle of solar rays
        self.elevation = {}  # [deg] elevation angle of the heliostat
        self.inc_ref_factor = (
            {}
        )  # [ - ] incidence factor for reflectance computation (1st or second surface)
        self.stow_tilt = {}  # [deg] tilt at which heliostats are stowed at night
        self.optical_efficiency = (
            {}
        )  # [ - ] average total optical efficiency of the sector represented by the heliostat

        # Properties of dust on heliostat (dicts of 3D arrays, indexed by [heliostat_index, time, diameter] with experiment numbers as keys)
        self.delta_soiled_area = (
            {}
        )  # [m^2/m^2] "pdf" of projected area of dust deposited on mirror for each time interval & each diameter
        self.mom_removal = {}
        self.mom_adhesion = {}
        self.soiling_factor = {}
        self.D = {}  # [µm] diameter discretization
        self.velocity = {}  # [m/s] velocity of falling dust for each diameter
        self.pdfqN = (
            {}
        )  # dq[particles/(s*m^2)]/dLog_{10}(D[µm]) "pdf" of dust flux 1 m2 of mirror (constant for each time interval) at each diameter
        self.delta_soiled_area_variance = {}
        self.soiling_factor_prediction_variance = {}

    def import_helios(
        self,
        file_params,
        file_solar_field=None,
        cleaning_rate: float = None,
        num_sectors: Optional[Union[int, Tuple[int, int], str]] = None,
        verbose=True,
    ) -> None:

        table = pd.read_excel(file_params, index_col="Parameter")
        # self.h_tower = float(table.loc['h_tower'].Value)
        self.hamaker = float(table.loc["hamaker_glass"].Value)
        self.poisson = float(table.loc["poisson_glass"].Value)
        self.youngs_modulus = float(table.loc["youngs_modulus_glass"].Value)
        self.nominal_reflectance = float(table.loc["nominal_reflectance"].Value)
        self.height = float(table.loc["heliostat_height"].Value)
        self.width = float(table.loc["heliostat_width"].Value)
        self.stow_tilt = float(table.loc["stow_tilt"].Value)
        solar_field = self.read_solarfield(file_solar_field)

        self.truck = Truck(config_path=file_params)
        self.truck.calculate_cleaning_rate(
            solar_field=solar_field,
            cleaning_rate=cleaning_rate,
            num_sectors=num_sectors,
            tolerance=0.05,
        )

        if (
            isinstance(self.truck.sectors, str) and self.truck.sectors.lower() == "manual"
        ):  # Manual importing of solar field respresentatives
            self.x = solar_field[:, 1]  # x cartesian coordinate of each heliostat (E>0)
            self.y = solar_field[:, 2]  # y cartesian coordinate of each heliostat (N>0)
            self.rho = np.sqrt(
                self.x**2 + self.y**2
            )  # angular polar coordinate of each heliostat (E=0, positive counterclockwise)
            self.theta = np.arctan2(self.y, self.x)  # radial polar coordinate of each heliostat
            self.num_radial_sectors = None
            self.num_theta_sectors = None
        elif (
            isinstance(self.truck.sectors, tuple)
            and isinstance(self.truck.sectors[0], int)
            and table.loc["receiver_type"].Value == "External cylindrical"
            and isinstance(self.truck.sectors[1], int)
        ):  # import and sectorize
            n_rho, n_theta = self.truck.sectors
            _print_if(
                "Sectorizing with {0:d} angular and {1:d} radial sectors".format(n_theta, n_rho),
                verbose,
            )
            self.num_radial_sectors, self.num_theta_sectors = self.truck.sectors
            self.sectorize_radial(solar_field, n_rho, n_theta)
        elif table.loc["receiver_type"].Value == "Flat plate":
            n_hor, n_vert = self.truck.sectors
            _print_if(
                "Sectorizing with {0:d} horizontal and {1:d} vertical sectors".format(
                    n_hor, n_vert
                ),
                verbose,
            )
            self.num_radial_sectors, self.num_theta_sectors = self.truck.sectors
            self.sectorize_kmeans_clusters(
                solar_field, self.truck.sectors[0] * self.truck.sectors[1]
            )
            # self.sectorize_corn_cleaningrows(solar_field,n_hor,n_vert)
        else:
            raise ValueError("num_sectors must be None or an a 2-tuple of intergers")

    def sectorize_radial(self, solar_field, n_rho, n_theta, verbose=True):
        x = solar_field[:, 1]  # x cartesian coordinate of each heliostat (E>0)
        y = solar_field[:, 2]  # y cartesian coordinate of each heliostat (N>0)
        # n_sec = n_rho*n_theta
        n_tot = len(x)
        extra_hel_th = np.mod(n_tot, n_theta)

        rho = np.sqrt(x**2 + y**2)  # radius - polar coordinates of each heliostat
        theta = np.arctan2(y, x)  # angle - polar coordinates of each heliostat

        val_t1 = np.sort(theta)  # sorts the heliostats by ascendent thetas
        idx_t = np.argsort(theta)  # store the indexes of the ascendent thetas
        val_r1 = rho[idx_t]  # find the corresponding values of the radii

        val_r = np.concatenate(
            (val_r1[val_t1 >= -np.pi / 2], val_r1[val_t1 < -np.pi / 2])
        )  # "rotates" to have -pi/2 as the first theta value
        val_t = np.concatenate(
            (val_t1[val_t1 >= -np.pi / 2], val_t1[val_t1 < -np.pi / 2] + 2 * np.pi)
        )  # "rotates" to have -pi/2 as the first theta value

        self.full_field["rho"] = val_r
        self.full_field["theta"] = val_t
        self.full_field["x"] = val_r * np.cos(val_t)
        self.full_field["y"] = val_r * np.sin(val_t)
        self.full_field["id"] = np.arange(n_tot, dtype=np.int64)
        self.full_field["sector_id"] = np.nan * np.ones(len(x))

        # compute the coordinates of the angular sector-delimiting heliostats
        n_th_hel = np.floor(n_tot / n_theta).astype(
            "int"
        )  # rounded-down number of heliostats per angular sector
        if extra_hel_th == 0:
            idx_th_sec = np.arange(0, len(val_r1), n_th_hel)
        else:
            # compute the angular-sector delimiting heliostats to have sectors with same (or as close as possible) number of heliostats
            id_at = np.array([0])
            id_bt = np.arange(1, extra_hel_th + 1, 1)
            id_ct = extra_hel_th * np.ones(n_theta - extra_hel_th - 1).astype("int")
            id_dt = np.array([extra_hel_th - 1])
            idx_th_sec = np.arange(0, len(val_r), n_th_hel) + np.concatenate(
                (id_at, id_bt, id_ct, id_dt)
            )

        theta_th_sec = val_t[idx_th_sec]
        # rho_th_sec = val_r[idx_th_sec]

        rho_r_sec = np.zeros((n_rho, n_theta))
        theta_r_sec = np.zeros((n_rho, n_theta))
        hel_sec = []
        hel_rep = np.zeros((n_rho * n_theta, 4))
        self.sector_area = np.zeros(n_rho * n_theta)
        kk = 0
        for ii in range(n_theta):
            if ii != n_theta - 1:
                in_theta_slice = (val_t >= theta_th_sec[ii]) & (val_t < theta_th_sec[ii + 1])
                thetas = val_t[
                    in_theta_slice
                ]  # selects the heliostats whose angular coordinate is within the ii-th angular sector
                rhos = val_r[in_theta_slice]  # selects the correspondent values of radius
            else:
                in_theta_slice = val_t >= theta_th_sec[ii]
                thetas = val_t[in_theta_slice]  # same as above for the last sector
                rhos = val_r[in_theta_slice]  # same as above for the last sector

            AR = np.sort(rhos)  # sort the heliostats belonging to each sector by radius
            AR_idx = np.argsort(rhos)  # store the indexes
            AT = thetas[AR_idx]  # find the corresponding thetas

            # compute the angular-sector delimiting heliostats to have sectors with same (or as close as possible) number of heliostats
            id_ar = np.array([0])
            id_br = (np.floor(len(AR) / n_rho) * np.ones(n_rho)).astype("int")
            id_cr = np.ones(np.mod(len(AR), n_rho)).astype("int")
            id_dr = np.zeros((n_rho - np.mod(len(AR), n_rho))).astype("int")
            idx_r_sec = np.cumsum(np.concatenate((id_ar, id_br + np.concatenate((id_cr, id_dr)))))
            AR_sec = AR[idx_r_sec[0:n_rho]]
            AT_sec = AT[idx_r_sec[0:n_rho]]
            rho_r_sec[:, ii] = AR_sec[
                0 : len(rho_r_sec[:, ii])
            ]  # finds the radial sector-delimiting heliostats for each angular sector
            theta_r_sec[:, ii] = AT_sec[
                0 : len(rho_r_sec[:, ii])
            ]  # finds the corresponding angles of the radial sector-delimiting heliostats for each angular sector

            # select the heliostats whose radial coordinate is within the jj-th radial sector of the ii-th angular sector
            for jj in range(n_rho):
                if jj != n_rho - 1:
                    and_in_radius_slice = (rhos >= rho_r_sec[jj, ii]) & (
                        rhos < rho_r_sec[jj + 1, ii]
                    )
                    rhos_jj = rhos[and_in_radius_slice]
                    thetas_jj = thetas[and_in_radius_slice]
                else:
                    and_in_radius_slice = rhos >= rho_r_sec[jj, ii]
                    rhos_jj = rhos[and_in_radius_slice]  # same as above for the last sector
                    thetas_jj = thetas[and_in_radius_slice]  # same as above for the last sector
                # hel_sec.append((rhos_jj,thetas_jj))   # store all the heliostats belonging to each sector
                rho_sec = np.mean(rhos_jj)  # compute the mean radius for the sector
                theta_sec = np.mean(thetas_jj)  # compute the mean angle for the sector

                idx = np.where(in_theta_slice)[0][and_in_radius_slice]
                self.full_field["sector_id"][idx] = kk
                self.sector_area[kk] = len(idx) * self.height * self.width  # sector area

                # define the representative heliostats for each sector
                hel_rep[kk, 0] = rho_sec
                hel_rep[kk, 1] = theta_sec
                kk += 1

        hel_rep[:, 2] = hel_rep[:, 0] * np.cos(hel_rep[:, 1])
        hel_rep[:, 3] = hel_rep[:, 0] * np.sin(hel_rep[:, 1])

        self.x = hel_rep[:, 2]
        self.y = hel_rep[:, 3]
        self.theta = hel_rep[:, 1]
        self.rho = hel_rep[:, 0]
        self.heliostats_in_sector = hel_sec

    def sectorize_kmeans_clusters(self, solar_field, num_sectors):
        """Cluster heliostats based on distance and set representative heliostats."""
        print("Clustering heliostats...")
        weighted_positions = solar_field[:, 1:]  # Get x,y coordinates
        kmeans = KMeans(n_clusters=num_sectors, random_state=42)
        labels = kmeans.fit_predict(weighted_positions)
        cluster_centers = kmeans.cluster_centers_

        # Initialize arrays to store information
        self.x = np.zeros(num_sectors)
        self.y = np.zeros(num_sectors)
        heliostats_in_sector = np.zeros(num_sectors, dtype=int)
        self.sector_area = np.zeros(num_sectors)

        # Store full field information
        self.full_field["x"] = solar_field[:, 1]
        self.full_field["y"] = solar_field[:, 2]
        self.full_field["id"] = solar_field[:, 0]
        self.full_field["sector_id"] = labels

        # For each cluster, find closest heliostat to center and calculate sector information
        for i in range(num_sectors):
            # Find heliostats in this cluster
            mask = labels == i
            # positions_in_cluster = weighted_positions[mask]

            # Count heliostats in this sector
            heliostats_in_sector[i] = np.sum(mask)

            # Calculate cluster area
            self.sector_area[i] = heliostats_in_sector[i] * self.height * self.width

            # Find closest heliostat to cluster center
            if np.sum(mask) > 0:  # Make sure cluster isn't empty
                distances = np.linalg.norm(weighted_positions - cluster_centers[i], axis=1)
                closest_idx = np.argmin(distances)

                # Set representative heliostat coordinates
                self.x[i] = solar_field[closest_idx, 1]
                self.y[i] = solar_field[closest_idx, 2]

        # Store the number of heliostats per sector
        self.heliostats_in_sector = heliostats_in_sector

    def sectorize_corn_cleaningrows(self, solar_field, n_hor, n_vert, verbose=True):
        """
        Sectorize the solar field by dividing it into a grid of horizontal and vertical sectors.

        This function reads the solar field coordinates from a CSV or XLSX file, generates a grid around
        the solar field, and assigns each heliostat to the closest grid point. The function then computes
        the representative heliostat for each sector and stores the sector information in the object's
        attributes.

        Parameters:
            whole_field_file (str): The file path to the CSV or XLSX file containing the solar field coordinates.
            n_hor (int): The number of horizontal sectors to divide the solar field into.
            n_vert (int): The number of vertical sectors to divide the solar field into.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.

        Returns:
            None
        """

        def generate_grid(num_hor, num_vert, x, y):  # Generate a grid around solarfield
            x_points = np.linspace(min(x), max(x), num_hor)
            y_points = np.linspace(min(y), max(y), num_vert)
            grid = np.array([(x, y) for x in x_points for y in y_points])
            return grid

        def find_closest_point(position, grid):
            distances = cdist(
                [position[1:3]], grid
            )  # Find distance between heliostats and grid coordinates
            closest_idx = np.argmin(distances)
            return distances[0][closest_idx], closest_idx

        grid = generate_grid(n_hor, n_vert, solar_field[:, 1], solar_field[:, 2])

        closest_grid = []  # Create a dictionary to store
        # [heliostat ID, x position, y position, distance to closest grid, closest grid point]
        for i in range(len(solar_field)):
            distance_grid, closest_idx = find_closest_point(solar_field[i, :], grid)
            if i == 0:
                closest_grid = np.hstack([solar_field[i, :], distance_grid, closest_idx])
            else:
                closest_grid = np.vstack(
                    [
                        closest_grid,
                        np.hstack([solar_field[i, :], distance_grid, closest_idx]),
                    ]
                )

        # Store Heliostat Field information
        self.full_field["x"] = closest_grid[:, 1]
        self.full_field["y"] = closest_grid[:, 2]
        self.full_field["id"] = np.array(closest_grid[:, 0], dtype=np.int64)
        self.full_field["sector_id"] = np.array(closest_grid[:, 4], dtype=np.int64)

        for i in np.unique(self.full_field["sector_id"]):
            sector_field = closest_grid[closest_grid[:, 4] == i, :]
            sector_size = len(sector_field)
            representative_info = sector_field[np.argmin(sector_field[:, 3])]
            if i == 0:
                representative_helio = np.hstack([representative_info, sector_size])
            else:
                representative_helio = np.vstack(
                    [
                        representative_helio,
                        np.hstack([representative_info, sector_size]),
                    ]
                )

        ##
        self.x = representative_helio[:, 1]
        self.y = representative_helio[:, 2]
        self.heliostats_in_sector = np.array(representative_helio[:, -1], dtype=np.int64)
        self.sector_area = self.heliostats_in_sector * self.height * self.width

    @staticmethod
    def read_solarfield(field_filepath):  # Load CSV containing solarfield coordintes
        positions = []
        if field_filepath.split(".")[-1] == "csv":
            whole_SF = pd.read_csv(field_filepath, skiprows=[1])
        elif field_filepath.split(".")[-1] == "xlsx":
            whole_SF = pd.read_excel(field_filepath, skiprows=[1])
        else:
            raise ValueError("Solar field file must be csv or xlsx")

        x_field = np.array(
            whole_SF.loc[:, "Loc. X"]
        )  # x cartesian coordinate of each heliostat (E>0)
        y_field = np.array(whole_SF.loc[:, "Loc. Y"])
        helioID = np.arange(len(x_field), dtype=np.int64)
        positions = np.column_stack((helioID, x_field, y_field))
        return positions

    def sector_plot(self, show_id=False, cmap_name="turbo_r", figsize=(12, 10)):
        """
        Plot the heliostat field with sectors colored distinctly.

        Parameters:
            show_id (bool, optional): Whether to show sector IDs. Default is False.
            cmap (str, optional): Matplotlib colormap to use. Default is 'tab20' which supports up to 20 distinct colors.
                                Other good options: 'tab10', 'viridis', 'plasma', 'Set1', 'Set2', 'Set3'.
            figsize (tuple, optional): Figure size (width, height) in inches.

        Returns:
            tuple: (fig, ax) The figure and axis objects for further customization.
        """
        Ns = self.x.shape[0]  # Number of sectors
        sid = self.full_field["sector_id"]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create a custom colormap where adjacent sectors have contrasting colors
        base_map = np.linspace(0.0, 1.0, len(np.unique(sid)))
        c_map = base_map
        for ii in range(1, len(np.unique(sid))):
            c_map = np.vstack((c_map, np.roll(base_map, 3 * ii)))
        c_map = c_map.flatten()
        color_map = plt.cm.get_cmap(cmap_name)(c_map)

        # Plot each sector
        for ii in range(Ns):
            mask = sid == ii
            ax.scatter(
                self.full_field["x"][mask],
                self.full_field["y"][mask],
                color=(
                    color_map[ii % len(color_map)]
                    if isinstance(color_map, np.ndarray)
                    else color_map(ii / max(1, Ns - 1))
                ),
                alpha=0.7,
                s=30,
                label=f"Sector {ii}" if ii < 10 else None,  # Limit legend entries
            )

            if show_id:
                # Add sector ID label with larger font
                center_x = np.mean(self.full_field["x"][mask])
                center_y = np.mean(self.full_field["y"][mask])
                ax.text(
                    center_x,
                    center_y,
                    str(ii),
                    alpha=1.0,
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"),
                )

        # Plot representative heliostats if not showing IDs
        if not show_id:
            ax.scatter(
                self.x,
                self.y,
                color="black",
                marker="X",
                s=100,
                label="Representative heliostats",
                zorder=10,
            )

        # Add plot styling
        ax.set_xlabel("Distance from receiver - X [m]")
        ax.set_ylabel("Distance from receiver - Y [m]")
        ax.set_title("Solar Field Sectors")
        ax.grid(True, linestyle="--", alpha=0.7)

        # Set aspect ratio to equal to ensure correct spatial representation
        ax.set_aspect("equal")

        plt.tight_layout()
        return fig, ax

    def _validate_lookup_table(
        self,
        lookup_path: Path,
        sim_dat: SimulationInputs,
        verbose: bool,
    ) -> bool:
        """Validate the metadata of an existing lookup table."""
        metadata_path = lookup_path / "metadata.json"
        if not metadata_path.is_file():
            _print_if(f"Validation failed: {metadata_path} not found.", verbose)
            return False

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            _print_if(f"Validation failed: Could not decode {metadata_path}.", verbose)
            return False

        # --- Start validation checks ---
        current_m_real = [s.real for s in sim_dat.dust.m.values()]
        current_m_imag = [s.imag for s in sim_dat.dust.m.values()]
        if not np.allclose(
            metadata.get("refractive_index_real", []), current_m_real
        ) or not np.allclose(metadata.get("refractive_index_imag", []), current_m_imag):
            _print_if("Validation failed: Refractive index mismatch.", verbose)
            return False

        _print_if("Lookup table metadata validation successful.", verbose)
        return True

    def _load_from_lookup_table(self, lookup_path: Path) -> Tuple[dict, dict, list]:
        """Load data from a validated lookup table directory."""
        try:
            with open(lookup_path / "extinction_weights_lookup_table.json", "r") as f:
                ext_weights = {int(k): v for k, v in json.load(f).items()}
            with open(lookup_path / "acceptance_angles.json", "r") as f:
                acc_angles = {int(k): v for k, v in json.load(f).items()}
            with open(lookup_path / "diameters.json", "r") as f:
                diameters = json.load(f)
            return ext_weights, acc_angles, diameters
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise IOError(f"Failed to load lookup table from {lookup_path}: {e}")

    def compute_extinction_weights(
        self,
        simulation_data: SimulationInputs,
        loss_model: str,
        num_acceptance_steps: int = 100,
        lookup_table_file_folder: Optional[str] = None,
        verbose: bool = True,
        show_plots: bool = False,
        options: dict = {},
    ) -> None:
        """
        Compute extinction weighting factors for each heliostat based on the specified loss model.

        This method supports both 'mie' and 'geometry' models. In the 'mie' case, extinction weights are computed
        using Mie theory and heliostat acceptance angles, with the option to read or write to a lookup table.
        In the 'geometry' case, extinction is assumed to be unity.

        Args:
            simulation_data (SimulationInputs): Object containing input parameters for the simulation,
                including dust characteristics, source spectrum, and file identifiers.
            loss_model (str): The model used to compute extinction weights. Must be either
                'mie' (for Mie-theory-based extinction) or 'geometry' (unity extinction).
            num_acceptance_steps (int, optional): Number of discrete acceptance angles to use when creating
                lookup tables. Required if a lookup table is requested but is invalid or does not exist.
            lookup_table_file_folder (str, optional): Directory path to read or store precomputed extinction
                weights and angle/diameter lookup tables.
            verbose (bool, optional): If True, prints detailed progress information to stdout. Defaults to True.
            show_plots (bool, optional): If True, plots extinction curves for each heliostat after computation. Defaults to False.
            options (dict, optional): Additional keyword arguments to pass to the extinction function used for Mie computations.

        Raises:
            ValueError: If `loss_model` is not one of 'mie' or 'geometry'.
            AssertionError: If required inputs for 'mie' model are missing or incorrectly configured.

        Returns:
            None
        """
        sim_dat = simulation_data
        dust = sim_dat.dust
        files = range(len(sim_dat.files))
        num_diameters = [len(dust.D[f]) for f in files]
        num_heliostats = [len(self.tilt[f]) for f in files]
        phia = self.acceptance_angles

        self.extinction_weighting = {
            f: np.zeros((num_heliostats[f], num_diameters[f])) for f in files
        }

        if loss_model == "mie":
            assert len(phia) > 0, "Please call compute_acceptance_angles before this method."
            use_lookup_table = lookup_table_file_folder is not None

            if use_lookup_table:
                lookup_path = get_project_root() / (lookup_table_file_folder)
                is_cache_valid = self._validate_lookup_table(lookup_path, sim_dat, verbose)

                if not is_cache_valid:
                    _print_if(
                        f"Lookup table in {lookup_path} is invalid or missing. Regenerating...",
                        verbose,
                    )
                    assert (
                        num_acceptance_steps is not None
                    ), "num_acceptance_steps must be set to generate a new lookup table."
                    lookup_path.mkdir(parents=True, exist_ok=True)

                    acceptance_angles_range = {
                        f: np.linspace(min(phia[f]), max(phia[f]), num_acceptance_steps)
                        for f in files
                    }
                    self._compute_extinction_weights_lookup_table(
                        sim_dat,
                        acceptance_angles_range,
                        verbose=verbose,
                        save_folder=str(lookup_path),
                        options=options,
                    )

                _print_if(f"Loading extinction weights from {lookup_path}...", verbose)
                ext_weights, acc_angles, diameters = self._load_from_lookup_table(lookup_path)

                for f in files:
                    interpolator = RegularGridInterpolator(
                        (acc_angles[f], diameters),
                        np.array(ext_weights[f]),
                        bounds_error=False,
                        fill_value=None,
                    )
                    grid_angles, grid_dia = np.meshgrid(phia[f], dust.D[f], indexing="ij")
                    points = np.stack([grid_angles.ravel(), grid_dia.ravel()], axis=1)
                    interpolated_values = interpolator(points).reshape(
                        len(phia[f]), len(dust.D[f])
                    )
                    self.extinction_weighting[f][:, :] = interpolated_values

            else:  # Direct computation without lookup table
                _print_if("Computing extinction weights directly (no lookup table)...", verbose)
                same_ext = _same_ext_coeff(self, sim_dat)
                computed = []
                for f in files:
                    for jj in tqdm(
                        range(num_heliostats[f]),
                        desc=f"File {f}",
                        postfix=lambda jj=0: f"acceptance angle {phia[f][jj] * 1e3:.2f} mrad",
                    ):
                        already_computed = [e in computed for _, e in enumerate(same_ext[f][jj])]
                        if any(already_computed):
                            idx = already_computed.index(True)
                            fe, he = same_ext[f][jj][idx]
                            self.extinction_weighting[f][jj, :] = self.extinction_weighting[fe][
                                he, :
                            ]
                        else:
                            ext_weight = _extinction_function(
                                dust.D[f],
                                sim_dat.source_wavelength[f],
                                sim_dat.source_normalized_intensity[f],
                                phia[f][jj],
                                dust.m[f],
                                verbose=verbose,
                                **options,
                            )
                            self.extinction_weighting[f][jj, :] = ext_weight
                            computed.append((f, jj))

                        if show_plots:
                            fig, ax = plt.subplots()
                            ax.semilogx(dust.D[f], self.extinction_weighting[f][jj, :])
                            ax.set_title(
                                f"Heliostat {jj}, acceptance angle {phia[f][jj] * 1e3:.2f} mrad"
                            )
                            plt.show()

            _print_if("... Extinction weight calculation Done!", verbose)

        elif loss_model == "geometry":
            _print_if(
                "Loss Model is 'geometry'. Setting extinction coefficients to unity.", verbose
            )
            for f in files:
                self.extinction_weighting[f] = np.ones((num_heliostats[f], num_diameters[f]))
        else:
            raise ValueError(f"Loss model '{loss_model}' not recognized.")

    def _compute_extinction_weights_lookup_table(
        self,
        simulation_data,
        acceptance_angle_grid=None,
        verbose=True,
        save_folder="./",
        options={},
    ):
        """
        Computes the extinction weights for the heliostat field based on the specified loss model.

        Parameters:
            simulation_data (object): An object containing simulation data, including dust properties and source information.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
            options (dict, optional): Additional options to pass to the extinction function.

        Returns:
            None
        """
        sim_dat = simulation_data
        dust = sim_dat.dust
        files = range(len(sim_dat.files))
        num_diameters = [len(dust.D[f]) for f in files]
        phia = acceptance_angle_grid

        extinction_weighting = {f: np.zeros((len(phia[f]), num_diameters[f])) for f in files}
        assert (
            phia is not None
        ), "When computing the lookup table, please set acceptance_angles_range as a list of values"
        # _print_if("Loss Model is ""mie"". Computing extinction coefficients ... ",verbose)

        for f in files:
            dia = sim_dat.dust.D[f]
            refractive_index = sim_dat.dust.m[f]
            lam = sim_dat.source_wavelength[f]
            intensities = sim_dat.source_normalized_intensity[f]
            h = 0
            for h in tqdm(
                range(len(phia[f])),
                desc=f"File {f}",
                postfix=f"Acceptance angle between {phia[f][0] * 1e3:.0f} and {phia[f][-1] * 1e3:.0f} mrad",
            ):

                ext_weight = _extinction_function(
                    dia,
                    lam,
                    intensities,
                    phia[f][h],
                    refractive_index,
                    verbose=verbose,
                    **options,
                )
                extinction_weighting[f][h, :] = ext_weight

            # df = pd.DataFrame(extinction_weighting[f], index=phia[f], columns=dia)
            # df.index.name = 'Acceptance angles'

        if save_folder[-1] != "/":
            save_folder += "/"
            # df.to_csv(save_folder+f'extinction_weights_lookup_table_{f}.csv')

        with open(save_folder + "extinction_weights_lookup_table.json", "w") as sf:
            json.dump(
                _to_dict_of_lists(extinction_weighting),
                sf,
                ensure_ascii=False,
                indent=4,
            )

        with open(save_folder + "acceptance_angles.json", "w") as sf:
            json.dump(_to_dict_of_lists(phia), sf, ensure_ascii=False, indent=4)

        with open(save_folder + "diameters.json", "w") as sf:
            json.dump(dia.tolist(), sf, ensure_ascii=False, indent=4)

        with open(save_folder + "metadata.json", "w") as sf:
            metadata = {
                "options": options,
                "num_acceptance_steps": len(phia[f]),
                "refractive_index_real": [s.real for s in sim_dat.dust.m.values()],
                "refractive_index_imag": [s.imag for s in sim_dat.dust.m.values()],
                "source_wavelength": _to_dict_of_lists(sim_dat.source_wavelength),
                "source_normalized_intensity": _to_dict_of_lists(
                    sim_dat.source_normalized_intensity
                ),
            }
            json.dump(metadata, sf, ensure_ascii=False, indent=4)

        _print_if("... Lookup done!", verbose)

    def plot_extinction_weights(self, simulation_data, fig_kwargs={}, plot_kwargs={}):
        """
        Plot the extinction weights for each heliostat and file in the simulation data.

        Parameters:
            simulation_data (object): The simulation data object containing the dust and other simulation parameters.
            fig_kwargs (dict, optional): Additional keyword arguments to pass to the `plt.figure()` function.
            plot_kwargs (dict, optional): Additional keyword arguments to pass to the `ax.semilogx()` function.

        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plots.
            ax (list of matplotlib.axes.Axes): The list of axes objects for each plot.
        """

        files = list(self.extinction_weighting.keys())
        Nhelios = [len(self.tilt[f]) for f in files]
        phia = [self.acceptance_angles[f] for f in files]
        nrows = len(files)
        ncols = max(Nhelios)
        # fig,ax = plt.subplots(nrows=len(files),sharex=True,**fig_kwargs)
        fig = plt.figure(**fig_kwargs)
        ax = []
        for ii, f in enumerate(files):
            num_heliostats = Nhelios[f]
            D = simulation_data.dust.D[f]
            idx = ii * ncols + 1
            for jj in range(num_heliostats):
                if jj > 0:
                    ax1 = fig.add_subplot(nrows, ncols, idx, sharex=ax[ii - 1], sharey=ax[ii - 1])
                else:
                    ax1 = fig.add_subplot(nrows, ncols, idx)

                ax.append(ax1)
                ax1.semilogx(D, self.extinction_weighting[f][jj, :], **plot_kwargs)
                ax1.set_xlabel(r"Diameter ($\mu$m)")
                ax1.set_ylabel(r"Extinction area multiplier (-)")
                ax1.set_title(f"File {f}, Mirror {jj}, Acceptance Angle {phia[f][jj]:.2e} rad")
                ax1.grid(True)
                idx += 1
        plt.tight_layout()

        return fig, ax


@dataclass
class Constants:
    """
    Holds physical and empirical constants, loaded from an Excel sheet.
    """

    air_rho: float = field(
        init=False,
        metadata={"units": "kg/m³", "description": "air density at T=293K and p=1 atm"},
    )
    air_mu: float = field(
        init=False,
        metadata={
            "units": "Pa*s",
            "description": "air dynamic viscosity at T=293K and p=1 atm",
        },
    )
    air_nu: float = field(
        init=False,
        metadata={
            "units": "m^2/s",
            "description": "air kinematic viscosity at T=293K and p=1 atm",
        },
    )
    air_lambda_p: float = field(
        init=False,
        metadata={
            "units": "m",
            "description": "mean free path in air at T=293K and p=1 atm",
        },
    )
    irradiation: float = field(
        init=False,
        metadata={"units": "W/m2", "description": "solar extraterrestrial constant"},
    )
    g: float = field(
        default=9.81,
        metadata={"units": "m/s^2", "description": "gravitational constant"},
    )
    A_slip: np.ndarray = field(
        init=False,
        metadata={
            "units": "dimensionless array",
            "description": "coefficients for slip correction factor",
        },
    )
    k_Boltzman: float = field(
        init=False, metadata={"units": "J/K", "description": "Boltzman constant"}
    )
    k_von_Karman: float = field(
        init=False,
        metadata={"units": "dimensionless", "description": "Von Karman constant"},
    )
    N_iter: int = field(
        init=False,
        metadata={
            "units": "count",
            "description": "max iterations to compute gravitational settling velocity",
        },
    )
    tol: float = field(
        init=False,
        metadata={
            "units": "dimensionless",
            "description": "tolerance for convergence in settling velocity computation",
        },
    )
    Re_Limit: np.ndarray = field(
        init=False,
        metadata={
            "units": "dimensionless array",
            "description": "Reynolds limit values for drag coefficient correlations",
        },
    )
    alpha_EIM: float = field(
        init=False,
        metadata={
            "units": "dimensionless",
            "description": "factor for impaction efficiency computation",
        },
    )
    beta_EIM: float = field(
        init=False,
        metadata={
            "units": "dimensionless",
            "description": "factor for impaction efficiency computation",
        },
    )
    eps0: float = field(
        init=False,
        metadata={
            "units": "dimensionless",
            "description": "empirical factor for boundary layer resistance",
        },
    )
    D0: float = field(
        init=False,
        metadata={"units": "m", "description": "common separation distance (Ahmadi)"},
    )

    def import_constants(self, file_params: str, verbose: bool = True):
        """
        Reads constants from an Excel file (indexed by 'Parameter') and populates this dataclass.
        """
        _print_if("\nImporting constants", verbose)

        table = pd.read_excel(file_params, index_col="Parameter")

        self.air_rho = float(table.loc["air_density"].Value)
        self.air_mu = float(table.loc["air_dynamic_viscosity"].Value)
        self.air_nu = self.air_mu / self.air_rho
        self.air_lambda_p = float(table.loc["mean_free_path_air"].Value)

        self.irradiation = float(table.loc["I_solar"].Value)
        # g remains default

        self.A_slip = np.array(table.loc["A1_A2_A3"].Value.split(";")).astype(float)

        self.k_Boltzman = float(table.loc["k_boltzman"].Value)
        self.k_von_Karman = float(table.loc["k_von_karman"].Value)

        self.N_iter = int(table.loc["N_iter"].Value)
        self.tol = float(table.loc["tol"].Value)

        self.Re_Limit = np.array(table.loc["Re_Limit"].Value.split(";")).astype(float)

        self.alpha_EIM = float(table.loc["alpha_EIM"].Value)
        self.beta_EIM = float(table.loc["beta_EIM"].Value)
        self.eps0 = float(table.loc["eps0"].Value)
        self.D0 = float(table.loc["D0"].Value)


@dataclass
class ReflectanceMeasurements:
    """
    Data class for managing reflectance measurement data.
    """

    files: List[Union[str, Path]] = field(
        default_factory=list,
        metadata={"description": "Files from which reflectance data was imported."},
    )
    time_grids: List[Any] = field(
        default_factory=list,
        metadata={
            "description": "A fine grid of times where reflectance measurements are desired (e.g. at times where simulations are available)."
        },
    )
    number_of_measurements: Optional[List[float]] = field(
        default_factory=list,
        metadata={
            "description": "Number of measurements for each file. This should be a float for later operations."
        },
    )
    reflectometer_incidence_angle: Optional[List[float]] = field(
        default_factory=list,
        metadata={
            "description": "Incidence angle of the reflectometer for each file.",
            "units": "degrees",
        },
    )
    reflectometer_acceptance_angle: Optional[List[float]] = field(
        default_factory=list,
        metadata={
            "description": "Half-angle describing the (conical) acceptance solid angle of the reflectometer ",
            "units": "radians",
        },
    )
    import_tilts: bool = False
    imported_column_names: Optional[List[str]] = field(
        default_factory=list,
        metadata={
            "description": "List of column names to import from the reflectance data files."
        },
    )
    verbose: bool = True

    # Internal dictionaries populated in __post_init__
    times: Dict[int, np.ndarray] = field(init=False, default_factory=dict)
    average: Dict[int, np.ndarray] = field(init=False, default_factory=dict)
    soiling_rate: Dict[int, Any] = field(init=False, default_factory=dict)
    delta_ref: Dict[int, Any] = field(init=False, default_factory=dict)
    sigma: Dict[int, np.ndarray] = field(init=False, default_factory=dict)
    sigma_of_the_mean: Dict[int, np.ndarray] = field(init=False, default_factory=dict)
    prediction_indices: Dict[int, Any] = field(init=False, default_factory=dict)
    prediction_times: Dict[int, Any] = field(init=False, default_factory=dict)
    rho0: Dict[int, Any] = field(init=False, default_factory=dict)
    mirror_names: Dict[int, List[str]] = field(init=False, default_factory=dict)
    tilts: Dict[int, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self):
        # Ensure file list
        self.files = _ensure_list(self.files)
        n = len(self.files)

        # Set up defaults or import lists
        if self.number_of_measurements is None:
            self.number_of_measurements = [1.0] * n
        else:
            self.number_of_measurements = _import_option_helper(
                self.files, self.number_of_measurements
            )

        if self.reflectometer_incidence_angle is None:
            self.reflectometer_incidence_angle = [0.0] * n
        else:
            self.reflectometer_incidence_angle = _import_option_helper(
                self.files, self.reflectometer_incidence_angle
            )

        if self.reflectometer_acceptance_angle is None:
            self.reflectometer_acceptance_angle = [0.0] * n
        else:
            self.reflectometer_acceptance_angle = _import_option_helper(
                self.files, self.reflectometer_acceptance_angle
            )

        # Finally, import the data
        self.import_reflectance_data(
            self.time_grids,
            self.reflectometer_incidence_angle,
            self.reflectometer_acceptance_angle,
            import_tilts=self.import_tilts,
            column_names_to_import=self.imported_column_names,
        )

    def import_reflectance_data(
        self,
        time_grids: List[Any],
        incidence_angles: List[float],
        acceptance_angles: List[float],
        import_tilts: bool = False,
        column_names_to_import: Optional[List[str]] = None,
    ):
        """
        Imports reflectance data from Excel source_files into the object's dictionaries.
        """
        for ii, fpath in enumerate(self.files):
            self.files[ii] = fpath
            reflectance_data = {
                "Average": pd.read_excel(fpath, sheet_name="Reflectance_Average"),
                "Sigma": pd.read_excel(fpath, sheet_name="Reflectance_Sigma"),
            }

            # Extract timestamps
            time_column = next(
                (
                    col
                    for col in reflectance_data["Average"].columns
                    if col.lower() in ["time", "timestamp", "tmsmp", "date time"]
                ),
                None,
            )
            if time_column is not None:
                self.times[ii] = reflectance_data["Average"][time_column].values
            else:
                raise ValueError(f"No 'Time' or 'Timestamp' column found in file {fpath}")

            # Import data and ensure proper dimensions, Reflectance assumed to be in % based hence / 100
            if column_names_to_import is not None:
                # Extract selected columns
                avg_data = reflectance_data["Average"][column_names_to_import].values / 100.0
                sig_data = reflectance_data["Sigma"][column_names_to_import].values / 100.0
                self.mirror_names[ii] = column_names_to_import
            else:
                # Extract all columns except the first (time) column
                avg_data = reflectance_data["Average"].iloc[:, 1:].values / 100.0
                sig_data = reflectance_data["Sigma"].iloc[:, 1:].values / 100.0
                self.mirror_names[ii] = list(reflectance_data["Average"].keys())[1:]

            # Ensure 2D arrays for both single and multiple columns
            if avg_data.ndim == 1:
                self.average[ii] = avg_data.reshape(-1, 1)
                self.sigma[ii] = sig_data.reshape(-1, 1)
            else:
                self.average[ii] = avg_data
                self.sigma[ii] = sig_data

            # Calculate delta_ref with proper dimensions
            self.delta_ref[ii] = np.vstack(
                (
                    np.zeros((1, self.average[ii].shape[1])),
                    -np.diff(self.average[ii], axis=0),
                )
            )

            # Set up prediction indices and times
            self.prediction_indices[ii] = []
            self.prediction_times[ii] = []
            for m in self.times[ii]:
                self.prediction_indices[ii].append(np.argmin(np.abs(m - time_grids[ii])))
            self.prediction_times[ii].append(time_grids[ii][self.prediction_indices[ii]])

            # Calculate initial reflectance (rho0), handling NaN values
            self.rho0[ii] = np.nanmax(self.average[ii], axis=0)

            # Set reflectometer parameters
            self.reflectometer_incidence_angle[ii] = incidence_angles[ii]
            self.reflectometer_acceptance_angle[ii] = acceptance_angles[ii]
            self.sigma_of_the_mean[ii] = self.sigma[ii] / np.sqrt(self.number_of_measurements[ii])

            # Import tilts if requested
            if import_tilts:
                tilt_data = pd.read_excel(fpath, sheet_name="Tilts")[self.mirror_names[ii]].values
                if tilt_data.ndim == 1:
                    self.tilts[ii] = tilt_data.reshape(1, -1)  # Single row becomes (1, n_times)
                else:
                    self.tilts[ii] = tilt_data.transpose()  # Shape becomes (n_heliostats, n_times)

    def get_experiment_subset(self, idx):
        attributes = [
            a for a in dir(self) if not a.startswith("__")
        ]  # filters out python standard attributes
        self_out = copy.deepcopy(self)
        for a in attributes:
            attr = self_out.__getattribute__(a)
            if isinstance(attr, dict):
                for k in list(attr.keys()):
                    if k not in idx:
                        attr.pop(k)
        return self_out

    def plot(self):
        files = list(self.average.keys())
        N_mirrors = self.average[0].shape[1]
        N_experiments = len(files)
        fig, ax = plt.subplots(N_mirrors, N_experiments, sharex="col", sharey=True)
        fig.suptitle("Reflectance Data Plot", fontsize=16)
        miny = 1.0
        for ii in range(N_experiments):
            f = files[ii]
            for jj in range(N_mirrors):

                # axis handle
                if N_experiments == 1:
                    a = ax[jj]  # experiment ii, mirror jj plot
                else:
                    a = ax[jj, ii]

                tilt = self.tilts[f][jj]
                if jj == 0:
                    tilt_str = r"Experiment " + str(ii + 1) + r", tilt = ${0:.0f}^{{\circ}}$"
                else:
                    tilt_str = r"tilt = ${0:.0f}^{{\circ}}$"

                if all(tilt == tilt[0]):
                    a.set_title(tilt_str.format(tilt[0]))
                else:
                    a.set_title(tilt_str.format(tilt.mean()) + " (average)")

                a.grid("on")
                m = self.average[f][:, jj]
                s = self.sigma_of_the_mean[f][:, jj]
                miny = min((m - 6 * s).min(), miny)
                error_two_sigma = 1.96 * s
                a.errorbar(
                    self.times[f],
                    m,
                    yerr=error_two_sigma,
                    label="Measurement mean",
                    marker=".",
                )

            a.set_ylabel(
                r"Reflectance at ${0:.1f}^{{\circ}}$".format(
                    self.reflectometer_incidence_angle[ii]
                )
            )
        a.set_ylim((miny, 1))
        a.set_xlabel("Date")
