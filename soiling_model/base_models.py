import numpy as np
from numpy import matlib
from numpy import radians as rad
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import turbo
from warnings import warn
import copy
from sklearn.cluster import KMeans
from pathlib import Path
from typing import Dict
from dataclasses import dataclass, field
from scipy.interpolate import interp2d
from soiling_model.utilities import _print_if,_ensure_list,\
                                    _extinction_function,_same_ext_coeff,\
                                    _import_option_helper,_parse_dust_str
from textwrap import dedent
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.distance import cdist
import copy
from tqdm.notebook import tqdm
import shutil
import os
from scipy.interpolate import RegularGridInterpolator


tol = np.finfo(float).eps # machine floating point precision

class soiling_base:
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
        self.latitude = None                  # latitude in degrees of site
        self.longitude = None                 # longitude in degrees of site
        self.timezone_offset = None           # [hrs from GMT] timezone of site
        self.constants = constants()          # a subclass for constant
        self.helios = helios()                # a subclass containing information about the heliostats
        self.sigma_dep = None                 # standard deviation for deposition velocity
        self.loss_model = None                # either "geometry" or "mie"

    def import_site_data_and_constants(self,file_params,verbose=True):
        
        _print_if(f"\nLoading data from {file_params} ... ",verbose)
        table = pd.read_excel(file_params,index_col="Parameter")

        # optional parameter imports
        try:
            self.latitude = float(table.loc['latitude'].Value)                  # latitude in degrees of site
            self.longitude = float(table.loc['longitude'].Value)                # longitude in degrees of site
            self.timezone_offset = float(table.loc['timezone_offset'].Value)    # [hrs from GMT] timezone of site
        except:
            _print_if(dedent(f"""\
            You are missing at least one of (lat,lon,timezone_offset) in:
            {file_params}
            Field performance cannot be simulated until all of these are defined. """),verbose)
            self.latitude = None
            self.longitude = None
            self.timezone_offset = None

        self.constants.import_constants(file_params,verbose=verbose)

class physical_base(soiling_base):
    def __init__(self):
        super().__init__()
        self.hrz0 =None                       # [-] site roughness height ratio

    def import_site_data_and_constants(self,file_params,verbose=True):
        super().import_site_data_and_constants(file_params)                               
        table = pd.read_excel(file_params,index_col="Parameter")

        try:
            self.loss_model = table.loc['loss_model'].Value     # either "geometry" or "mie"
        except:
            _print_if(f"No loss model defined in {file_params}. You will need to define this before simulating",verbose)

        try:
            self.hrz0 =float(table.loc['hr_z0'].Value)          # [-] site roughness height ratio
        except:
            _print_if(f"No hrz0 model defined in {file_params}. You will need to define this before simulating",verbose)

    def deposition_velocity(self,dust,wind_speed=None,air_temp=None,hrz0=None,verbose=True,Ra=True):
        dust = dust
        constants = self.constants
        if hrz0 == None: # hrz0 from constants file
            hrz0 = self.hrz0
            _print_if("No value for hrz0 supplied. Using value in self.hrz0 = "+str(self.hrz0)+".",verbose)
        else:
            _print_if("Value for hrz0 = "+str(hrz0)+" supplied. Value in self.hrz0 ignored.",verbose)

        # N_sims = sim_in.N_simulations
        # _print_if("Calculating deposition velocity for each of the "+str(N_sims)+" simulations",verbose)

        
        D_meters = dust.D[0]*1e-6  # µm --> m
        Ntimes = len(wind_speed) #.shape[0]

        Cc = 1+2*(constants.air_lambda_p/D_meters)* \
                (constants.A_slip[0]+constants.A_slip[1]*\
                    np.exp(-constants.A_slip[2]*D_meters/constants.air_lambda_p)) # slip correction factor
                
        # computation of the gravitational settling velocity
        vg = (constants.g*(D_meters**2)*Cc*(dust.rho[0]))/(18*constants.air_mu);    # terminal velocity [m/s] if Re<0.1 
        Re = constants.air_rho*vg*D_meters/constants.air_mu                      # Reynolds number for vg(Re<0.1)
        for ii in range(constants.N_iter):
            Cd_g = 24/Re
            Cd_g[Re>constants.Re_Limit[0]] = 24/Re[Re>constants.Re_Limit[0]] * \
                (1 + 3/16*Re[Re>constants.Re_Limit[0]] + 9/160*(Re[Re>constants.Re_Limit[0]]**2)*\
                    np.log(2*Re[Re>constants.Re_Limit[0]]))
            Cd_g[Re>constants.Re_Limit[1]] = 24/Re[Re>constants.Re_Limit[1]] * (1 + 0.15*Re[Re>constants.Re_Limit[1]]**0.687)      
            Cd_g[Re>constants.Re_Limit[2]] = 0.44;      
            vnew = np.sqrt(4*constants.g*D_meters*Cc*dust.rho[0]/(3*Cd_g*constants.air_rho))
            if max(abs(vnew-vg)/vnew)<constants.tol:
                vg = vnew
                break
            vg = vnew
            Re = constants.air_rho*vg*D_meters/constants.air_mu
        if ii == constants.N_iter:
            _print_if('Max iter reached in Reynolds calculation for gravitational settling velocity',verbose)
            
        # computation of the settling velocity due to inertia and diffusion
        u_friction = constants.k_von_Karman*wind_speed/np.log(hrz0)                                           # [m/s] friction velocity
        diffusivity = constants.k_Boltzman/(3*np.pi*constants.air_mu)* \
            np.transpose(matlib.repmat(air_temp+273.15,len(D_meters),1))* \
                matlib.repmat(Cc/D_meters,Ntimes,1)                                                      # [m^2/s] brownian diffusivity (Stokes-Einstein expression)
        Schmidt_number = constants.air_nu/diffusivity                                                               # Schmidt number
        Stokes_number = np.transpose(matlib.repmat((u_friction**2),len(D_meters),1))* \
            vg/constants.air_nu/constants.g                                                                         # Stokes number
        Cd_momentum = constants.k_von_Karman**2/((np.log(hrz0))**2)                                                # drag coefficient for momentum
        E_brownian = Schmidt_number**(-2/3)                                                                         # Brownian factor
        E_impaction = (Stokes_number**constants.beta_EIM)/(constants.alpha_EIM+Stokes_number**constants.beta_EIM)   # Impaction factor (Giorgi, 1986)
        E_interception = 0                                                                                          # Interception factor (=0 in this model)
        R1 = np.exp(-np.sqrt(Stokes_number))                                                                        # 'stick' factor for boundary layer resistance computation
        R1[R1<=tol]=tol                                                                                             # to avoid division by 0
        if Ra:
            aerodynamic_resistance = 1/(Cd_momentum*wind_speed) 
            _print_if('Aerodynamic resistance is considered',verbose)                                                   # [s/m] 
        elif not Ra:
            aerodynamic_resistance = 0
            _print_if('Aerodynamic resistance is neglected',verbose)
        else:
            _print_if('Choose whether or not considering the aerodynamic resistance',verbose)
        
        boundary_layer_resistance = 1/(constants.eps0*\
            np.transpose(matlib.repmat((u_friction),len(D_meters),1))*R1*\
                (E_brownian+E_impaction+E_interception)) # [s/m]
        
        # Rt = np.transpose(matlib.repmat(aerodynamic_resistance,len(D_meters),1))+boundary_layer_resistance
        
        vt = 1/(np.transpose(matlib.repmat(aerodynamic_resistance,len(D_meters),1))\
            +boundary_layer_resistance)   # [m/s]

        vz = (vg + vt).transpose() # [m/s]

        return(aerodynamic_resistance,boundary_layer_resistance,vg,vt,vz)

    def deposition_flux(self,simulation_inputs,hrz0=None,verbose=True,Ra=True):
        sim_in = simulation_inputs
        helios = self.helios
        dust = sim_in.dust
        constants = self.constants
        if hrz0 == None: # hrz0 from constants file
            hrz0 = self.hrz0
            _print_if("No value for hrz0 supplied. Using value in self.hrz0 = "+str(self.hrz0)+".",verbose)
        else:
            _print_if("Value for hrz0 = "+str(hrz0)+" supplied. Value in self.hrz0 ignored.",verbose)

        N_sims = sim_in.N_simulations
        _print_if("Calculating deposition velocity for each of the "+str(N_sims)+" simulations",verbose)

        files = list(sim_in.wind_speed.keys())
        for f in list(files):
            D_meters = dust.D[f]*1e-6  # µm --> m
            Ntimes = len(sim_in.wind_speed[f]) #.shape[0]
            Nhelios = helios.tilt[f].shape[0] 
            Nd = D_meters.shape[0]

            Cc = 1+2*(constants.air_lambda_p/D_meters)* \
                    (constants.A_slip[0]+constants.A_slip[1]*\
                        np.exp(-constants.A_slip[2]*D_meters/constants.air_lambda_p)) # slip correction factor
                    
            # computation of the gravitational settling velocity
            vg = (constants.g*(D_meters**2)*Cc*(dust.rho[f]))/(18*constants.air_mu);    # terminal velocity [m/s] if Re<0.1 
            Re = constants.air_rho*vg*D_meters/constants.air_mu                      # Reynolds number for vg(Re<0.1)
            for ii in range(constants.N_iter):
                Cd_g = 24/Re
                Cd_g[Re>constants.Re_Limit[0]] = 24/Re[Re>constants.Re_Limit[0]] * \
                    (1 + 3/16*Re[Re>constants.Re_Limit[0]] + 9/160*(Re[Re>constants.Re_Limit[0]]**2)*\
                        np.log(2*Re[Re>constants.Re_Limit[0]]))
                Cd_g[Re>constants.Re_Limit[1]] = 24/Re[Re>constants.Re_Limit[1]] * (1 + 0.15*Re[Re>constants.Re_Limit[1]]**0.687)      
                Cd_g[Re>constants.Re_Limit[2]] = 0.44;      
                vnew = np.sqrt(4*constants.g*D_meters*Cc*dust.rho[f]/(3*Cd_g*constants.air_rho))
                if max(abs(vnew-vg)/vnew)<constants.tol:
                    vg = vnew
                    break
                vg = vnew
                Re = constants.air_rho*vg*D_meters/constants.air_mu
            if ii == constants.N_iter:
                _print_if('Max iter reached in Reynolds calculation for gravitational settling velocity',verbose)
                
            # computation of the settling velocity due to inertia and diffusion
            u_friction = constants.k_von_Karman*sim_in.wind_speed[f]/np.log(hrz0)                                           # [m/s] friction velocity
            diffusivity = constants.k_Boltzman/(3*np.pi*constants.air_mu)* \
                np.transpose(matlib.repmat(sim_in.air_temp[f]+273.15,len(D_meters),1))* \
                    matlib.repmat(Cc/D_meters,Ntimes,1)                                                      # [m^2/s] brownian diffusivity (Stokes-Einstein expression)
            Schmidt_number = constants.air_nu/diffusivity                                                               # Schmidt number
            Stokes_number = np.transpose(matlib.repmat((u_friction**2),len(D_meters),1))* \
                vg/constants.air_nu/constants.g                                                                         # Stokes number
            Cd_momentum = constants.k_von_Karman**2/((np.log(hrz0))**2)                                                # drag coefficient for momentum
            E_brownian = Schmidt_number**(-2/3)                                                                         # Brownian factor
            E_impaction = (Stokes_number**constants.beta_EIM)/(constants.alpha_EIM+Stokes_number**constants.beta_EIM)   # Impaction factor (Giorgi, 1986)
            E_interception = 0                                                                                          # Interception factor (=0 in this model)
            R1 = np.exp(-np.sqrt(Stokes_number))                                                                        # 'stick' factor for boundary layer resistance computation
            R1[R1<=tol]=tol                                                                                             # to avoid division by 0
            if Ra:
                aerodynamic_resistance = 1/(Cd_momentum*sim_in.wind_speed[f]) 
                _print_if('Aerodynamic resistance is considered',verbose)                                                   # [s/m] 
            elif not Ra:
                aerodynamic_resistance = 0
                _print_if('Aerodynamic resistance is neglected',verbose)
            else:
                _print_if('Choose whether or not considering the aerodynamic resistance',verbose)
            
            boundary_layer_resistance = 1/(constants.eps0*\
                np.transpose(matlib.repmat((u_friction),len(D_meters),1))*R1*\
                    (E_brownian+E_impaction+E_interception)) # [s/m]
            
            # Rt = np.transpose(matlib.repmat(aerodynamic_resistance,len(D_meters),1))+boundary_layer_resistance
            
            vt = 1/(np.transpose(matlib.repmat(aerodynamic_resistance,len(D_meters),1))\
                +boundary_layer_resistance)   # [m/s]
            
            # computation of vertical deposition velocity
            vz = (vg + vt).transpose() # [m/s]
            
            helios.pdfqN[f] = np.empty((Nhelios,Ntimes,Nd))
            for idx in range(helios.tilt[f].shape[0]):
                Fd = np.cos(rad(helios.tilt[f][idx,:]))*vz   # Flux per unit concentration at each time, for each heliostat [m/s] (Eq. 28 in [1] without Cd)
                if Fd.min() < 0:
                    warn("Deposition velocity is negative (min value: "+str(Fd.min())+"). Setting negative components to zero.")
                    Fd[Fd<0]=0
                helios.pdfqN[f][idx,:,:] = Fd.transpose()*dust.pdfN[f]  # Dust flux pdf, i.e. [dq[particles/(s*m^2)]/dLog_{10}(D[µm]) ] deposited on 1m2. 
            
        self.helios = helios
        
    def adhesion_removal(self,simulation_inputs,verbose=True):
        _print_if("Calculating adhesion/removal balance",verbose)
        helios = self.helios
        dust = simulation_inputs.dust
        dt = simulation_inputs.dt
        constants = self.constants
        files = list(simulation_inputs.time.keys())
        
        for f in files:
            D_meters = dust.D[f]*1e-6  # Change to µm
            youngs_modulus_composite = 4/3*((1-dust.poisson[f]**2)/dust.youngs_modulus[f] + \
                (1-helios.poisson**2)/helios.youngs_modulus)**(-1);                             # [N/m2] composite Young modulus 
            hamaker_system = np.sqrt(dust.hamaker[f]*helios.hamaker)                            # [J] system Hamaker constant (Israelachvili)
            work_adh = hamaker_system/(12*np.pi*constants.D0**2)                                # [J/m^2] work of adhesion
            radius_sep = ((3*np.pi*work_adh*D_meters**2)/(8*youngs_modulus_composite))**(1/3)   # [m] contact radius at separation (JKR model)
            F_adhesion = 3/4*np.pi*work_adh*D_meters                                            # [N] van der Waals adhesion force (JKR model)
            F_gravity = dust.rho[f]*np.pi/6*constants.g*D_meters**3                             # [N] weight force   

            if helios.stow_tilt == None: # No common stow angle supplied. Need to use raw tilts to compute removal moments
                _print_if("  No common stow_tilt. Use values in helios.tilt to compute removal moments. This might take some time.",verbose)
                Nhelios = helios.tilt[f].shape[0]
                Ntimes = helios.tilt[f].shape[1]
                helios.pdfqN[f] = cumulative_trapezoid(y=helios.pdfqN[f],dx=dt[f],axis=1,initial=0) # Accumulate in time so that we ensure we remove all dust present on mirror if removal condition is satisfied at a particular time
                for h in range(Nhelios):
                    for k in range(Ntimes):
                        mom_removal = np.sin(rad(helios.tilt[f][h,k]))* F_gravity*np.sqrt((D_meters**2)/4-radius_sep**2) # [Nm] removal moment exerted by gravity at each tilt for each diameter
                        mom_adhesion =  (F_adhesion+F_gravity*np.cos(rad(helios.tilt[f][h,k])))*radius_sep             # [Nm] adhesion moment  
                        helios.pdfqN[f][h,k:,mom_adhesion<mom_removal] = 0 # ALL dust desposited at this diameter up to this point falls off
                        # if any(mom_adhesion<mom_removal):
                        #     _print_if("Some dust is removed",verbose)

                
                helios.pdfqN[f] = np.gradient(helios.pdfqN[f],dt[f],axis=1) # Take derivative so that pdfqN is the rate at wich dust is deposited at each diameter

            else: # common stow angle at night for all heliostats. Assumes tilt at night is close to vertical at night.
                # Since the heliostats are stowed at a large tilt angle at night, we assume that any dust that falls off at this stow
                # is never deposited. This introduces a small error since the dust deposited during the day never affects the reflectance, but faster computation.
                _print_if("  Using common stow_tilt. Assumes all heliostats are stored at helios.stow_tilt at night.",verbose)
                mom_removal = np.sin(rad(helios.stow_tilt))* F_gravity*np.sqrt((D_meters**2)/4-radius_sep**2) # [Nm] removal moment exerted by gravity
                mom_adhesion =  (F_adhesion+F_gravity*np.cos(rad(helios.stow_tilt)))*radius_sep             # [Nm] adhesion moment
                helios.pdfqN[f][:,:,mom_adhesion<mom_removal] = 0 # Remove this diameter from consideration
        
        self.helios = helios
    
    def calculate_delta_soiled_area(self,simulation_inputs,sigma_dep=None,verbose=True): 
        
        # info and error checking
        _print_if("Calculating soil deposited in a timestep [m^2/m^2]",verbose)
        
        sim_in = simulation_inputs
        helios = self.helios
        dust = sim_in.dust
        extinction_weighting = helios.extinction_weighting
        
        files = list(sim_in.wind_speed.keys())
        for f in files:
            D_meters = dust.D[f]*1e-6
            helios.delta_soiled_area[f] = np.empty((helios.tilt[f].shape[0],helios.tilt[f].shape[1]))
            
            if sigma_dep is not None or self.sigma_dep is not None:
                helios.delta_soiled_area_variance[f] = np.empty((helios.tilt[f].shape[0],helios.tilt[f].shape[1]))

            # compute alpha
            try:
                attr = _parse_dust_str(sim_in.dust_type[f])
                den = getattr(dust,attr) # dust.(sim_in.dust_type[f])
            except:
                raise ValueError("Dust measurement = "+sim_in.dust_type[f]+\
                    " not present in dust class. Use dust_type="+sim_in.dust_type[f]+\
                        " option when loading the simulation data.")
            alpha = sim_in.dust_concentration[f]/den[f]

            # Compute the area coverage by dust at each time step
            N_helios = helios.tilt[f].shape[0]
            N_times = helios.tilt[f].shape[1]
            for ii in range(N_helios):
                for jj in range(N_times):

                    # if loss_model == 'geometry':
                    #     # The below two integrals are equivalent, but the version with the log10(D)
                    #     # as the independent variable is used due to the log spacing of the diameter grid
                    #     #
                    #     # helios.delta_soiled_area[f][ii,jj] = alpha[jj] * np.trapz(helios.pdfqN[f][ii,jj,:]*\
                    #     #     (np.pi/4*D_meters**2)*sim_in.dt[f]/dust.D[f]/np.log(10),dust.D[f])
                        
                    #     helios.delta_soiled_area[f][ii,jj] = alpha[jj] * np.pi/4 *np.trapz(helios.pdfqN[f][ii,jj,:]*\
                    #         (D_meters**2)*sim_in.dt[f],np.log10(dust.D[f]))
                    # else: # loss_model == "mie"
                    helios.delta_soiled_area[f][ii,jj] = alpha[jj] * np.pi/4 * np.trapz(helios.pdfqN[f][ii,jj,:]*\
                        (D_meters**2)*sim_in.dt[f]*extinction_weighting[f][ii,:],np.log10(dust.D[f])) # pdfqN includes cos(tilt)

            # variance of noise for each measurement
            if sigma_dep is not None:
                theta = np.radians(self.helios.tilt[f])
                helios.delta_soiled_area_variance[f] = sigma_dep**2 * (alpha**2*np.cos(theta)**2)
                # sigma_dep**2*helios.inc_ref_factor[f]*np.cumsum(alpha**2*np.cos(theta)**2,axis=1)
            elif self.sigma_dep is not None:
                theta = np.radians(self.helios.tilt[f])
                helios.delta_soiled_area_variance[f] = self.sigma_dep**2 * (alpha**2*np.cos(theta)**2)

        self.helios = helios
    
    def plot_area_flux(self,sim_data,exp_idx,hel_id,air_temp,wind_speed,
                        tilt=0.0,hrz0=None,constants=None,
                        ax=None,Ra=True,verbose=True):
        
        dummy_sim = simulation_inputs()

        for att_name in sim_data.dust.__dict__.keys():
            val = {0:getattr(sim_data.dust,att_name)[exp_idx]}
            setattr(dummy_sim.dust,att_name,val)
        
        # dummy_sim.dust.import_dust(dust_file,verbose=False,dust_measurement_types="PM10")
        dummy_sim.air_temp = {0:np.array([air_temp])}
        dummy_sim.wind_speed = {0:np.array([wind_speed])}
        dummy_sim.dt = {0:1.0}
        dummy_sim.dust_type = {0:"PM10"}                    # this doesn't matter for this function
        dummy_sim.dust_concentration = {0:np.array([dummy_sim.dust.PM10[0]])}   # makes alpha = 1
        dummy_sim.N_simulations = 1

        if self.loss_model == "mie":
            dummy_sim.source_normalized_intensity = {0:sim_data.source_normalized_intensity[exp_idx]}
            dummy_sim.source_wavelength = {0:sim_data.source_wavelength[exp_idx]}
            acceptance_angle = self.helios.acceptance_angles[exp_idx][hel_id]
            _print_if("Loss model is ""mie"" ",verbose)
        else:
            _print_if("Loss model is ""geometry"". Extinction weights are unity for all diameters.",verbose)
            acceptance_angle = np.nan

        dummy_model = copy.deepcopy(self)
        dummy_model.helios = helios()
        dummy_model.helios.tilt = {0:np.array([[tilt]])}
        dummy_model.sigma_dep = None
        dummy_model.loss_model = self.loss_model
        # dummy_model.helios.acceptance_angles = [acceptance_angle]
        # dummy_model.helios.extinction_weighting = {0:np.atleast_2d(self.helios.extinction_weighting[exp_idx][0,:])}
        dummy_model.helios.extinction_weighting = {0:np.atleast_2d(self.helios.extinction_weighting[exp_idx][hel_id,:])}
        
        fmt = "Setting constants.{0:s} to {1:s} (was {2:s})"
        if constants is not None:
            for kk in constants.keys():
                temp = str(getattr(dummy_model.constants,kk))
                print(fmt.format(str(kk), str(constants[kk]),temp))
                setattr(dummy_model.constants,kk,constants[kk])

        if hrz0 is None:
            hrz0 = dummy_model.hrz0        
            dummy_model.deposition_flux(dummy_sim,Ra=Ra)
        else:
            dummy_model.deposition_flux(dummy_sim,hrz0=hrz0,Ra=Ra)

        dummy_model.calculate_delta_soiled_area(dummy_sim)

        if ax is None:
            _,ax1 = plt.subplots()
        else:
            ax1 = ax

        title = f'''
            Area loss rate for given dust distribution at acceptance angle {acceptance_angle*1e3:.2f} mrad,
            wind_speed= {wind_speed:.1f} m/s, air_temperature={air_temp:.1f} C
            (total area loss is {dummy_model.helios.delta_soiled_area[0][0,0]:.2e} m$^2$/(s$\\cdot$m$^2$))
        '''
        area_loss_rate = (dummy_model.helios.pdfqN[0][0,0,:]*np.pi/4*dummy_sim.dust.D[0]**2*1e-12*dummy_model.helios.extinction_weighting[0][0,:])
        ax1.plot(dummy_sim.dust.D[0],area_loss_rate)
        ax1.set_title(title.format(wind_speed,air_temp,))
        ax1.set_xlabel(r"D [$\mu$m]")
        ax1.set_ylabel(r'$\frac{dA [m^2/m^2/s] }{dLog(D \;[\mu m])}$', color='black',size=20)
        plt.xscale('log')   
        ax1.set_xticks([0.001,0.01,0.1,1,2.5,4,10,20,100])

class constant_mean_base(soiling_base):
    def __init__(self):
        super().__init__()
        self.mu_tilde = None
    
    def import_site_data_and_constants(self,file_params,verbose=True):
        super().import_site_data_and_constants(file_params)                               
        table = pd.read_excel(file_params,index_col="Parameter")
        try:
            self.mu_tilde =float(table.loc['mu_tilde'].Value)          # [-] constant average deposition
        except:
            _print_if(f"No mu_tilde model defined in {file_params}. You will need to define this before simulating",verbose)
        try: 
            self.sigma_dep = float(table.loc['sigma_dep'].Value)
        except: 
            _print_if(f"No sigma_dep model defined in {file_params}.",verbose)

    def calculate_delta_soiled_area(self,simulation_inputs,mu_tilde=None,sigma_dep=None,verbose=True):

        _print_if("Calculating soil deposited in a timestep [m^2/m^2]",verbose)
        
        sim_in = simulation_inputs
        helios = self.helios
        dust = sim_in.dust

        if mu_tilde == None: # use value in self
            mu_tilde = self.mu_tilde
        else:
            mu_tilde = mu_tilde
            _print_if("Using supplied value for mu_tilde = "+str(mu_tilde),verbose)

        if sigma_dep is not None or self.sigma_dep is not None:
            if sigma_dep == None: # use value in self
                sigma_dep = self.sigma_dep
            else:
                sigma_dep = sigma_dep
                _print_if("Using supplied value for sigma_dep = "+str(sigma_dep),verbose)
        
        files = list(sim_in.time.keys())
        for f in files:
            helios.delta_soiled_area[f] = np.empty((helios.tilt[f].shape[0],helios.tilt[f].shape[1]))

            # compute alpha
            try:
                attr = _parse_dust_str(sim_in.dust_type[f])
                den = getattr(dust,attr) # dust.(sim_in.dust_type[f])
            except:
                raise ValueError("Dust measurement = "+sim_in.dust_type[f]+\
                    " not present in dust class. Use dust_type="+sim_in.dust_type[f]+\
                        " option when initializing the model")

            alpha = sim_in.dust_concentration[f]/den[f]

            # Compute the area coverage by dust at each time step
            N_helios = helios.tilt[f].shape[0]
            N_times = helios.tilt[f].shape[1]
            for ii in range(N_helios):
                for jj in range(N_times):
                    helios.delta_soiled_area[f][ii,jj] = \
                        alpha[jj] * np.cos(rad(helios.tilt[f][ii,jj]))*mu_tilde

            # Predict confidence interval if sigma_dep is defined. Fixed tilt assumed in this class. 
            if sigma_dep is not None:
                theta = np.radians(self.helios.tilt[f])
                inc_factor = self.helios.inc_ref_factor[f]
                dsav = sigma_dep**2* (alpha**2*np.cos(theta)**2)
                
                helios.delta_soiled_area_variance[f] = dsav
                self.helios.soiling_factor_prediction_variance[f] = \
                    np.cumsum( inc_factor**2 * dsav,axis=1 )

        self.helios = helios

class simulation_inputs:
    """
    Defines a `simulation_inputs` class that manages the input data for a soiling model simulation.
    
    The class provides methods to import weather and dust data from Excel files, and stores the data in dictionaries
    with the file number as the key. The class also includes a `dust` attribute that stores the dust properties
    for each experiment.
    
    The `import_weather` method reads weather data such as air temperature, wind speed, dust concentration, etc.
    from the Excel files and stores them in the corresponding dictionaries.
    
    The `import_source_intensity` method reads the source intensity data from the Excel files and stores it in
    the `source_wavelength` and `source_normalized_intensity` dictionaries.
    
    The `get_experiment_subset` method creates a copy of the `simulation_inputs` object with only the specified
    experiments included.
    """
    def __init__(self,experiment_files=None,k_factors=None,dust_type=None,verbose=True):

        # the below will be dictionaries of 1D arrays with file numbers as keys 
        self.file_name = {}                     # name of the input file
        self.dt = {}                            # [seconds] simulation time step
        self.time = {}                          # absolute time (taken from 1st Jan)
        self.time_diff = {}                     # [days] delta_time since start date
        self.start_datetime = {}                # datetime64 for start 
        self.end_datetime = {}                  # datetime64 for end
        self.air_temp = {}                      # [C] air temperature
        self.wind_speed = {}                    # [m/s] wind speed
        self.wind_speed_mov_avg = {}            # [m/s] wind speed hourly moving average
        self.wind_direction = {}                # [degrees] wind direction
        self.dust_concentration = {}            # [µg/m3] PM10 or TSP concentration in air
        self.dust_conc_mov_avg = {}             # [µg/m3] PM10 or TSP hourly moving average of dust concentration
        self.rain_intensity = {}                # [mm/hr] rain intensity
        self.dust_type = {}                     # Usually either "TSP" or "PM10", but coule be any PMX or PMX.X
        self.dni = {}                           # [W/m^2] Direct Normal Irradiance
        self.relative_humidity = {}             # [%] relative humidity
        self.source_normalized_intensity = {}   # [1/m^2/nm] normalized source intensity
        self.source_wavelength = {}             # [nm] source wavelengths corersponding to source_intensity 

        self.dust = dust()                      # dust properties will be per experiment

        # if experiment files are supplied, import
        if experiment_files is not None:
            experiment_files = _ensure_list(experiment_files)
            self.N_simulations = len(experiment_files)

            if k_factors == None: 
                k_factors = [1.0]*len(experiment_files)
            elif k_factors == "import": # import k-factors from parameter file
                k_factors = []
                for f in experiment_files:
                    k_factors.append(pd.read_excel(f,sheet_name="Dust",index_col="Parameter").loc['k_factor'].values[0])
            else:
                k_factors = _import_option_helper(experiment_files,k_factors)
                if len(k_factors) != len(experiment_files):
                    raise ValueError("Please specify a k-factor for each weather file")

            self.k_factors = {ii:k_factors[ii] for ii in range(self.N_simulations)} 
            self.import_weather(experiment_files,dust_type,verbose=verbose)
            self.dust.import_dust(experiment_files,verbose=verbose,dust_measurement_type=dust_type)

            # will import source intensity if the sheet exists
            self.import_source_intensity(experiment_files,verbose=verbose)

    def import_source_intensity(self,files,verbose=True):
        for ii,f in enumerate(files):
            xl = pd.ExcelFile(f)
            if "Source_Intensity" in xl.sheet_names:
                _print_if(f"Loading source (normalized) intensity from {f}",verbose)
                intensity = xl.parse("Source_Intensity")
                self.source_wavelength[ii] = intensity['Wavelength (nm)'].to_numpy()
                self.source_normalized_intensity[ii] = intensity['Source Intensity (W/m^2 nm)'].to_numpy()
                norm = np.trapz(y=self.source_normalized_intensity[ii],x=self.source_wavelength[ii])
                self.source_normalized_intensity[ii] = self.source_normalized_intensity[ii]/norm # make sure intensity is normalized for later computations
            else:
                self.source_normalized_intensity[ii] = None
            xl.close()


    def import_weather(self, files, dust_type, verbose=True, smallest_windspeed=1e-6):
        files = _ensure_list(files)
        dust_type = _import_option_helper(files, dust_type)
        
        weather_variables = { # List of possible weather variable names and the combination of possibly names
            'air_temp': ['airtemp', 'temperature', 'temp', 'ambt', 't1'],
            'wind_speed': ['windspeed', 'ws', 'wind_speed'],
            'dni': ['dni', 'directnormalirradiance'],
            'rain_intensity': ['rainintensity', 'precipitation'],
            'relative_humidity': ['rh', 'relativehumidity', 'rhx'],
            'wind_direction': ['wd', 'winddirection']
        }
        
        dust_names = { # List of possible dust concentration names and the combination of possibly names
            'pm_tot': ['pm_tot', 'pmtot', 'pmt', 'pm20'],
            'tsp': ['tsp'],
            'pm10': ['pm10'],
            'pm2p5': ['pm2_5', 'pm2p5', 'pm2.5'],
            'pm1': ['pm1'],
            'pm4': ['pm4']
        }

        self.weather_variables = []

        for ii, file in enumerate(files): # Loop through each campaign and import weather files
            self.file_name[ii] = file
            if file.endswith('.csv'):
                raise ValueError("Please use an excel file for data file")
            weather = pd.read_excel(
                file, 
                sheet_name="Weather"
            )
            # Look for time column with different possible names
            time_column = None
            for col in weather.columns:
                if col.lower() in ['time', 'timestamp', 'date', 'datetime', 'date time']:
                    time_column = col
                    break
                
            if time_column is None:
                raise ValueError(f"No time column found in file {file}. Expected column names: 'Time', 'Timestamp', 'Date', 'DateTime', or 'Date Time'")
            
            # Convert to datetime and round to minutes
            weather[time_column] = pd.to_datetime(weather[time_column]).dt.round("min")
            time = pd.to_datetime(weather[time_column])
            self.start_datetime[ii] = time.iloc[0]
            self.end_datetime[ii] = time.iloc[-1]
            
            _print_if(f"Importing site data (weather,time). Using dust_type = {dust_type[ii]}, test_length = {(self.end_datetime[ii]-self.start_datetime[ii]).days} days", verbose)
            
            self.time[ii] = time
            self.dt[ii] = (self.time[ii][1] - self.time[ii][0]).total_seconds()
            self.time_diff[ii] = (self.time[ii].values - self.time[ii].values.astype('datetime64[D]')).astype('timedelta64[h]').astype('int')

            for attr_name, column_names in weather_variables.items(): # Search for weather variables inside the weather file and save them to self
                for column in column_names:
                    if column in [col.lower() for col in weather.columns]:
                        setattr(self, attr_name, {}) if not hasattr(self, attr_name) else None
                        col_match = [col for col in weather.columns if col.lower() == column][0]
                        getattr(self, attr_name)[ii] = np.array(weather.loc[:, col_match])
                        _print_if(f"Importing {col_match} data as {attr_name}...", verbose)
                        if attr_name not in self.weather_variables:
                            self.weather_variables.append(attr_name)
                        break

            if hasattr(self, 'wind_speed') and ii in self.wind_speed:
                idx_too_low = np.where(self.wind_speed[ii] == 0)[0]
                if len(idx_too_low) > 0:
                    self.wind_speed[ii][idx_too_low] = smallest_windspeed
                    _print_if(f"Warning: some windspeeds were <= 0 and were set to {smallest_windspeed}", verbose)
            self.wind_speed_mov_avg[ii] = pd.Series(self.wind_speed[ii]).rolling(window=int(60.0/(self.dt[ii]/60)), min_periods=1).mean().values

            self.dust_concentration[ii] = self.k_factors[ii] * np.array(weather.loc[:, dust_type[ii]]) # Set dust concentration to be used for soiling predictions
            if 'dust_concentration' not in self.weather_variables:
                self.weather_variables.append('dust_concentration')
            self.dust_type[ii] = dust_type[ii]

            for dust_key, dust_aliases in dust_names.items(): # Load all dust concentration data inside weather file
                for alias in dust_aliases:
                    if alias in [col.lower() for col in weather.columns]:
                        col_match = [col for col in weather.columns if col.lower() == alias][0]
                        dust_value = np.array(weather.loc[:, col_match])
                        if not hasattr(self, dust_key.lower()):
                            setattr(self, dust_key.lower(), {})
                        getattr(self, dust_key.lower())[ii] = dust_value
                        _print_if(f"Importing {dust_key} data...", verbose)
                        if dust_key.lower() not in self.weather_variables:
                            self.weather_variables.append(dust_key.lower())
                        break
            
            self.dust_conc_mov_avg[ii] = pd.Series(self.dust_concentration[ii]).rolling(window=int(60.0/(self.dt[ii]/60)), min_periods=1).mean().values

            if verbose:
                T = (time.iloc[-1] - time.iloc[0]).days
                _print_if(f"Length of simulation for file {file}: {T} days", verbose)                
    def get_experiment_subset(self,idx):
        attributes = [a for a in dir(self) if not a.startswith("__")] # filters out python standard attributes
        self_out = copy.deepcopy(self)
        for a in attributes:
            attr = self_out.__getattribute__(a)
            if isinstance(attr,dict):
                for k in list(attr.keys()):
                    if k not in idx:
                        attr.pop(k)
        return self_out

class dust:
    def __init__(self):
        self.D     = {}          # [µm] dust particles diameter 
        self.rho   = {}          # [kg/m^3] particle material density
        self.m     = {}          # [-] complex refractive index
        self.pdfN  = {}          # "pdf" of dust number d(N [1/cm3])/d(log10(D[µm]))
        self.pdfM  = {}          # "pdf" of dust mass dm[µg/m3]/dLog10(D[µm])
        self.pdfA  = {}          # "pdf" of dust mass dm[µg/m3]/dLog10(D[µm])
        self.hamaker = {}        # [J] hamaker constant of dust  
        self.poisson = {}        # [-] poisson ratio of dust
        self.youngs_modulus = {} # [Pa] young's modulus of dust
        self.PM10 = {}           # [µg/m^3] PM10 concentration computed with the given dust size distribution
        self.TSP = {}            # [µg/m^3] TSP concentration computed with the given dust size distribution
        self.PMT = {}            # [µg/m^3] PMT concentration computed with the given dust size distribution        
        self.Nd = {}
        self.log10_mu = {}
        self.log10_sig = {}
    
    def import_dust(self,experiment_files,verbose=True,dust_measurement_type=None):
        
        _print_if("Importing dust properties for each experiment",verbose)
        experiment_files = _ensure_list(experiment_files)
        dust_measurement_type = _import_option_helper(experiment_files,dust_measurement_type)
        
        for ii,f in enumerate(experiment_files):
            table = pd.read_excel(f,sheet_name="Dust",index_col="Parameter")
            rhoii = float(table.loc['rho'].Value)
            self.rho[ii] = rhoii
            self.m[ii] = table.loc['refractive_index_real_part'].Value - \
                            table.loc['refractive_index_imaginary_part'].Value*1j

            # definition of parameters to compute the dust size distribution
            diameter_grid_info = np.array(table.loc['D'].Value.split(';')) # [µm]
            diameter_end_points = np.log10(diameter_grid_info[0:2].astype('float'))
            spacing = diameter_grid_info[2].astype('int')
            Dii = np.logspace(diameter_end_points[0],diameter_end_points[1],num=spacing)
            self.D[ii] = Dii
            
            if isinstance(table.loc['Nd'].Value,str): # if this is imported as a string, we need to split it.
                self.Nd[ii] = np.array(table.loc['Nd'].Value.split(';'),dtype=float)
                self.log10_mu[ii] = np.log10(np.array(table.loc['mu'].Value.split(';'),dtype=float))
                self.log10_sig[ii] = np.log10(np.array(table.loc['sigma'].Value.split(';'),dtype=float))
            elif isinstance(table.loc['Nd'].Value,float): # handle single-component case
                self.Nd[ii] = np.array([table.loc['Nd'].Value])
                self.log10_mu[ii] = np.log10([np.array(table.loc['mu'].Value)])
                self.log10_sig[ii] = np.log10([np.array(table.loc['sigma'].Value)])
            else:
                raise ValueError("Format of dust distribution components is not recognized in file {0:s}".format(f))
                
            # computation of the dust size distribution
            N_components = len(self.Nd[ii])
            nNd = np.zeros((len(Dii),N_components))
            for jj in range(N_components):
                Ndjj = self.Nd[ii][jj]
                lsjj = self.log10_sig[ii][jj]
                lmjj = self.log10_mu[ii][jj]
                nNd[:,jj] = Ndjj/(np.sqrt(2*np.pi)*lsjj)*np.exp(-(np.log10(Dii)-lmjj)**2/(2*lsjj**2))

            pdfNii = np.sum(nNd,axis=1)*1e6 # pdfN (number) distribution dN[m^-3]/dLog10(D[µm]), 1e6 factor from { V(cm^3->m^3) 1e6 }
            self.pdfN[ii] = pdfNii
            self.pdfA[ii] = pdfNii*(np.pi/4*Dii**2)*1e-12 # pdfA (area) dA[m^2/m^3]/dLog10(D[µm]), 1e-12 factor from { D^2(µm^2->m^2) 1e-12}
            self.pdfM[ii] = pdfNii*(rhoii*np.pi/6*Dii**3)*1e-9 # pdfm (mass) dm[µg/m^3]/dLog10(D[µm]), 1e-9 factor from { D^3(µm^3->m^3) 1e-18 , m(kg->µg) 1e9}
            self.TSP[ii] = np.trapz(self.pdfM[ii],np.log10(Dii)) 
            self.PMT[ii] = self.TSP[ii]
            self.PM10[ii] = np.trapz(self.pdfM[ii][Dii<=10],np.log10(Dii[Dii<=10]))  # PM10 = np.trapz(self.pdfM[self.D<=10],dx=np.log10(self.D[self.D<=10]))

            self.hamaker[ii] = float(table.loc['hamaker_dust'].Value)
            self.poisson[ii] = float(table.loc['poisson_dust'].Value)
            self.youngs_modulus[ii] = float(table.loc['youngs_modulus_dust'].Value)

        # add dust measurements if they are PMX
        for dt in dust_measurement_type:
            if dt not in [None,"TSP","PMT"]: # another concentration is of interest (possibly because we have PMX measurements)
                X = dt[2::]
                if len(X) in [1,2]: # integer, e.g. PM20
                    X = int(X)
                    att = "PM{0:d}".format(X)
                elif len(X)==3: # decimal, e.g. PM2.5
                    att = "PM"+"_".join(X.split('.'))
                    X = float(X)
            
                new_meas = {f: None for f,_ in enumerate(experiment_files)}
                for ii,_ in enumerate(experiment_files):
                    new_meas[ii] = np.trapz(self.pdfM[ii][Dii<=X],np.log10(Dii[Dii<=X]))
            
                setattr(self,att,new_meas)
                _print_if("Added "+att+" attribute to dust class to all experiment dust classes",verbose)

    def plot_distributions(self,figsize=(5,5)):
        N_files = len(self.D)
        fig,ax1 = plt.subplots(nrows=N_files,sharex=True,squeeze=False,figsize=figsize)

        ax2 = []
        for ff in range(N_files):
            D_dust = self.D[ff]
            pdfN = self.pdfN[ff]
            pdfM = self.pdfM[ff]

            color = 'tab:red'
            ax1[ff,0].set_xlabel(r"D [$\mu$m]")
            ax1[ff,0].set_ylabel(r'$\frac{dN [m^{{-3}} ] }{dLog(D \;[\mu m])}$', color=color,size=20)
            ax1[ff,0].plot(D_dust,pdfN, color=color)
            ax1[ff,0].tick_params(axis='y', labelcolor=color)
            ax1[ff,0].grid('on')

            ax2.append(ax1[ff,0].twinx())  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2[ff].set_ylabel(r'$\frac{dm \; [\mu g \, m^{{-3}} ] }{dLog(D \; [\mu m])}$', color=color,size=20)  # we already handled the x-label with ax1
            ax2[ff].plot(D_dust,pdfM, color=color)
            ax2[ff].tick_params(axis='y', labelcolor=color)
            ax2[ff].grid('on')
        
        plt.xscale('log')
        ax2[-1].set_xticks(10.0**np.arange(np.log10(D_dust[0]),np.log10(D_dust[-1]),1))
        plt.tight_layout()
        fig.suptitle("Number and Mass PDFs")

        return fig,ax1,ax2

    def plot_area_distribution(self,figsize=(5,5)):
        N_files = len(self.D)
        _,ax1 = plt.subplots(nrows=N_files,sharex=True,squeeze=False,figsize=figsize)
        
        for ii in range(N_files):
            D_dust = self.D[ii]
            pdfA = self.pdfA[ii]

            color = 'black'
            ax1[ii,0].set_xlabel(r"D [$\mu$m]")
            ax1[ii,0].set_ylabel(r'$\frac{dA [m^2/m^3] }{dLog(D \;[\mu m])}$', color=color,size=20)
            ax1[ii,0].plot(D_dust,pdfA, color=color)
            ax1[ii,0].tick_params(axis='y', labelcolor=color)
            plt.xscale('log')
            ax1[ii,0].set_title("Area PDF")
            ax1[ii,0].set_xticks(10.0**np.arange(np.log10(D_dust[0]),np.log10(D_dust[-1]),1))

        return ax1

@dataclass
class TruckParameters:
    """Default parameters for cleaning truck configuration."""
    # Cost parameters
    cost_water: float = field(default=0.87,
        metadata={'units': '$/kL',
                 'description': 'Cost of water'})
    usage_water: float = field(default=0.4,
        metadata={'units': 'L/m²',
                 'description': 'Water usage per square meter cleaned'})
    cost_fuel: float = field(default=2.0,
        metadata={'units': '$/L',
                 'description': 'Cost of fuel'})
    usage_fuel: float = field(default=25.0,
        metadata={'units': 'L/hour',
                 'description': 'Fuel consumption rate'})
    salary_operator: float = field(default=80e3,
        metadata={'units': '$/year',
                'description': 'Operator salary'})
    cost_purchase: float = field(default=150e3,
        metadata={'units': '$/truck',
                'description': 'Cost of truck'})
    cost_maintenance: float = field(default=15e3,
        metadata= {'units': '$/year',
            'description': 'Annual maintenance cost'})
    useful_life: float = field(default=10.0,
        metadata={'units': 'years',
                'description': 'Useful life of truck'})
    # Velocities 
    velocity_cleaning: float = field(default=2.0,
        metadata={'units': 'km/h',
                 'description': 'Truck velocity during cleaning'})
    velocity_travel: float = field(default=20.0,
        metadata={'units': 'km/h',
                 'description': 'Truck velocity during travel'})
    # Times
    time_setup: float = field(default=30.0,
        metadata={'units': 'seconds/heliostat',
                 'description': 'Setup time per heliostat'})
    time_shift: float = field(default=8.0,
        metadata={'units': 'hours',
                 'description': 'Duration of cleaning shift'})
    # Distances and volumes
    distance_reload_station: float = field(default=750.0,
        metadata={'units': 'm',
                 'description': 'Distance to reload station'})
    truck_water_volume: float = field(default=15000.0,
        metadata={'units': 'L',
                 'description': 'Water tank capacity'})
    # Heliostat dimensions
    heliostat_width: float = field(default=None,
        metadata={'units': 'm',
                 'description': 'Width of heliostat'})
    heliostat_height: float = field(default=None,
        metadata={'units': 'm',
                 'description': 'Height of heliostat'})

class Truck:
    """Truck class with parameter management system that will automatically update cleaning sectors and cleaning rate with each update."""
    def __init__(self, config_path: Path = None):
        self._params = TruckParameters()
        self._solar_field = None # Solarfield ID and positions with respect to receiver (m) [ID, x-x, y-y]
        self._cleaning_rate = None # Number of heliostats cleaned per truck per shift
        self._sectors = None # Number of cleaning sectors to create in the field
        self._n_sectors_per_truck = None # Number of sectors cleaned per truck per shift
        self._consumable_costs = {
            'water': None,
            'fuel': None,
            'total': None
        }
        if config_path:
            self.load_config(Path(config_path))
            
    @property
    def consumable_costs(self) -> dict:
        """Get current cleaning costs per sector."""
        if self._consumable_costs['water'] is None or self._consumable_costs['fuel'] is None:
            self._calculate_costs()
        return self._consumable_costs

    def _calculate_costs(self) -> None:
        """Calculate water and fuel costs per cleaning sector."""
        if not all([hasattr(self._params, attr) for attr in ['heliostat_width', 'heliostat_height']]):
            raise ValueError("Must set heliostat dimensions before calculating costs")

        p = self._params

        # Calculate area per cleaning sector
        area_heliostat_cleaning_sector = (
            p.heliostat_width * # [m]
            p.heliostat_height * # [m]
            self.cleaning_rate / # [heliostats/shift]
            self.n_sectors_per_truck # [sectors/shift]
        )  # [m²/sector]

        # Calculate water cost per sector
        self._consumable_costs['water'] = (
            p.usage_water * # [L/m²]
            p.cost_water /1e3 * # [$/L]
            area_heliostat_cleaning_sector # [m²/sector]
        )  # [$/cleaning sector]

        # Calculate fuel cost per sector
        self._consumable_costs['fuel'] = (
            p.usage_fuel * # [L/hour]
            p.cost_fuel *  # [$/L]
            p.time_shift /  # [hours/shift]
            self.n_sectors_per_truck  # [sectors/shift]
        )  # [$/cleaning sector]

        # Calculate total cost
        self._consumable_costs['total'] = self._consumable_costs['water'] + self._consumable_costs['fuel']
        
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
        print(f"Updated sectors to {new_sectors[0]} x {new_sectors[1]} = {new_sectors[0] * new_sectors[1]} total sectors")

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
            print(f"Cleaning rate changed from {old_rate:.1f} to {new_rate:.1f} heliostats per shift")
            print(f"Updated costs per sector:")
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
        
    def calculate_cleaning_rate(self, solar_field, cleaning_rate:float=None, tolerance:float=0.05) -> tuple:
        """Calculate cleaning rate based on truck parameters or use provided rate.
        
        Args:
            solar_field (np.ndarray): Array containing heliostat positions
            cleaning_rate (float, optional): Manual override for cleaning rate
            tolerance (float, optional): Maximum allowed difference between calculated and discretized rates
            
        Returns:
            tuple: (cleaning_rate, (n_rad, n_az), n_sectors_per_truck)
        """
        # Calculate or use provided cleaning rate
        if cleaning_rate is None and self._sectors is None:
            hour_per_reload = self._hour_per_reload_consumables()
            spacing = self._heliostat_spacing(solar_field[:,1], solar_field[:,2])
            hour_per_clean = self._hour_per_heliostat_cleaning(heliostat_spacing=spacing,
                truck_velocity=self._params.velocity_travel,
                truck_cleaning_velocity=self._params.velocity_cleaning,
                cleaning_setup_seconds=self._params.time_setup
            )
            target_rate = self._heliostats_cleaned_shift(
                shift_hours=self._params.time_shift,
                hour_per_heliostat_clean=hour_per_clean,
                hour_reloading_per_heliostat=hour_per_reload
            )
            print(f'Calculated cleaning rate: {target_rate:.1f} heliostats/shift')
        elif cleaning_rate is not None:
            target_rate = cleaning_rate
            print(f'Using config specified cleaning rate: {target_rate:.1f} heliostats/shift')
        else:
            raise ValueError("Must provide either:\n1. Manual cleaning rate: cleaning_rate only,\n2. Auto cleaning rate calculation: no num_sectors and no cleaning_rate.\n3. Manual sector configuration: num_sectors")
        
        if target_rate > len(solar_field):
            target_rate = len(solar_field)
            cleaning_rate = len(solar_field)
            print(f'Warning: Target rate {target_rate} exceeds number of heliostats {len(solar_field)}. Setting to {len(solar_field)}')
        # Calculate sectors based on target rate
        self._optimize_sectors(solar_field, target_rate, tolerance)
        
        # Update consumable costs
        self._calculate_costs()

    def _optimize_sectors(self, solar_field, target_rate: float, tolerance: float) -> tuple:
        """Calculate optimal sector configuration for given cleaning rate."""
        n_sectors_per_truck = 1
        best_error = float('inf')
        best_sectors = None
        
        if target_rate >= len(solar_field): # Increase field resoltuion if we are cleaning full field in one truck
            best_n_sectors = int(np.ceil(len(solar_field) / 50 ))
            best_sectors = (int(np.floor(np.sqrt(best_n_sectors))), int(np.ceil(np.sqrt(best_n_sectors))))
            best_rate = len(solar_field) / (best_sectors[0] * best_sectors[1]/best_n_sectors)
            best_error = abs(best_rate - target_rate) / target_rate
            if best_rate < target_rate:
                raise ValueError("Target rate exceeds number of heliostats in field")
        else:
            while n_sectors_per_truck <= 10:
                n_sectors = len(solar_field) / target_rate
                n_sectors_scaled = n_sectors * n_sectors_per_truck
                
                n_rad = max(1, int(np.sqrt(n_sectors_scaled)))
                n_az = int(np.ceil(n_sectors_scaled/n_rad))
                
                while n_rad * n_az < n_sectors_scaled:
                    n_rad += 1
                    n_az = int(np.ceil(n_sectors_scaled/n_rad))
                
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
        
        print(f'Grid size: {best_sectors[0]} x {best_sectors[1]} = {best_sectors[0] * best_sectors[1]} sectors')
        print(f'Sectors per truck: {best_n_sectors}')
        print(f'Effective cleaning rate: {best_rate:.1f} heliostats/shift')
        print(f'Error from target: {best_error*100:.1f}%')
        
    def _heliostats_cleaned_shift(self, shift_hours=None, 
                               hour_per_heliostat_clean=None, 
                               hour_reloading_per_heliostat=None) -> float:
        """Calculate heliostats cleaned per shift."""
        
        return shift_hours / (hour_per_heliostat_clean + hour_reloading_per_heliostat)

    def _hour_per_heliostat_cleaning(self, heliostat_spacing:float, truck_velocity: float=10.0, truck_cleaning_velocity: float=2.0, cleaning_setup_seconds: float=30.0):
        """Calculates the time it takes to move to a heliostat and clean it."""
        return heliostat_spacing / (truck_velocity*1e3) + self._params.heliostat_width / (truck_cleaning_velocity*1e3) + (cleaning_setup_seconds/3600)# [hours] time it takes to move to a heliostat and clean it

    def _heliostat_spacing(self, positions_x: np.ndarray, positions_y: np.ndarray) -> float:
        """Calculate average spacing between heliostats."""
        n_heliostats = len(positions_x)
        min_distances = np.zeros(n_heliostats)
        for i in range(n_heliostats):
            distances = np.sqrt((positions_x - positions_x[i])**2 + (positions_y - positions_y[i])**2)
            distances[distances == 0] = np.inf
            min_distances[i] = np.min(distances)
        return np.mean(min_distances) - self._params.heliostat_width

    def _hour_per_reload_consumables(self) -> float:
        """Calculate time required for consumable reloading."""
        p = self._params
        
        heliostat_area = p.heliostat_width * p.heliostat_height # [m^2] area of heliostat
        cleaning_capacity_area = p.truck_water_volume / p.usage_water # [m^2] cleaning capacity of water
        reload_occurence_rate = heliostat_area / cleaning_capacity_area # []
        
        hour_travel_reload = 2 * p.distance_reload_station / (p.velocity_travel * 1e3) # [hours] travel time to reload station
        
        return reload_occurence_rate * (hour_travel_reload + p.time_shift)  # [hours]

class sun:
    def __init__(self):
        self.irradiation = {}   # [W/m^2] Extraterrestrial nominal solar irradiation
        self.elevation = {}     # [degrees]
        self.declination = {}   # [degrees]
        self.azimuth = {}       # [degrees]
        self.zenith = {}        # [degrees]
        self.hourly = {}        
        self.time = {}          # time vector for solar angles (datetime)
        self.DNI = {}           # [W/m^2] direct normal irradiance at ground
        self.stow_angle = {}    # [deg] minimum sun elevation angle where heliostat field operates
        
    def import_sun(self,file_params):
        table = pd.read_excel(file_params,index_col="Parameter")
        self.stow_angle = float(table.loc['stowangle'].Value)
        
class helios:
    def __init__(self):
        
        # Properties of heliostat (scalars, assumes identical heliostats)
        self.hamaker = []          # [J] hamaker constant of heliostat glass
        self.poisson = []          # [-] poisson ratio of heliostat glass
        self.youngs_modulus = []   # [Pa] young's modulus of glass
        self.nominal_reflectance = [] #[-] as clean reflectance of the heliostat
        self.height = []
        self.width = []
        self.num_radial_sectors = []
        self.num_theta_sectors = []
        
        # Properties of individual heliostats (1D array indexed by heliostat_index)
        self.x = []                         # [m] x (east-west) position of representative heliostats
        self.y = []                         # [m] y (north-south) position of representative heliostats
        self.rho = []                       # [m] radius for polar coordinates of representative heliostats
        self.theta = []                     # [deg] angle (from north) for polar coordinates of representative heliostats
        self.dist = []                      # planar distance to tower
        self.elevation_angle_to_tower = []  # elevation angle from heliostats to tower
        self.sector_area = []               # [m**2] sector area
        self.full_field = { 'rho':[],
                            'theta':[],
                            'x':[],
                            'y':[],
                            'z':[],
                            'sector_id':[]
                        }                   # populated if representative heliostats are from a sectorization of a field
        
        self.acceptance_angles = {}          # acceptance angle for receiver

        # Mie extinction weighting (dict of 2D arrays indexed by heliostat index, dust diameter)
        self.extinction_weighting = {}       
        
        # Movement properties (dicts of 2D arrays indexed by [heliostat_index, time] with weather file name keys )
        self.tilt = {}                      # [deg] tilt angle of the heliostat
        self.azimuth = {}                   # [deg] azimuth angle of the heliostat
        self.incidence_angle = {}           # [deg] incidence angle of solar rays
        self.elevation = {}                 # [deg] elevation angle of the heliostat
        self.inc_ref_factor = {}            # [ - ] incidence factor for reflectance computation (1st surface for now)
        self.stow_tilt = {}                 # [deg] tilt at which heliostats are stowed at night
        self.optical_efficiency = {}        # [ - ] average total optical efficiency of the sector represented by the heliostat
        
        # Properties of dust on heliostat (dicts of 3D arrays, indexed by [heliostat_index, time, diameter] with experiment numbers as keys)
        self.delta_soiled_area = {}         # [m^2/m^2] "pdf" of projected area of dust deposited on mirror for each time interval & each diameter
        self.mom_removal = {}
        self.mom_adhesion = {}
        self.soiling_factor = {}
        self.D = {}                         # [µm] diameter discretization
        self.velocity = {}                  # [m/s] velocity of falling dust for each diameter
        self.pdfqN = {}                     # dq[particles/(s*m^2)]/dLog_{10}(D[µm]) "pdf" of dust flux 1 m2 of mirror (constant for each time interval) at each diameter
        self.delta_soiled_area_variance = {}
        self.soiling_factor_prediction_variance = {}

    def import_helios(self,file_params,file_solar_field=None,cleaning_rate:float=None,verbose=True):
        
        table = pd.read_excel(file_params,index_col="Parameter")
        # self.h_tower = float(table.loc['h_tower'].Value)
        self.hamaker = float(table.loc['hamaker_glass'].Value)
        self.poisson = float(table.loc['poisson_glass'].Value)
        self.youngs_modulus = float(table.loc['youngs_modulus_glass'].Value)
        self.nominal_reflectance = float(table.loc['nominal_reflectance'].Value)
        self.height = float(table.loc['heliostat_height'].Value)
        self.width = float(table.loc['heliostat_width'].Value)
        self.stow_tilt = float(table.loc['stow_tilt'].Value)
        solar_field = self.read_solarfield(file_solar_field)
        
        self.truck = Truck(config_path=file_params)
        self.truck.calculate_cleaning_rate(solar_field=solar_field, cleaning_rate=cleaning_rate)
            
        if isinstance(self.truck.sectors,str) and self.truck.sectors.lower() == 'manual': # Manual importing of solar field respresentatives
            self.x = solar_field[:,1] # x cartesian coordinate of each heliostat (E>0)
            self.y = solar_field[:,2] # y cartesian coordinate of each heliostat (N>0)
            self.rho = np.sqrt(self.x**2+self.y**2) # angular polar coordinate of each heliostat (E=0, positive counterclockwise)
            self.theta = np.arctan2(self.y,self.x) # radial polar coordinate of each heliostat
            self.num_radial_sectors = None
            self.num_theta_sectors = None    
        elif isinstance(self.truck.sectors,tuple) and isinstance(self.truck.sectors[0],int) and table.loc['receiver_type'].Value == 'External cylindrical'\
            and isinstance(self.truck.sectors[1],int):                 # import and sectorize
            n_rho,n_theta = self.truck.sectors
            _print_if("Sectorizing with {0:d} angular and {1:d} radial sectors".format(n_theta,n_rho),verbose)
            self.num_radial_sectors,self.num_theta_sectors = self.truck.sectors
            self.sectorize_radial(solar_field,n_rho,n_theta)
        elif table.loc['receiver_type'].Value == 'Flat plate':
            n_hor,n_vert = self.truck.sectors
            _print_if("Sectorizing with {0:d} horizontal and {1:d} vertical sectors".format(n_hor,n_vert),verbose)
            self.num_radial_sectors,self.num_theta_sectors = self.truck.sectors
            self.sectorize_kmeans_clusters(solar_field, self.truck.sectors[0] * self.truck.sectors[1])
            # self.sectorize_corn_cleaningrows(solar_field,n_hor,n_vert)
        else:
            raise ValueError("num_sectors must be None or an a 2-tuple of intergers")  
    
    def sectorize_radial(self,solar_field,n_rho,n_theta,verbose=True):
        x = solar_field[:,1] # x cartesian coordinate of each heliostat (E>0)
        y = solar_field[:,2] # y cartesian coordinate of each heliostat (N>0)
        # n_sec = n_rho*n_theta
        n_tot = len(x)
        extra_hel_th = np.mod(n_tot,n_theta)
    
        rho = np.sqrt(x**2+y**2)                    # radius - polar coordinates of each heliostat
        theta = np.arctan2(y,x)                     # angle - polar coordinates of each heliostat
       
        val_t1 = np.sort(theta)                     # sorts the heliostats by ascendent thetas
        idx_t = np.argsort(theta)                   # store the indexes of the ascendent thetas
        val_r1 = rho[idx_t]                         # find the corresponding values of the radii
        
        val_r = np.concatenate((val_r1[val_t1>=-np.pi/2], val_r1[val_t1<-np.pi/2]))          # "rotates" to have -pi/2 as the first theta value
        val_t = np.concatenate((val_t1[val_t1>=-np.pi/2], val_t1[val_t1<-np.pi/2]+2*np.pi))  # "rotates" to have -pi/2 as the first theta value
        
        self.full_field['rho'] = val_r
        self.full_field['theta'] = val_t
        self.full_field['x'] = val_r*np.cos(val_t)
        self.full_field['y'] = val_r*np.sin(val_t)
        self.full_field['id'] = np.arange(n_tot,dtype=np.int64)
        self.full_field['sector_id'] = np.nan*np.ones(len(x))

        # compute the coordinates of the angular sector-delimiting heliostats
        n_th_hel = np.floor(n_tot/n_theta).astype('int')                 # rounded-down number of heliostats per angular sector
        if extra_hel_th==0:
            idx_th_sec = np.arange(0,len(val_r1),n_th_hel)
        else:
            # compute the angular-sector delimiting heliostats to have sectors with same (or as close as possible) number of heliostats
            id_at = np.array([0])
            id_bt = np.arange(1,extra_hel_th+1,1)
            id_ct = extra_hel_th*np.ones(n_theta-extra_hel_th-1).astype('int')
            id_dt = np.array([extra_hel_th-1])
            idx_th_sec = np.arange(0,len(val_r),n_th_hel)+np.concatenate((id_at,id_bt,id_ct,id_dt))

        theta_th_sec = val_t[idx_th_sec]
        rho_th_sec = val_r[idx_th_sec]

        rho_r_sec = np.zeros((n_rho,n_theta))
        theta_r_sec = np.zeros((n_rho,n_theta))
        hel_sec = []
        hel_rep = np.zeros((n_rho*n_theta,4))
        self.sector_area = np.zeros(n_rho*n_theta)
        kk = 0
        for ii in range(n_theta):
            if ii!=n_theta-1:
                in_theta_slice = (val_t>=theta_th_sec[ii]) & (val_t<theta_th_sec[ii+1])
                thetas = val_t[in_theta_slice] # selects the heliostats whose angular coordinate is within the ii-th angular sector
                rhos = val_r[in_theta_slice]   # selects the correspondent values of radius
            else:
                in_theta_slice = (val_t>=theta_th_sec[ii])
                thetas = val_t[in_theta_slice]                              # same as above for the last sector
                rhos = val_r[in_theta_slice]                             # same as above for the last sector
                
            AR = np.sort(rhos)                              # sort the heliostats belonging to each sector by radius
            AR_idx = np.argsort(rhos)                       # store the indexes
            AT = thetas[AR_idx]                             # find the corresponding thetas
            
            # compute the angular-sector delimiting heliostats to have sectors with same (or as close as possible) number of heliostats
            id_ar = np.array([0])
            id_br = (np.floor(len(AR)/n_rho)*np.ones(n_rho)).astype('int')
            id_cr = np.ones(np.mod(len(AR),n_rho)).astype('int')
            id_dr = np.zeros((n_rho-np.mod(len(AR),n_rho))).astype('int')
            idx_r_sec = np.cumsum(np.concatenate((id_ar,id_br+np.concatenate((id_cr,id_dr)))))
            AR_sec = AR[idx_r_sec[0:n_rho]]
            AT_sec = AT[idx_r_sec[0:n_rho]]
            rho_r_sec[:,ii] = AR_sec[0:len(rho_r_sec[:,ii])]        # finds the radial sector-delimiting heliostats for each angular sector
            theta_r_sec[:,ii] = AT_sec[0:len(rho_r_sec[:,ii])]      # finds the corresponding angles of the radial sector-delimiting heliostats for each angular sector
            
            # select the heliostats whose radial coordinate is within the jj-th radial sector of the ii-th angular sector
            for jj in range(n_rho):
                if jj!=n_rho-1:
                    and_in_radius_slice = (rhos>=rho_r_sec[jj,ii]) & (rhos<rho_r_sec[jj+1,ii])
                    rhos_jj = rhos[and_in_radius_slice] 
                    thetas_jj = thetas[and_in_radius_slice]
                else:
                    and_in_radius_slice = (rhos>=rho_r_sec[jj,ii])
                    rhos_jj = rhos[and_in_radius_slice]         # same as above for the last sector
                    thetas_jj = thetas[and_in_radius_slice]       # same as above for the last sector
                # hel_sec.append((rhos_jj,thetas_jj))   # store all the heliostats belonging to each sector
                rho_sec = np.mean(rhos_jj)         # compute the mean radius for the sector
                theta_sec = np.mean(thetas_jj)     # compute the mean angle for the sector

                idx = np.where(in_theta_slice)[0][and_in_radius_slice]
                self.full_field['sector_id'][idx] = kk
                self.sector_area[kk] = len(idx)*self.height*self.width # sector area
                
                # define the representative heliostats for each sector
                hel_rep[kk,0] = rho_sec
                hel_rep[kk,1] = theta_sec
                kk += 1

        hel_rep[:,2] = hel_rep[:,0]*np.cos(hel_rep[:,1])
        hel_rep[:,3] = hel_rep[:,0]*np.sin(hel_rep[:,1])

        self.x = hel_rep[:,2]
        self.y = hel_rep[:,3]
        self.theta = hel_rep[:,1]
        self.rho = hel_rep[:,0]
        self.heliostats_in_sector = hel_sec

    def sectorize_kmeans_clusters(self, solar_field, num_sectors):
        """Cluster heliostats based on distance and set representative heliostats."""
        print("Clustering heliostats...")
        weighted_positions = solar_field[:,1:]  # Get x,y coordinates
        kmeans = KMeans(n_clusters=num_sectors, random_state=42)
        labels = kmeans.fit_predict(weighted_positions)
        cluster_centers = kmeans.cluster_centers_
        
        # Initialize arrays to store information
        self.x = np.zeros(num_sectors)
        self.y = np.zeros(num_sectors)
        heliostats_in_sector = np.zeros(num_sectors, dtype=int)
        self.sector_area = np.zeros(num_sectors)
        
        # Store full field information
        self.full_field['x'] = solar_field[:,1]
        self.full_field['y'] = solar_field[:,2]
        self.full_field['id'] = solar_field[:,0]
        self.full_field['sector_id'] = labels
        
        # For each cluster, find closest heliostat to center and calculate sector information
        for i in range(num_sectors):
            # Find heliostats in this cluster
            mask = labels == i
            positions_in_cluster = weighted_positions[mask]
            
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
    
    def sectorize_corn_cleaningrows(self,solar_field,n_hor,n_vert,verbose=True):
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
        def generate_grid(num_hor,num_vert,x,y): # Generate a grid around solarfield
            x_points = np.linspace(min(x),max(x),num_hor)
            y_points = np.linspace(min(y),max(y),num_vert)
            grid = np.array([(x,y) for x in x_points for y in y_points])
            return grid
        
        def find_closest_point(position,grid): 
            distances = cdist([position[1:3]], grid) # Find distance between heliostats and grid coordinates
            closest_idx = np.argmin(distances) 
            return distances[0][closest_idx], closest_idx
            
        grid = generate_grid(n_hor,n_vert,solar_field[:,1],solar_field[:,2])
        
        closest_grid = [] # Create a dictionary to store 
        # [heliostat ID, x position, y position, distance to closest grid, closest grid point]
        for i in range(len(solar_field)):
            distance_grid, closest_idx = find_closest_point(solar_field[i,:],grid) 
            if i == 0:
                closest_grid = np.hstack([solar_field[i,:],distance_grid,closest_idx])
            else:
                closest_grid = np.vstack([closest_grid,np.hstack([solar_field[i,:],distance_grid,closest_idx])])
        
        
        # Store Heliostat Field information
        self.full_field['x'] = (closest_grid[:,1])
        self.full_field['y'] = (closest_grid[:,2])
        self.full_field['id'] = np.array(closest_grid[:,0],dtype=np.int64)
        self.full_field['sector_id'] = np.array(closest_grid[:,4],dtype=np.int64)
        
        for i in np.unique(self.full_field['sector_id']):
            sector_field = closest_grid[closest_grid[:,4] == i,:]
            sector_size = len(sector_field)
            representative_info = sector_field[np.argmin(sector_field[:,3])]
            if i == 0:
                representative_helio = np.hstack([representative_info,sector_size])
            else:
                representative_helio = np.vstack([representative_helio,np.hstack([representative_info,sector_size])])
                
        ##
        self.x = (representative_helio[:,1])
        self.y = (representative_helio[:,2])
        self.heliostats_in_sector = np.array(representative_helio[:,-1],dtype=np.int64)
        self.sector_area = self.heliostats_in_sector * self.height * self.width
        
    @staticmethod
    def read_solarfield(field_filepath): # Load CSV containing solarfield coordintes
        positions = []
        if field_filepath.split('.')[-1] == 'csv':
            whole_SF = pd.read_csv(field_filepath,skiprows=[1])
        elif field_filepath.split('.')[-1] == 'xlsx':
            whole_SF = pd.read_excel(field_filepath,skiprows=[1])
        else:
            raise ValueError("Solar field file must be csv or xlsx")
        
        x_field = np.array(whole_SF.loc[:,'Loc. X'])                    # x cartesian coordinate of each heliostat (E>0)
        y_field = np.array(whole_SF.loc[:,'Loc. Y'])
        helioID = np.arange(len(x_field),dtype=np.int64)
        positions = np.column_stack((helioID, x_field, y_field))
        return positions
    
    def sector_plot(self, show_id=False, cmap_name='turbo_r', figsize=(12, 10)):
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
        sid = self.full_field['sector_id']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a custom colormap where adjacent sectors have contrasting colors
        base_map = np.linspace(0.0, 1.0, len(np.unique(sid)))
        c_map = base_map
        for ii in range(1, len(np.unique(sid))):
            c_map = np.vstack((c_map, np.roll(base_map, 3*ii)))
        c_map = c_map.flatten()
        color_map = plt.cm.get_cmap(cmap_name)(c_map)
        
        # Plot each sector
        for ii in range(Ns):
            mask = sid == ii
            ax.scatter(
                self.full_field['x'][mask], 
                self.full_field['y'][mask], 
                color=color_map[ii % len(color_map)] if isinstance(color_map, np.ndarray) else color_map(ii / max(1, Ns-1)),
                alpha=0.7,
                s=30,
                label=f"Sector {ii}" if ii < 10 else None  # Limit legend entries
            )
            
            if show_id:
                # Add sector ID label with larger font
                center_x = np.mean(self.full_field['x'][mask])
                center_y = np.mean(self.full_field['y'][mask])
                ax.text(center_x, center_y, str(ii), 
                        alpha=1.0, ha='center', va='center', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Plot representative heliostats if not showing IDs
        if not show_id:
            ax.scatter(self.x, self.y, color='black', marker='X', s=100, 
                    label='Representative heliostats', zorder=10)
        
        # Add plot styling
        ax.set_xlabel('Distance from receiver - X [m]')
        ax.set_ylabel('Distance from receiver - Y [m]')
        ax.set_title('Solar Field Sectors')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set aspect ratio to equal to ensure correct spatial representation
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax


    def compute_extinction_weights(self,simulation_data,loss_model=None,lookup_tables=True,verbose=True,show_plots=False,options={}):
        """
        Computes the extinction weights for the heliostat field based on the specified loss model.
        
        Parameters:
            simulation_data (object): An object containing simulation data, including dust properties and source information.
            loss_model (str, optional): The loss model to use for computing the extinction weights. Can be either 'mie' or 'geometry'. Defaults to None.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
            options (dict, optional): Additional options to pass to the extinction function.
        
        Returns:
            None
        """
        sim_dat = simulation_data
        dust = sim_dat.dust
        files = list(sim_dat.file_name.keys())
        num_diameters = [len(dust.D[f]) for f in files]
        num_heliostats = [len(self.tilt[f]) for f in files]
        phia = self.acceptance_angles

        self.extinction_weighting = {f:np.zeros((num_heliostats[f],num_diameters[f])) for f in files}
        if loss_model == 'mie':
            assert ( (phia is not None) and all( [len(phia[f])==num_heliostats[f] for f in files] ) ),\
                 "When loss_model == ""mie"", please set helios.acceptance_angles as a list with a value for each heliostat"
            _print_if("Loss Model is ""mie"". Computing extinction coefficients ... ",verbose)

            if lookup_tables:
                print("Reading values from lookup tables...")
                for f in files:
                    dia = sim_dat.dust.D[f]
                    angles = phia[f]
                    df = pd.read_csv(f'extinction_weights_lookup_table_{f}.csv', index_col=0)
                    diameters = df.columns.astype(float)  # asse colonne
                    acc_angles = df.index.astype(float)    # asse righe
                    ew = df.values                 # matrice dei valori
                    
                    interpolator = RegularGridInterpolator((acc_angles, diameters), ew)

                    # Crea mesh di tutti i punti da interpolare (shape: len(angles) * len(dia), 2)
                    grid_angles, grid_dia = np.meshgrid(angles, dia, indexing='ij')  # shape (M,N)
                    points = np.stack([grid_angles.ravel(), grid_dia.ravel()], axis=1)

                    # Interpolazione in blocco
                    interpolated_values = interpolator(points).reshape(len(angles), len(dia))

                    # Salva direttamente nella struttura dati
                    self.extinction_weighting[f][:, :] = interpolated_values
            else:
                same_ext = _same_ext_coeff(self,sim_dat)
                computed = []
                for f in files:
                    dia = sim_dat.dust.D[f]
                    refractive_index = sim_dat.dust.m[f]
                    lam = sim_dat.source_wavelength[f]
                    intensities = sim_dat.source_normalized_intensity[f]
                    h=0
                    for h in tqdm(range(num_heliostats[f]), 
                                desc=f"File {f}", 
                                postfix=f"acceptance angle {phia[f][h]*1e3:.2f} mrad"):
                        already_computed = [e in computed for _,e in enumerate(same_ext[f][h])]
                        if any(already_computed):
                            idx = already_computed.index(True)
                            fe,he = same_ext[f][h][idx]                        
                            self.extinction_weighting[f][h,:] = self.extinction_weighting[fe][he,:]
                        else:
                            ext_weight = _extinction_function(dia,lam,intensities,phia[f][h],
                                                            refractive_index,verbose=verbose,
                                                            **options)
                            self.extinction_weighting[f][h,:] = ext_weight
                            computed.append((f,h))
                    
                    if show_plots:
                        fig,ax = plt.subplots()
                        ax.semilogx(sim_dat.dust.D[f],self.extinction_weighting[f][h,:])
                        ax.set_title(f'Heliostat {h}, acceptance angle {phia[f][h]*1e3:.2f} mrad')
                        plt.show()

            _print_if("... Done!",verbose)

        else: #self.loss_model == 'geometry'
            _print_if(f"Loss Model is ""geometry"". Setting extinction coefficients to unity for all heliostats in all files.",verbose)
            for f in files:
                num_diameters = len(dust.D[f])
                self.extinction_weighting[f] = np.ones((num_heliostats[f],num_diameters))

    def compute_extinction_weights_lookup_table(self,simulation_data,acceptance_angles_range=None,verbose=True,save=True,options={}):
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
        files = list(sim_dat.file_name.keys())
        num_diameters = [len(dust.D[f]) for f in files]
        phia = acceptance_angles_range

        extinction_weighting = {f:np.zeros((len(acceptance_angles_range[f]),num_diameters[f])) for f in files}
        assert (phia is not None),\
                "When computing the lookup table, please set acceptance_angles_range as a list of values"
        _print_if("Loss Model is ""mie"". Computing extinction coefficients ... ",verbose)

        computed = []
        for f in files:
            dia = sim_dat.dust.D[f]
            refractive_index = sim_dat.dust.m[f]
            lam = sim_dat.source_wavelength[f]
            intensities = sim_dat.source_normalized_intensity[f]
            h=0
            for h in tqdm(range(len(acceptance_angles_range[f])), 
                desc=f"File {f}", 
                postfix=f"acceptance angle {phia[f][h]*1e3:.2f} mrad"):
                
                ext_weight = _extinction_function(dia,lam,intensities,phia[f][h],
                                                    refractive_index,verbose=verbose,
                                                    **options)
                extinction_weighting[f][h,:] = ext_weight

            if save:
                df = pd.DataFrame(extinction_weighting[f], index=acceptance_angles_range[f], columns=dia)
                df.index.name = 'Acceptance angles'
                df.to_csv(f'extinction_weights_lookup_table_{f}.csv')
                
        _print_if("... Done!",verbose)


    def plot_extinction_weights(self,simulation_data,fig_kwargs={},plot_kwargs={}):
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
        for ii,f in enumerate(files):
            num_heliostats = Nhelios[f]
            D = simulation_data.dust.D[f]
            idx = ii*ncols+1
            for jj in range(num_heliostats):
                if jj > 0:
                    ax1 = fig.add_subplot(nrows,ncols,idx,sharex=ax[ii-1],sharey=ax[ii-1])
                else:
                    ax1 = fig.add_subplot(nrows,ncols,idx)

                ax.append(ax1)
                ax1.semilogx(D,self.extinction_weighting[f][jj,:],**plot_kwargs)
                ax1.set_xlabel(r"Diameter ($\mu$m)")
                ax1.set_ylabel(r"Extinction area multiplier (-)")
                ax1.set_title(f"File {f}, Mirror {jj}, Acceptance Angle {phia[f][jj]:.2e} rad")
                ax1.grid(True)
                idx += 1
        plt.tight_layout()
        
        return fig,ax

class constants:
    def __init__(self):
        self.air_rho = []
        self.air_mu = []
        self.air_nu = []
        self.air_lambda_p = []
        self.irradiation = []
        self.g = []
        self.A_slip = []
        self.k_Boltzman = []
        self.k_von_Karman = []
        self.N_iter = []
        self.tol = []
        self.Re_Limit =[]
        self.alpha_EIM = []
        self.beta_EIM = []
        self.eps0 = []
        self.D0 = []
        
    def import_constants(self,file_params,verbose=True):
        _print_if("\nImporting constants",verbose)
        table = pd.read_excel(file_params,index_col="Parameter")
        self.air_rho = float(table.loc['air_density'].Value)            # [kg/m3] air density at T=293K and p=1 atm
        self.air_mu = float(table.loc['air_dynamic_viscosity'].Value)   # [Pa*s] air dynamic viscosity at T=293K and p=1 atm
        self.air_nu = self.air_mu/self.air_rho                          # [m^2/s] air kinematic viscosity  at T=293K and p=1 atm
        self.air_lambda_p = float(table.loc['mean_free_path_air'].Value)# [m] mean free path in air at T=293K and p=1 atm
        self.irradiation = float(table.loc['I_solar'].Value)            # [W/m2] solar extraterrestrial constant
        self.g = 9.81                                                   # [m/s^2] gravitational constant
        self.A_slip = np.array(table.loc['A1_A2_A3'].Value.split(';')).astype('float')  # coefficients for slip correction factor
        self.k_Boltzman = float(table.loc['k_boltzman'].Value)          # [J/K] Boltzman constant 
        self.k_von_Karman = float(table.loc['k_von_karman'].Value)      # Von Karman constant
        self.N_iter = int(table.loc['N_iter'].Value)                    # max interations to compute the gravitational settling velocity
        self.tol = float(table.loc['tol'].Value)                        # tolerance to reach convergence in the gravitational settling velocity computation
        self.Re_Limit = np.array(table.loc['Re_Limit'].Value.split(';')).astype('float') # Reynolds limit values to choose among correlations for the drag coefficient
        self.alpha_EIM = float(table.loc['alpha_EIM'].Value)            # factor for impaction factor computation
        self.beta_EIM = float(table.loc['beta_EIM'].Value)              # factor for impaction factor computation
        self.eps0 = float(table.loc['eps0'].Value)                      # empirical factor for boundary layer resistance computation
        self.D0 = float(table.loc['D0'].Value)                          # [m] common value of separation distance (Ahmadi)

class reflectance_measurements:
    """
    Represents a class for managing reflectance measurement data.
    
    The `reflectance_measurements` class is used to import and manage reflectance data from multiple experiments. It can handle multiple files, each containing 
    average and standard deviation of reflectance measurements, as well as optional tilt information. The class provides methods to access the imported data and generate plots.
    
    Args:
        reflectance_files (str or list): Path(s) to the Excel file(s) containing the reflectance data.
        time_grids (list): List of time grids corresponding to each reflectance file.
        number_of_measurements (int or list, optional): Number of reflectance measurements for each file. If not provided, defaults to 1 for each file.
        reflectometer_incidence_angle (float or list, optional): Incidence angle of the reflectometer for each file. If not provided, defaults to 0 for each file.
        reflectometer_acceptance_angle (float or list, optional): Acceptance angle of the reflectometer for each file. If not provided, defaults to 0 for each file.
        import_tilts (bool, optional): Whether to import tilt information from the files. Defaults to False.
        column_names_to_import (list, optional): List of column names to import from the data sheets. If not provided, all columns will be imported.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
    """
    def __init__(self,reflectance_files,time_grids,number_of_measurements=None, 
                    reflectometer_incidence_angle=None,reflectometer_acceptance_angle=None,
                    import_tilts=False,column_names_to_import=None,verbose=True):
        
        reflectance_files = _ensure_list(reflectance_files)
        N_experiments = len(reflectance_files)
        if number_of_measurements == None:
            self.number_of_measurements = [1.0]*N_experiments
        else:
            self.number_of_measurements = _import_option_helper(reflectance_files,number_of_measurements)

        
        if reflectometer_incidence_angle == None:
            reflectometer_incidence_angle = [0]*N_experiments
        else:
            reflectometer_incidence_angle = _import_option_helper(reflectance_files,reflectometer_incidence_angle)
        
        if reflectometer_acceptance_angle == None:
            reflectometer_acceptance_angle = [0]*N_experiments
        else:
            reflectometer_acceptance_angle = _import_option_helper(reflectance_files,reflectometer_acceptance_angle)

        self.file_name = {}
        self.times = {}
        self.average = {}
        self.soiling_rate = {}
        self.delta_ref = {}
        self.sigma = {}
        self.sigma_of_the_mean = {}
        self.prediction_indices = {}
        self.prediction_times = {}
        self.rho0 = {}
        self.reflectometer_incidence_angle = {}
        self.reflectometer_acceptance_angle = {}
        self.mirror_names = {}

        if import_tilts:
            self.tilts = {}
            
        self.import_reflectance_data(reflectance_files,time_grids,reflectometer_incidence_angle,
                                     reflectometer_acceptance_angle,import_tilts=import_tilts,
                                     column_names_to_import=column_names_to_import)
        
    def import_reflectance_data(self,reflectance_files,time_grids,reflectometer_incidence_angle,
                                reflectometer_acceptance_angle, import_tilts=False,column_names_to_import=None):
        """
        Imports reflectance data from Excel files and stores the data in the object's attributes.

        Args:
            reflectance_files (str or list): Path(s) to the Excel file(s) containing the reflectance data.
            time_grids (list): List of time grids corresponding to each reflectance file.
            reflectometer_incidence_angle (float or list, optional): Incidence angle of the reflectometer for each file. If not provided, defaults to 0 for each file.
            reflectometer_acceptance_angle (float or list, optional): Acceptance angle of the reflectometer for each file. If not provided, defaults to 0 for each file.
            import_tilts (bool, optional): Whether to import tilt information from the files. Defaults to False.
            column_names_to_import (list, optional): List of column names to import from the data sheets. If not provided, all columns will be imported.
        """
        for ii in range(len(reflectance_files)):
            
            self.file_name[ii] = reflectance_files[ii]
            reflectance_data = {
                "Average": pd.read_excel(reflectance_files[ii], sheet_name="Reflectance_Average"),
                "Sigma": pd.read_excel(reflectance_files[ii], sheet_name="Reflectance_Sigma")
            }

            # Extract timestamps
            time_column = next((col for col in reflectance_data['Average'].columns 
                            if col.lower() in ['time', 'timestamp', 'tmsmp', 'date time']), None)
            if time_column is not None:
                self.times[ii] = reflectance_data['Average'][time_column].values
            else:
                raise ValueError(f"No 'Time' or 'Timestamp' column found in file {reflectance_files[ii]}")
            
            # Import data and ensure proper dimensions, Reflectance assumed to be in % based hence / 100
            if column_names_to_import is not None:
                # Extract selected columns
                avg_data = reflectance_data['Average'][column_names_to_import].values / 100.0
                sig_data = reflectance_data['Sigma'][column_names_to_import].values / 100.0
                self.mirror_names[ii] = column_names_to_import
            else:
                # Extract all columns except the first (time) column
                avg_data = reflectance_data['Average'].iloc[:, 1:].values / 100.0
                sig_data = reflectance_data['Sigma'].iloc[:, 1:].values / 100.0
                self.mirror_names[ii] = list(reflectance_data['Average'].keys())[1:]
            
            # Ensure 2D arrays for both single and multiple columns
            if avg_data.ndim == 1:
                self.average[ii] = avg_data.reshape(-1, 1)
                self.sigma[ii] = sig_data.reshape(-1, 1)
            else:
                self.average[ii] = avg_data
                self.sigma[ii] = sig_data
                
            # Calculate delta_ref with proper dimensions
            self.delta_ref[ii] = np.vstack((
                np.zeros((1, self.average[ii].shape[1])),
                -np.diff(self.average[ii], axis=0)
            ))
            
            # Set up prediction indices and times
            self.prediction_indices[ii] = []
            self.prediction_times[ii] = []
            for m in self.times[ii]:
                self.prediction_indices[ii].append(np.argmin(np.abs(m - time_grids[ii])))
            self.prediction_times[ii].append(time_grids[ii][self.prediction_indices[ii]])
            
            # Calculate initial reflectance (rho0), handling NaN values
            self.rho0[ii] = np.nanmax(self.average[ii], axis=0)
            
            # Set reflectometer parameters
            self.reflectometer_incidence_angle[ii] = reflectometer_incidence_angle[ii]
            self.reflectometer_acceptance_angle[ii] = reflectometer_acceptance_angle[ii]
            self.sigma_of_the_mean[ii] = self.sigma[ii] / np.sqrt(self.number_of_measurements[ii])

            # Import tilts if requested
            if import_tilts:
                tilt_data = pd.read_excel(reflectance_files[ii], sheet_name="Tilts")[self.mirror_names[ii]].values
                if tilt_data.ndim == 1:
                    self.tilts[ii] = tilt_data.reshape(1, -1)  # Single row becomes (1, n_times)
                else:
                    self.tilts[ii] = tilt_data.transpose()  # Shape becomes (n_heliostats, n_times)
                    
    def get_experiment_subset(self,idx):
        attributes = [a for a in dir(self) if not a.startswith("__")] # filters out python standard attributes
        self_out = copy.deepcopy(self)
        for a in attributes:
            attr = self_out.__getattribute__(a)
            if isinstance(attr,dict):
                for k in list(attr.keys()):
                    if k not in idx:
                        attr.pop(k)
        return self_out
   
    def plot(self):
        files = list(self.average.keys())
        N_mirrors = self.average[0].shape[1]
        N_experiments = len(files)
        fig,ax = plt.subplots(N_mirrors,N_experiments,sharex="col",sharey=True)
        fig.suptitle("Reflectance Data Plot", fontsize=16)
        miny = 1.0
        for ii in range(N_experiments):
            f = files[ii]
            for jj in range(N_mirrors):

                # axis handle
                if N_experiments == 1:
                    a = ax[jj] # experiment ii, mirror jj plot
                else:
                    a = ax[jj,ii]

                tilt = self.tilts[f][jj]
                if jj == 0:
                    tilt_str = r"Experiment "+str(ii+1)+ r", tilt = ${0:.0f}^{{\circ}}$"
                else:
                    tilt_str = r"tilt = ${0:.0f}^{{\circ}}$"
                
                if all(tilt==tilt[0]):
                    a.set_title(tilt_str.format(tilt[0]))
                else:
                    a.set_title(tilt_str.format(tilt.mean())+" (average)")

                a.grid('on')
                m = self.average[f][:,jj]
                s = self.sigma_of_the_mean[f][:,jj]
                miny = min((m-6*s).min(),miny)
                error_two_sigma = 1.96*s
                a.errorbar(self.times[f],m,yerr=error_two_sigma,label="Measurement mean",marker=".")
            
            a.set_ylabel(r"Reflectance at ${0:.1f}^{{\circ}}$".format(self.reflectometer_incidence_angle[ii]))
        a.set_ylim((miny,1))
        a.set_xlabel("Date")

class field_model(soiling_base):
    def __init__(self,file_params,file_SF,cleaning_rate=None):
        super().__init__(file_params)

        self.sun = sun()
        self.sun.import_sun(file_params)
        
        self.helios.import_helios(file_params,file_SF,cleaning_rate=cleaning_rate)
        if not(isinstance(self.helios.stow_tilt,float)) and not(isinstance(self.helios.stow_tilt,int)):
            self.helios.stow_tilt = None

    def sun_angles(self,simulation_inputs,verbose=True):
        sim_in = simulation_inputs
        sun = self.sun
        constants = self.constants
        timezone = pytz.FixedOffset(int(self.timezone_offset*60))
        
        _print_if("Calculating sun apparent movement and angles for "+str(sim_in.N_simulations)+" simulations",verbose)
        
        files = list(sim_in.time.keys())
        for f in list(files):
            time_utc = sim_in.time[f].dt.tz_localize(timezone) # Apply UTC to timeseries
            time_utc = time_utc.tolist() # Convert to list
            
            # Loop through all times and calculate azimuth and altitude/elevation
            solar_angles = np.array([solar.get_position(self.latitude,self.longitude,time.to_pydatetime()) for time in time_utc]) 
            sun.azimuth[f] = solar_angles[:,0] # solar_angles(:,[azimuth,elevation])
            sun.elevation[f] = solar_angles[:,1]
            sun.DNI[f] = np.array([radiation.get_radiation_direct(time.to_pydatetime(),elevation) for time, elevation in zip(time_utc,solar_angles[:,1])])
            
        self.sun = sun # update sun in the main model 
    
    def helios_angles(self,plant,verbose=True,second_surface=True):
        """
        Calculates the heliostat movement and angles for a given solar field and simulation inputs.
        
        Parameters:
            plant (object): The solar plant object containing information about the plant configuration.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
            second_surface (bool, optional): Whether to use the second surface model for the incidence reflection factor. Defaults to True.
        
        Returns:
            None
        """
        sun = self.sun  
        helios = self.helios

        files = list(sun.elevation.keys())
        N_sims = len(files)
        _print_if("Calculating heliostat movement and angles for "+str(N_sims)+" simulations",verbose)

        for f in files:
            stowangle = sun.stow_angle
            h_tower = plant.receiver['tower_height']
            helios.dist = np.sqrt(helios.x**2+helios.y**2)                                  # horizontal distance between mirror and tower
            helios.elevation_angle_to_tower = np.degrees(np.arctan((h_tower/helios.dist)))  # elevation angle from heliostats to tower
            
            T_m = np.array([-helios.y,-helios.x,np.ones((len(helios.x)))*h_tower])          # relative position of tower from mirror (left-handed ref.sys.)
            L_m = np.sqrt(np.sum(T_m**2,axis=0))                                            # distance mirror-tower [m]
            t_m = T_m/L_m                                                                   # unit vector in the direction of the tower (from mirror, left-handed ref.sys.)
            s_m = np.array(\
                        [np.cos(rad(sun.elevation[f]))*np.cos(rad(sun.azimuth[f])), \
                            np.cos(rad(sun.elevation[f]))*np.sin(rad(sun.azimuth[f])), \
                            np.sin(rad(sun.elevation[f]))])                                # unit vector of direction of the sun from mirror (left-handed)
            s_m = np.transpose(s_m)
            THETA_m = 0.5*np.arccos(s_m.dot(t_m))
            THETA_m = np.transpose(THETA_m)                                                 # incident angle (the angle a ray of sun makes with the normal to the surface of the mirrors) in radians
            helios.incidence_angle[f] = np.degrees(THETA_m)                                 # incident angle in degrees
            helios.incidence_angle[f][:,sun.elevation[f]<=stowangle] = np.nan               # heliostats are stored vertically at night facing north
            
            # apply the formula (Guo et al.) to obtain the components of the normal for each mirror
            A_norm = np.zeros((len(helios.x),max(s_m.shape),min(s_m.shape)))
            B_norm = np.sin(THETA_m)/np.sin(2*THETA_m)
            for ii in range(len(helios.x)):
                A_norm[ii,:,:] = s_m+t_m[:,ii]
            
            N = A_norm[:,:,0]*B_norm                                # north vector
            E = A_norm[:,:,1]*B_norm                                # east vector
            H = A_norm[:,:,2]*B_norm                                # height vector
            N[:,sun.elevation[f]<=stowangle] = 1                    # heliostats are stored at night facing north
            E[:,sun.elevation[f]<=stowangle] = 0                    # heliostats are stored at night facing north
            H[:,sun.elevation[f]<=stowangle] = 0                    # heliostats are stored at night facing north
            # Nd = np.degrees(np.arctan2(E,N))                      
            # Ed = np.degrees(np.arctan2(N,E))
            # Hd = np.degrees(np.arctan2(H,np.sqrt(E**2+N**2)))   
            
            helios.elevation[f] = np.degrees(np.arctan(H/(np.sqrt(N**2+E**2))))         # [deg] elevation angle of the heliostats
            helios.elevation[f][:,sun.elevation[f]<=stowangle] = 90 - helios.stow_tilt  # heliostats are stored at stow_tilt at night facing north
            helios.tilt[f] = 90-helios.elevation[f]                                     # [deg] tilt angle of the heliostats
            helios.azimuth[f] = np.degrees(np.arctan2(E,N))                             # [deg] azimuth angle of the heliostat

            if second_surface==False:
                helios.inc_ref_factor[f] = (1+np.sin(rad(helios.incidence_angle[f])))/np.cos(rad(helios.incidence_angle[f])) # first surface
                _print_if("First surface model",verbose)
            elif second_surface==True:
                helios.inc_ref_factor[f] = 2/np.cos(rad(helios.incidence_angle[f]))  # second surface model
                _print_if("Second surface model",verbose)
            else:
                _print_if("Choose either first or second surface model",verbose)
   
        self.helios = helios

    def reflectance_loss(self,simulation_inputs,cleans,verbose=True):
        
        sim_in = simulation_inputs
        N_sims = sim_in.N_simulations
        _print_if("Calculating reflectance losses with cleaning for "+str(N_sims)+" simulations",verbose)

        helios = self.helios
        n_helios = helios.x.shape[0]
        
        files = list(sim_in.time.keys())
        for fi in range(len(files)):
            f = files[fi]
            T_days = (sim_in.time[f].iloc[-1]-sim_in.time[f].iloc[0]).days # needs to be more than one day
            n_hours = int(helios.delta_soiled_area[f].shape[1] )
            
            # accumulate soiling between cleans
            temp_soil = np.zeros((n_helios,n_hours))
            temp_soil2 =  np.zeros((n_helios,n_hours))
            for hh in range(n_helios):
                sra = copy.deepcopy(helios.delta_soiled_area[f][hh,:])     # use copy.deepcopy otherwise when modifying sra to compute temp_soil2, also helios.delta_soiled_area is modified
                clean_idx = np.where(cleans[fi][hh,:])[0]
                clean_at_0 = True                                       # kept true if sector hh-th is cleaned on day 0
                if len(clean_idx)>0 and clean_idx[0]!=0:
                    clean_idx = np.insert(clean_idx,0,0)                # insert clean_idx = 0 to compute soiling since the beginning
                    clean_at_0 = False                                  # true only when sector hh-th is cleaned on day 0
                if len(clean_idx)==0 or clean_idx[-1]!=(sra.shape[0]):                      
                    clean_idx = np.append(clean_idx,sra.shape[0])       # append clean_idx = 8760 to compute soiling until the end
                
                clean_idx_n = np.arange(len(clean_idx))
                for cc in clean_idx_n[:-1]:  
                    temp_soil[hh,clean_idx[cc]:clean_idx[cc+1]] = \
                        np.cumsum(sra[clean_idx[cc]:clean_idx[cc+1]])  # Note: clean_idx = 8760 would be outside sra, but Python interprets it as "till the end"
                  
                # Run again with initial condition equal to final soiling to obtain an approximation of "steady-state" soiling
                # if clean_at_0:
                #     temp_soil2[hh,:] = temp_soil[hh,:]
                # else:
                sra[0] = temp_soil[hh,-1]
                for cc in clean_idx_n[:-1]:                  
                    temp_soil2[hh,clean_idx[cc]:clean_idx[cc+1]] = \
                        np.cumsum(sra[clean_idx[cc]:clean_idx[cc+1]])
                                
            helios.soiling_factor[f] = 1-temp_soil2*helios.inc_ref_factor[f]  # hourly soiling factor for each sector of the solar field
        
        self.helios = helios

    def optical_efficiency(self,plant,simulation_inputs,climate_file,verbose=True,n_az=10,n_el=10):
        helios = self.helios
        sim_in = simulation_inputs
        sun = self.sun

        # check that lat/lon and timezone correspond to the same location
        if climate_file.split('.')[-1] == "epw":
            with open(climate_file) as f:
                line = f.readline()
                line = line.split(',')
                lat = line[6]
                lon = line[7]
                tz = line[-2]
        else:
            raise ValueError("Climate file type must be .epw")

        # check that parameter file and climate file have same location and timezone
        if self.latitude != float(lat) or self.longitude != float(lon):
            raise ValueError("Location of field_model and climate file do not match")
        if self.timezone_offset != float(tz):
            raise ValueError("Timezone offset of field_model and climate file do not match")
        
        cp = CoPylot()
        r = cp.data_create()
        assert cp.data_set_string(
            r,
            "ambient.0.weather_file",
            climate_file,
        )
        
        # layout setup
        assert cp.data_set_number(r,"heliostat.0.height",helios.height)
        assert cp.data_set_number(r,"heliostat.0.width",helios.width)
        assert cp.data_set_number(r,"heliostat.0.soiling",1) # Sets cleanliness to 100%
        assert cp.data_set_number(r,"heliostat.0.reflectivity",1)
        try:
            assert cp.data_set_string(r,"receiver.0.rec_type",plant.receiver['receiver_type'])
        except:
            assert cp.data_set_string(r,"receiver.0.rec_type",'External cylindrical')
        if plant.receiver['receiver_type'] == 'External cylindrical':
            assert cp.data_set_number(r,"receiver.0.rec_diameter",plant.receiver['width_diameter'])
        elif plant.receiver['receiver_type'] == 'Flat plate':
            assert cp.data_set_number(r,"receiver.0.rec_width",plant.receiver['width_diameter'])
            try:    
                assert cp.data_set_number(r,"receiver.0.rec_elevation",plant.receiver['orientation_elevation'])
            except:
                assert cp.data_set_number(r,"receiver.0.rec_elevation",-35)
        assert cp.data_set_number(r,"receiver.0.rec_diameter",plant.receiver['width_diameter']) 
        assert cp.data_set_number(r,"receiver.0.rec_height",plant.receiver['panel_height'])
        assert cp.data_set_number(r,"receiver.0.optical_height",plant.receiver['tower_height'])

        # field setup
        ff = helios.full_field
        N_helios = ff['id'].shape[0]
        zz = [0 for ii in range(N_helios)]
        layout = [[ff['id'][ii],ff['x'][ii],ff['y'][ii],zz[ii]] for ii in range(N_helios)] #[list(id),list(ff['x']),list(ff['y']),list(zz)]
        assert cp.assign_layout(r,layout)
        field = cp.get_layout_info(r)
        
        # simulation parameters
        assert cp.data_set_number(r,"fluxsim.0.flux_time_type",0) # 1 for time simulation, 0 for solar angles
        assert cp.data_set_number(r,"fluxsim.0.flux_dni",1000.0)  # set the simulation DNI to 1000 W/m2. Only used to display indicative receiver power.
        assert cp.data_set_string(r,"fluxsim.0.aim_method",plant.aim_point_strategy)

        files = list(sim_in.time.keys())
        Ns = len(helios.x)
        helios.optical_efficiency = {f: [] for f in files}
        az_grid = np.linspace(0,360,num=n_az) 
        el_grid = np.linspace(sun.stow_angle,90,num=n_el)
        eff_grid = np.zeros((Ns,n_az,n_el))
        rec_power_grid = np.zeros((n_az,n_el))
        sec_ids = ff['sector_id'][field['id'].values.astype(int)] # CoPylot re-orders sectors
        fmt = "Getting efficiencies for az={0:.3f}, el={1:.3f}"
        
        # buliding the lookup table for grid of solar angles
        fmt_pwr = "Power absorbed by receiver at DNI=1000 W/m2: {0:.2e} kW"
        for ii in range(len(az_grid)):
            for jj in range(len(el_grid)):
                assert cp.data_set_number(r,"fluxsim.0.flux_solar_az_in",az_grid[ii])
                assert cp.data_set_number(r,"fluxsim.0.flux_solar_el_in",el_grid[jj])
                assert cp.simulate(r)
                dat = cp.detail_results(r)
                dat_summary = cp.summary_results(r)

                effs = dat['efficiency']
                _print_if(fmt.format(az_grid[ii],el_grid[jj]),verbose)
                if dat_summary['Power absorbed by the receiver'] == ' -nan(ind)':
                    raise ValueError("SolarPILOT unable to simulate with current parameter configuration")
                else:
                    _print_if(fmt_pwr.format(dat_summary['Power absorbed by the receiver']),verbose)
                    
                for kk in range(Ns):
                    idx = np.where(sec_ids==kk)[0]
                    eff_grid[kk,ii,jj] = effs[idx].mean()
        
        # Apply lookup table to simulation
        for f in files:
            T = len(sim_in.time[f])
            # Use above lookup to compute helios.optical_efficiency[f]
            _print_if("Computing optical efficiency time series for file "+str(f),verbose)
            helios.optical_efficiency[f] = np.zeros((Ns,T))
            for ll in range(Ns):
                opt_fun = interp2d(el_grid,az_grid,eff_grid[ll,:,:],fill_value=np.nan)
                helios.optical_efficiency[f][ll,:] = np.array( [opt_fun(sun.elevation[f][tt],sun.azimuth[f][tt])[0] \
                    for tt in range(T)] )
            _print_if("Done!",verbose)
        self.helios = helios

