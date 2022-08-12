from multiprocessing import dummy
import numpy as np
from numpy import matlib
from numpy import radians as rad
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import get_cmap, turbo
from warnings import warn
import copy
from scipy.interpolate import interp2d
from soiling_model.utilities import _print_if,_ensure_list,_check_keys
from scipy.optimize import minimize_scalar
import copy
from copylot import CoPylot

tol = np.finfo(float).eps # machine floating point precision

class base_model:
    def __init__(self,file_params):
        table = pd.read_excel(file_params,index_col="Parameter")
        self.latitude = float(table.loc['latitude'].Value)                  # latitude in degrees of site
        self.longitude = float(table.loc['longitude'].Value)                # longitude in degrees of site
        self.timezone_offset = float(table.loc['timezone_offset'].Value)    # [hrs from GMT] timezone of site
        self.hrz0 =float(table.loc['hr_z0'].Value)                          # [-] site roughness height ratio

        self.constants = constants()
        self.helios = helios()
        # self.dust = dust() # trying to put into simulation inputs now

        self.sigma_dep = None
        self.Ra = None

        self.constants.import_constants(file_params)
        # self.dust.import_dust(file_params,dust_measurement_type=dust_measurement_type)

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
                helios.pdfqN[f][idx,:,:] = Fd.transpose()*dust.pdfN[f]*1e6  # Dust flux pdf, i.e. [dq[particles/(s*m^2)]/dLog_{10}(D[µm]) ] deposited on 1m2. 1e6 for cm^3->m^3 
            
        self.helios = helios
        
    def adhesion_removal(self,simulation_inputs,verbose=True):
        _print_if("Calculating adhesion/removal balance",verbose)
        helios = self.helios
        dust = simulation_inputs.dust
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
                helios.pdfqN[f] = np.cumsum(helios.pdfqN[f],axis=1) # Accumulate in time so that we ensure we reomove all dust present on mirror if removal condition is satisfied at a particular time
                for h in range(Nhelios):
                    for k in range(Ntimes):
                        mom_removal = np.sin(rad(helios.tilt[f][h,k]))* F_gravity*np.sqrt((D_meters**2)/4-radius_sep**2) # [Nm] removal moment exerted by gravity at each tilt for each diameter
                        mom_adhesion =  (F_adhesion+F_gravity*np.cos(rad(helios.tilt[f][h,k])))*radius_sep             # [Nm] adhesion moment  
                        helios.pdfqN[f][h,k,mom_adhesion<mom_removal] = 0 # ALL dust desposited at this diameter up to this point falls off
                
                helios.pdfqN[f] = np.diff(helios.pdfqN[f],axis=1,prepend=0) # Take difference again so that pdfqN is the difference in dust deposited at each diameter

            else: # common stow angle at night for all heliostats. Assumes tilt at night is close to vertical at night.
                # Since the heliostats are stowed at a large tilt angle at night, we assume that any dust that falls off at this stow
                # is never deposited. This introduces a small error since the dust deposited during the day never affects the reflectance, but faster computation.
                _print_if("  Using common stow_tilt. Assumes all heliostats are stored at helios.stow_tilt at night.",verbose)
                mom_removal = np.sin(rad(helios.stow_tilt))* F_gravity*np.sqrt((D_meters**2)/4-radius_sep**2) # [Nm] removal moment exerted by gravity
                mom_adhesion =  (F_adhesion+F_gravity*np.cos(rad(helios.stow_tilt)))*radius_sep             # [Nm] adhesion moment
                helios.pdfqN[f][:,:,mom_adhesion<mom_removal] = 0 # Remove this diameter from consideration
        
        self.helios = helios
    
    def calculate_delta_soiled_area(self,simulation_inputs,sigma_dep=None,verbose=True): 
        _print_if("Calculating soil deposited in a timestep [m^2/m^2]",verbose)
        sim_in = simulation_inputs
        helios = self.helios
        dust = sim_in.dust
        
        files = list(sim_in.wind_speed.keys())
        
        for f in files:
            D_meters = dust.D[f]*1e-6
            helios.delta_soiled_area[f] = np.empty((helios.tilt[f].shape[0],helios.tilt[f].shape[1]))
            
            if sigma_dep is not None or self.sigma_dep is not None:
                helios.delta_soiled_area_variance[f] = np.empty((helios.tilt[f].shape[0],helios.tilt[f].shape[1]))

            # compute alpha
            try:
                den = getattr(dust,sim_in.dust_type[f]) # dust.(sim_in.dust_type[f])
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
                    helios.delta_soiled_area[f][ii,jj] = alpha[jj] * np.trapz(helios.pdfqN[f][ii,jj,:]*\
                        (np.pi/4*D_meters**2)*sim_in.dt[f],np.log10(dust.D[f]))

            # variance of noise for each measurement
            if sigma_dep != None:
                theta = np.radians(self.helios.tilt[f])
                helios.delta_soiled_area_variance[f] = sigma_dep**2* (alpha**2*np.cos(theta)**2) 
                # sigma_dep**2*helios.inc_ref_factor[f]*np.cumsum(alpha**2*np.cos(theta)**2,axis=1)
            elif self.sigma_dep != None:
                theta = np.radians(self.helios.tilt[f])
                helios.delta_soiled_area_variance[f] = self.sigma_dep**2 * (alpha**2*np.cos(theta)**2)

        self.helios = helios
    
    def plot_area_flux(self,dust_file,air_temp,wind_speed,tilt=0.0,hrz0=None,constants=None,ax=None,Ra=True):
        dummy_sim = simulation_inputs()
        dummy_sim.dust.import_dust(dust_file,verbose=False,dust_measurement_types="PM10")
        dummy_sim.air_temp = {0:np.array([air_temp])}
        dummy_sim.wind_speed = {0:np.array([wind_speed])}
        dummy_sim.dt = {0:1.0}
        dummy_sim.dust_type = {0:"PM10"}                    # this doesn't matter for this function
        dummy_sim.dust_concentration = {0:np.array([dummy_sim.dust.PM10[0]])}   # makes alpha = 1
        dummy_sim.N_simulations = 1

        dummy_model = copy.deepcopy(self)
        dummy_model.helios = helios()
        dummy_model.helios.tilt = {0:np.array([[tilt]])}
        dummy_model.sigma_dep = None
        
        fmt = "Setting constants.{0:s} to {1:s} (was {2:s})"
        if constants is not None:
            for kk in constants.keys():
                temp = str(getattr(dummy_model.constants,kk))
                print(fmt.format(str(kk), str(constants[kk]),temp))
                setattr(dummy_model.constants,kk,constants[kk])
        
        if hrz0 is None:
            _print_if("No hrz0 supplied. Using base_model.hrz0.",verbose=True)
            hrz0 = dummy_model.hrz0
        
        dummy_model.deposition_flux(dummy_sim,hrz0=hrz0,Ra=Ra)
        dummy_model.calculate_delta_soiled_area(dummy_sim)

        if ax is None:
            _,ax1 = plt.subplots()
        else:
            ax1 = ax

        title = 'Area loss rate for given dust distribution at wind_speed= {0:.1f} m/s, air_temperature={1:.1f} C \n (Area loss is {2:.2e} $m^2$/($s\cdot m^2$))'
        ax1.plot(dummy_sim.dust.D[0],dummy_model.helios.pdfqN[0][0,0,:]*np.pi/4*dummy_sim.dust.D[0]**2*1e-12)
        ax1.set_title(title.format(wind_speed,air_temp,dummy_model.helios.delta_soiled_area[0][0,0]))
        ax1.set_xlabel(r"D [$\mu$m]")
        ax1.set_ylabel(r'$\frac{dA [m^2/m^2/s] }{dLog(D \;[\mu m])}$', color='black',size=20)
        plt.xscale('log')   
        ax1.set_xticks([0.001,0.01,0.1,1,2.5,4,10,20,100])

class simulation_inputs:
    def __init__(self,experiment_files=None,k_factors=None,dust_types=None,verbose=True):

        # the below will be dictionaries of 1D arrays with file numbers as keys 
        self.file_name = {}             # name of the input file
        self.dt = {}                    # [seconds] simulation time step
        self.time = {}                  # absolute time (taken from 1st Jan)
        self.time_diff = {}             # [days] delta_time since start date
        self.start_datetime = {}        # datetime64 for start 
        self.end_datetime = {}          # datetime64 for end
        self.air_temp = {}              # [C] air temperature
        self.wind_speed = {}            # [m/s] wind speed
        self.dust_concentration = {}    # [µg/m3] PM10 or TSP concentration in air
        self.rain_intensity = {}        # [mm/hr] rain intensity
        self.dust_type = {}             # Usually either "TSP" or "PM10", but coule be any PMX or PMX.X
        self.dni = {}                   # [W/m^2] Direct Normal Irradiance
        self.relative_humidity = {}     # [%] relative humidity

        self.dust = dust()              # dust properties will be per experiment

        # if experiment files are supplied, import
        if experiment_files is not None:
            experiment_files = _ensure_list(experiment_files)
            self.N_simulations = len(experiment_files)

            if k_factors == None:
                k_factors = [1.0]*len(experiment_files)
            
            k_factors = _ensure_list(k_factors)
            if len(k_factors) != len(experiment_files):
                raise ValueError("Please specify a k-factor for each weather file")

            self.k_factors = {ii:k_factors[ii] for ii in range(self.N_simulations)} 
            self.import_weather(experiment_files,dust_types,verbose=verbose)

            dust_types = _ensure_list(dust_types)
            self.dust.import_dust(experiment_files,verbose=verbose,dust_measurement_types=dust_types)

    def import_weather(self,files,dust_types,verbose=True):
        
        dust_types = _ensure_list(dust_types)
        assert len(dust_types) == len(files),"Please supply a dust_type for each file."
        for ii in range(len(files)):

            self.file_name[ii] = files[ii]
            weather = pd.read_excel(files[ii],sheet_name="Weather")

            # Set time vector. Get from the weather file
            time = weather['Time'].to_numpy(dtype = 'datetime64[s]')
            self.start_datetime[ii] = time[0] # pd.to_datetime(weather['Time'].iloc[0]).to_numpy()
            self.end_datetime[ii] = time[-1] # pd.to_datetime(weather['Time'].iloc[-1]).to_numpy()
            # time =  pd.date_range(self.start_datetime[ii],self.end_datetime[ii],freq='1H').to_numpy(dtype = 'datetime64[s]')  # allow for flexible frequency later
                    
            _print_if("Importing site data (weather,time). Using dust_type = "+dust_types[ii]+", test_length = "+\
                str( ((self.end_datetime[ii]-self.start_datetime[ii]) + np.timedelta64(1,'h')).astype('timedelta64[h]') ),verbose)
            
            self.time[ii] = time

            if verbose:
                T = ( (time[-1]-time[0])+np.timedelta64(1,'h') ).astype('timedelta64[D]')
                _print_if("Length of simulation for file "+files[ii]+": "+str(T.astype(float))+" days",verbose)

            self.dt[ii] = np.diff(self.time[ii])[0].astype(float) # [s] assumed constant, in hours
            self.time_diff[ii] = (self.time[ii]-self.time[ii].astype('datetime64[D]')).astype('timedelta64[h]').astype('int')  # time difference from midnight in integer hours
            self.air_temp[ii] = np.array(weather.loc[:,'AirTemp'])
            self.wind_speed[ii] = np.array(weather.loc[:,'WindSpeed'])

            if 'DNI' in weather.columns: # only import DNI if it exists
                self.dni[ii] = np.array(weather.loc[:,'DNI']) 
            else:
                _print_if("No DNI data to import. Skipping.",verbose)
                
            # import dust measurements
            self.dust_concentration[ii] = self.k_factors[ii]*np.array(weather.loc[:,dust_types[ii]])
            self.dust_type[ii] = dust_types[ii]

            if "RainIntensity" in weather:
                _print_if("Importing rain intensity data...",verbose)
                self.rain_intensity[ii] = np.array(weather.loc[:,'RainIntensity'])
            else:
                _print_if("No rain intensity data to import.",verbose)

            if "RH" in weather:
                _print_if("Importing relative humidity data...",verbose)
                self.relative_humidity[ii] = np.array(weather.loc[:,'RH'])
            else:
                _print_if("No relative humidity data to import.",verbose)

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
        self.pdfN  = {}          # "pdf" of dust number d(N [1/cm3])/d(log10(D[µm]))
        self.pdfM  = {}          # "pdf" of dust mass dm[µg/m3]/dLog10(D[µm])
        self.pdfA  = {}          # "pdf" of dust mass dm[µg/m3]/dLog10(D[µm])
        self.hamaker = {}        # [J] hamaker constant of dust  
        self.poisson = {}        # [-] poisson ratio of dust
        self.youngs_modulus = {} # [Pa] young's modulus of dust
        self.PM10 = {}           # [µg/m^3] PM10 concentration computed with the given dust size distribution
        self.TSP = {}            # [µg/m^3] TSP concentration computed with the given dust size distribution
        self.Nd = {}
        self.log10_mu = {}
        self.log10_sig = {}
    
    def import_dust(self,experiment_files,verbose=True,dust_measurement_types=None):
        _print_if("Importing dust properties for each experiment",verbose)

        experiment_files = _ensure_list(experiment_files)
        dust_measurement_types = _ensure_list(dust_measurement_types)
        assert len(experiment_files) == len(dust_measurement_types), "Please supply a dust measruement type for each experiment. "
        
        for ii,f in enumerate(experiment_files):
            table = pd.read_excel(f,sheet_name="Dust",index_col="Parameter")
            rhoii = float(table.loc['rho'].Value)
            self.rho[ii] = rhoii

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

            pdfNii = np.sum(nNd,axis=1) # pdfN (number) distribution dN[cm^-3]/dLog10(D[µm])
            self.pdfN[ii] = pdfNii
            self.pdfA[ii] = pdfNii*(np.pi/4*Dii**2)*1e-6 # pdfA (area) dA[m^2/m^3]/dLog10(D[µm]), 1e-6 factor from { D^2(µm^2->m^2) 1e-12 , V(cm^3->m^3) 1e6 }
            self.pdfM[ii] = pdfNii*(rhoii*np.pi/6*Dii**3)*1e-3 # pdfm (mass) dm[µg/m^3]/dLog10(D[µm]), 1e-3 factor from { D^3(µm^3->m^3) 1e-18 , m(kg->µg) 1e9 , V(cm^3->m^3) 1e6 }
            self.TSP[ii] = np.trapz(self.pdfM[ii],np.log10(Dii)) 
            self.PM10[ii] = np.trapz(self.pdfM[ii][Dii<=10],np.log10(Dii[Dii<=10]))  # PM10 = np.trapz(self.pdfM[self.D<=10],dx=np.log10(self.D[self.D<=10]))

            self.hamaker[ii] = float(table.loc['hamaker_dust'].Value)
            self.poisson[ii] = float(table.loc['poisson_dust'].Value)
            self.youngs_modulus[ii] = float(table.loc['youngs_modulus_dust'].Value)

        # add dust measurements if they are PMX
        for dt in dust_measurement_types:
            if dt not in [None,"TSP"]: # another concentration is of interest (possibly because we have PMX measurements)
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
        _,ax1 = plt.subplots(nrows=N_files,sharex=True,squeeze=False,figsize=figsize)

        ax2 = []
        for ff in range(N_files):
            D_dust = self.D[ff]
            pdfN = self.pdfN[ff]
            pdfM = self.pdfM[ff]

            color = 'tab:red'
            ax1[ff,0].set_xlabel("D [$\mu$m]")
            ax1[ff,0].set_ylabel(r'$\frac{dN [cm^{{-3}} ] }{dLog(D \;[\mu m])}$', color=color,size=20)
            ax1[ff,0].plot(D_dust,pdfN, color=color)
            ax1[ff,0].tick_params(axis='y', labelcolor=color)
            ax1[ff,0].grid('on')

            ax2.append(ax1[ff,0].twinx())  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2[ff].set_ylabel(r'$\frac{dm \; [\mu g \, m^{{-3}} ] }{dLog(D \; [\mu m])}$', color=color,size=20)  # we already handled the x-label with ax1
            ax2[ff].plot(D_dust,pdfM, color=color)
            ax2[ff].tick_params(axis='y', labelcolor=color)
            ax2[ff].set_title("Number and Mass PDFs")
            ax2[ff].grid('on')
        
        plt.xscale('log')
        ax2[-1].set_xticks(10.0**np.arange(np.log10(D_dust[0]),np.log10(D_dust[-1]),1))

        return ax1,ax2

    def plot_area_distribution(self,figsize=(5,5)):
        N_files = len(self.D)
        _,ax1 = plt.subplots(nrows=N_files,sharex=True,squeeze=False,figsize=figsize)
        
        for ii in range(N_files):
            D_dust = self.D[ii]
            pdfA = self.pdfA[ii]

            color = 'black'
            ax1[ii,0].set_xlabel("D [$\mu$m]")
            ax1[ii,0].set_ylabel(r'$\frac{dA [m^2/m^3] }{dLog(D \;[\mu m])}$', color=color,size=20)
            ax1[ii,0].plot(D_dust,pdfA, color=color)
            ax1[ii,0].tick_params(axis='y', labelcolor=color)
            plt.xscale('log')
            ax1[ii,0].set_title("Area PDF")
            ax1[ii,0].set_xticks(10.0**np.arange(np.log10(D_dust[0]),np.log10(D_dust[-1]),1))

        return ax1

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
        
        # Geometry of field (1D array indexed by heliostat_index)
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
        self.num_radial_sectors = []
        self.num_theta_sectors = []

        # Movement properties (dicts of 2D array indexed by [heliostat_index, time] with weather file name keys )
        self.tilt = {}                      # [deg] tilt angle of the heliostat
        self.azimuth = {}                   # [deg] azimuth angle of the heliostat
        self.incidence_angle = {}           # [deg] incidence angle of solar rays
        self.elevation = {}                 # [deg] elevation angle of the heliostat
        self.inc_ref_factor = {}            # [ - ] incidence factor for reflectance computation (1st surface for now)
        self.stow_tilt = {}                 # [deg] tilt at which heliostats are stowed at night
        self.optical_efficiency = {}        # [ - ] average total optical efficiency of the sector represented by the heliostat
        
        # Properties of dust on heliostat (dicts of 3D arrays, indexed by [heliostat_index, time, diameter] with experiment numbers as keys)
        self.pdfqN = {}
        self.delta_soiled_area = {}         # [m^2/m^2] "pdf" of projected area of dust deposited on mirror for each time interval & each diameter
        self.mom_removal = {}
        self.mom_adhesion = {}
        self.soiling_factor = {}
        self.D = {}                         # [µm] diameter discretization
        self.velocity = {}                  # [m/s] velocity of falling dust for each diameter
        self.pdfqN = {}                     # dq[particles/(s*m^2)]/dLog_{10}(D[µm]) "pdf" of dust flux 1 m2 of mirror (constant for each time interval) at each diameter
        self.delta_soiled_area_variance = {}
        self.soiling_factor_prediction_variance = {}

    def import_helios(self,file_params,file_solar_field=None,num_sectors=None,verbose=True):
        
        table = pd.read_excel(file_params,index_col="Parameter")
        # self.h_tower = float(table.loc['h_tower'].Value)
        self.hamaker = float(table.loc['hamaker_glass'].Value)
        self.poisson = float(table.loc['poisson_glass'].Value)
        self.youngs_modulus = float(table.loc['youngs_modulus_glass'].Value)
        self.nominal_reflectance = float(table.loc['nominal_reflectance'].Value)
        self.height = float(table.loc['heliostat_height'].Value)
        self.width = float(table.loc['heliostat_width'].Value)
        self.stow_tilt = float(table.loc['stow_tilt'].Value)

        if file_solar_field==None:
            _print_if("Warning: solar field not defined. You will need to define it manually.",verbose)
        elif num_sectors == None:                                 # Import all heliostats
            _print_if("Importing representative heliostat directly. You will need to define the sector areas manually.",verbose)
            if file_solar_field.split('.')[-1] == 'csv':
                SF = pd.read_csv(file_solar_field,skiprows=[1])
            elif file_solar_field.split('.')[-1] == 'xlsx':
                SF = pd.read_excel(file_solar_field,skiprows=[1])
            else:
                raise ValueError("Solar field file must be csv or xlsx")

            self.x = np.array(SF.loc[:,'Loc. X'])               # x cartesian coordinate of each heliostat (E>0)
            self.y = np.array(SF.loc[:,'Loc. Y'])               # y cartesian coordinate of each heliostat (N>0)
            self.rho = np.sqrt(self.x**2+self.y**2)             # angular polar coordinate of each heliostat (E=0, positive counterclockwise)
            self.theta = np.arctan2(self.y,self.x)              # radial polar coordinate of each heliostat
            self.num_radial_sectors = None
            self.num_theta_sectors = None
        elif isinstance(num_sectors,tuple) and isinstance(num_sectors[0],int)\
            and isinstance(num_sectors[1],int):                 # import and sectorize
            n_rho,n_theta = num_sectors
            _print_if("Importing full solar field and sectorizing with {0:d} angular and {1:d} radial sectors".format(n_theta,n_rho),verbose)
            self.num_radial_sectors,self.num_theta_sectors = num_sectors
            self.sectorize(file_solar_field,n_rho,n_theta)
        else:
            raise ValueError("num_sectors must be None or an a 2-tuple of intergers")  

    def sectorize(self,whole_field_file,n_rho,n_theta,verbose=True):
        
        if whole_field_file.split('.')[-1] == 'csv':
            whole_SF = pd.read_csv(whole_field_file,skiprows=[1])
        elif whole_field_file.split('.')[-1] == 'xlsx':
            whole_SF = pd.read_excel(whole_field_file,skiprows=[1])
        else:
            raise ValueError("Solar field file must be csv or xlsx")

        x = np.array(whole_SF.loc[:,'Loc. X'])                         # x cartesian coordinate of each heliostat (E>0)
        y = np.array(whole_SF.loc[:,'Loc. Y'])                         # y cartesian coordinate of each heliostat (N>0)
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

    def sector_plot(self):
        Ns = self.x.shape[0]
        n_theta = self.num_theta_sectors
        n_radius = self.num_radial_sectors

        if n_theta == None:
            print("No sectorization defined")
        
        else:
            # set up colormap to make sure adjacent sectors have a different color
            base_map = np.linspace(0.0,1.0,n_radius)
            c_map = base_map
            for ii in range(1,n_theta):
                c_map = np.vstack( (c_map,np.roll(base_map,3*ii)) )
            c_map = c_map.flatten()
            c_map = turbo(c_map)

            sid = self.full_field['sector_id']
            fig,ax = plt.subplots()
            for ii in range(Ns):
                ax.scatter(self.full_field['x'][sid==ii],self.full_field['y'][sid==ii],color=c_map[ii])
            ax.scatter(self.x,self.y,color='black',marker='o',label='representative heliostats')
            plt.legend()
            plt.xlabel('distance from receiver - x [m]')
            plt.ylabel('distance from receiver -y [m]')
            plt.title('Solar Field Sectors')
            plt.show()

class plant:
    def __init__(self):
        self.receiver = {   'tower_height': [],
                            'panel_height': [],
                            'diameter': [],
                            'thermal_losses': [],
                            'thermal_max': [],
                            'thermal_min': []  }
        self.power_block_efficiency = []
        self.aim_point_strategy = []
        
    def import_plant(self,file_params):
        table = pd.read_excel(file_params,index_col="Parameter")
        self.receiver['tower_height'] = float(table.loc['receiver_tower_height'].Value)
        self.receiver['panel_height'] = float(table.loc['receiver_height'].Value)
        self.receiver['diameter'] = float(table.loc['receiver_diameter'].Value)
        self.receiver['thermal_losses'] = float(table.loc['receiver_thermal_losses'].Value)
        self.receiver['thermal_min'] = float(table.loc['minimum_receiver_power'].Value)
        self.receiver['thermal_max'] = float(table.loc['maximum_receiver_power'].Value)
        self.power_block_efficiency = float(table.loc['power_block_efficiency'].Value)
        self.aim_point_strategy = table.loc['heliostat_aim_point_strategy'].Value

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
        _print_if("Importing constants",verbose)
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
    def __init__(self,reflectance_files,time_grids,number_of_measurements=None, reflectometer_incidence_angle=None,\
        import_tilts=False,column_names_to_import=None):
        
        reflectance_files = _ensure_list(reflectance_files)
        reflectometer_incidence_angle = _ensure_list(reflectometer_incidence_angle)
        N_experiments = len(reflectance_files)
        if number_of_measurements == None:
            number_of_measurements = [1.0]*N_experiments
        
        if reflectometer_incidence_angle == None:
            reflectometer_incidence_angle = [0]*N_experiments

        self.file_name = {}
        self.times = {}
        self.average = {}
        self.sigma = {}
        self.sigma_of_the_mean = {}
        self.prediction_indices = {}
        self.prediction_times = {}
        self.rho0 = {}
        self.reflectometer_incidence_angle = {}
        if import_tilts:
            self.tilts = {}
        self.import_reflectance_data(reflectance_files,time_grids,number_of_measurements,reflectometer_incidence_angle,\
            import_tilts=import_tilts,column_names_to_import=column_names_to_import)
        
    def import_reflectance_data(self,reflectance_files,time_grids,number_of_measurements,reflectometer_incidence_angle,\
        import_tilts=False,column_names_to_import=None):
        for ii in range(len(reflectance_files)):
            
            self.file_name[ii] = reflectance_files[ii]
            reflectance_data = {"Average": pd.read_excel(reflectance_files[ii],sheet_name="Reflectance_Average"),\
                "Sigma": pd.read_excel(reflectance_files[ii],sheet_name="Reflectance_Sigma")}

            self.times[ii] = reflectance_data['Average']['Time'].values
            if column_names_to_import != None: # extract relevant column names of the pandas dataframe
                self.average[ii] = reflectance_data['Average'][column_names_to_import].values/100.0 # Note division by 100.0. Data in sheets are assumed to be in percentage
                self.sigma[ii] = reflectance_data['Sigma'][column_names_to_import].values/100.0 # Note division by 100.0. Data in sheets are assumed to be in percentage
            else:
                self.average[ii] = reflectance_data['Average'].iloc[:,1::].values/100.0 # Note division by 100.0. Data in sheets are assumed to be in percentage
                self.sigma[ii] = reflectance_data['Sigma'].iloc[:,1::].values/100.0 # Note division by 100.0. Data in sheets are assumed to be in percentage

            self.prediction_indices[ii] = []
            self.prediction_times[ii] = []
            for m in self.times[ii]:
                self.prediction_indices[ii].append(np.argmin(np.abs(m-time_grids[ii])))        
            self.prediction_times[ii].append(time_grids[ii][self.prediction_indices[ii]])
            self.rho0[ii] = self.average[ii][0,:]

            # idx = reflectance_files.index(f) 
            self.reflectometer_incidence_angle[ii] = reflectometer_incidence_angle[ii]
            self.sigma_of_the_mean[ii] = self.sigma[ii]/np.sqrt(number_of_measurements)

            if import_tilts:
                if column_names_to_import != None: # extract relevant column names of the pandas dataframe
                    self.tilts[ii] = pd.read_excel(reflectance_files[ii],sheet_name="Tilts")[column_names_to_import].values.transpose()
                else:
                    self.tilts[ii] = pd.read_excel(reflectance_files[ii],sheet_name="Tilts").iloc[:,1::].to_numpy().transpose()
                    
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

class field_model(base_model):
    def __init__(self,file_params,file_SF,num_sectors=None):
        super().__init__(file_params)

        self.sun = sun()
        self.sun.import_sun(file_params)
        
        self.helios.import_helios(file_params,file_SF,num_sectors=num_sectors)
        if not(isinstance(self.helios.stow_tilt,float)) and not(isinstance(self.helios.stow_tilt,int)):
            self.helios.stow_tilt = None

    def sun_angles(self,simulation_inputs,verbose=True):
        sim_in = simulation_inputs
        sun = self.sun
        constants = self.constants

        _print_if("Calculating sun apparent movement and angles for "+str(sim_in.N_simulations)+" simulations",verbose)
        
        files = list(sim_in.time.keys())
        for f in list(files):
            sun.hourly[f] = ((sim_in.time_diff[f] - 12)/24*360+self.longitude-15*self.timezone_offset)
            jan1 = np.datetime64(sim_in.start_datetime[f],'Y')   # calendar year given by start_date - required to compute the solar declination angle
            delta_time = (sim_in.start_datetime[f]-jan1).astype('timedelta64[s]')  # difference in time between the start date and the start of the calendar year
            time_delta_jan1 = (sim_in.time[f]-jan1+delta_time).astype('float')/3600/24     # transforms deltatime64[s] to number of days
            sun.declination[f] = 23.44*np.sin(rad(360/365*(time_delta_jan1+284)))   # Cooper equation (1969)
            sun.zenith[f] = np.degrees(np.arccos(np.sin(rad(self.latitude))*np.sin(rad(sun.declination[f]))+ \
                                np.cos(rad(self.latitude))*np.cos(rad(sun.declination[f]))* \
                                np.cos(rad(sun.hourly[f]))))
            sun.elevation[f] = 90 - sun.zenith[f]
            az = np.degrees(np.sign(sun.hourly[f])*abs(np.arccos((np.cos(rad(sun.zenith[f]))* \
                                np.sin(rad(self.latitude))-np.sin(rad(sun.declination[f])))/ \
                                (np.sin(rad(sun.zenith[f]))*np.cos(rad(self.latitude))))))
            sun.azimuth[f] = az+180  # 0->N 90->E 180->S 270->O
            I0 = constants.irradiation*(1+0.033*np.cos(2*np.pi*time_delta_jan1/365))
            AM = 1/np.cos(rad(sun.zenith[f]))
            AM[sun.elevation[f]<=sun.stow_angle] = float("NAN")
            sun.DNI[f] = I0*0.7**(AM**0.678)
            
        self.sun = sun # update sun in the main model 
    
    def helios_angles(self,plant,verbose=True,second_surface=True):
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
            T_days = (sim_in.time[f][-1]-sim_in.time[f][0]).astype('timedelta64[D]') # needs to be more than one day
            T_days = T_days.astype(float)
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

        # receiver setup
        assert cp.data_set_number(r,"receiver.0.rec_diameter",plant.receiver['diameter']) 
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

class fitting_experiment(base_model):
    def __init__(self,file_params):
        table = pd.read_excel(file_params,index_col="Parameter")
        super().__init__(file_params)
        self.helios.hamaker = float(table.loc['hamaker_glass'].Value)
        self.helios.poisson = float(table.loc['poisson_glass'].Value)
        self.helios.youngs_modulus = float(table.loc['youngs_modulus_glass'].Value)
        if not(isinstance(self.helios.stow_tilt,float)) and not(isinstance(self.helios.stow_tilt,int)):
            self.helios.stow_tilt = None

    def helios_angles(self,simulation_inputs,reflectance_data,verbose=True,second_surface=True):

        sim_in = simulation_inputs
        files = list(sim_in.time.keys())
        N_experiments = len(files)

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(sim_in,reflectance_data)

        _print_if("Setting tilts for "+str(N_experiments)+" experiments",verbose)
        helios = self.helios
        helios.tilt = {f: None for f in files} # clear the existing tilts
        for ii in range(N_experiments):
            f = files[ii]
            tilts = reflectance_data.tilts[f]
            N_times = len(sim_in.time[f])
            N_helios = tilts.shape[0]
            
            helios.tilt[f] = np.zeros((0,N_times))
            for jj in range(N_helios):
                row_mask = np.ones((1,N_times))
                helios.tilt[f] = np.vstack((helios.tilt[f],tilts[jj]*row_mask))

            helios.elevation[f] = 90-helios.tilt[f]
            helios.incidence_angle[f] = reflectance_data.reflectometer_incidence_angle[f]

            if second_surface==False:
                helios.inc_ref_factor[f] = (1+np.sin(rad(helios.incidence_angle[f])))/np.cos(rad(helios.incidence_angle[f])) # first surface
                _print_if("First surface model",verbose)
            elif second_surface==True:
                helios.inc_ref_factor[f] = 2/np.cos(rad(helios.incidence_angle[f]))  # second surface model
                _print_if("Second surface model",verbose)
            else:
                _print_if("Choose either first or second surface model",verbose)
    
        self.helios = helios

    def reflectance_loss(self):
        
        files = list(self.helios.tilt.keys())
        helios = self.helios
        helios.soiling_factor = {f: None for f in files} # clear the soiling factor
        for f in files:
            cumulative_soil =    np.cumsum(helios.delta_soiled_area[f],axis=1)   # accumulate soiling between cleans
            cumulative_soil = np.c_[np.zeros(cumulative_soil.shape[0]), cumulative_soil]
            helios.soiling_factor[f] = 1- cumulative_soil[:,1::]*helios.inc_ref_factor[f]  # soiling factor for each sector of the solar field
        
        self.helios = helios

    def predict_reflectance(self,simulation_inputs,hrz0=None,sigma_dep=None,verbose=True):

        self.deposition_flux(simulation_inputs,hrz0=hrz0,verbose=verbose)
        self.adhesion_removal(simulation_inputs,verbose=verbose)
        self.calculate_delta_soiled_area(simulation_inputs,sigma_dep=sigma_dep,verbose=verbose)
        self.reflectance_loss()

        # prediction variance
        if self.sigma_dep is not None:
            for f in self.helios.soiling_factor.keys():
                inc_factor = self.helios.inc_ref_factor[f]
                dsav = self.helios.delta_soiled_area_variance[f]
                self.helios.soiling_factor_prediction_variance[f] = np.cumsum( inc_factor**2 * dsav,axis=1 )
        else:
            self.helios.soiling_factor_prediction_variance = {}

    def _sse(self,hrz0,simulation_inputs,reflectance_data):
        pi = reflectance_data.prediction_indices
        meas = reflectance_data.average
        r0 = reflectance_data.rho0

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs,reflectance_data)

        sse = 0
        self.update_model_parameters(hrz0)
        self.predict_reflectance(simulation_inputs,verbose=False)
        sf = self.helios.soiling_factor
        files = list(sf.keys())
        for f in files:
            rho_prediction = r0[f]*sf[f][:,pi[f]].transpose()
            sse += np.sum( (rho_prediction -meas[f] )**2 )
        return sse  

    def fit_hrz0_least_squares(self,simulation_inputs,reflectance_data,verbose=True):

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs,reflectance_data)

        fun = lambda x: self._sse(x,simulation_inputs,reflectance_data)
        _print_if("Fitting hrz0 with least squares ...",verbose)
        xL = 1e-6 + 1.0
        xU = 1000 
        res = minimize_scalar(fun,bounds=(xL,xU),method="Bounded") # use bounded to prevent evaluation at values <=1
        _print_if("... done! \n hrz0 = "+str(res.x),verbose)
        return res.x, res.fun

    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
        # direction is either "forward" (to log-scaled space) or "inverse" (back to original scale)
        x = np.array(x)
        if direction == "inverse":
            z = np.array( [np.exp(np.exp(x[0])),np.exp(x[1])] )
        elif direction == "forward":
            z = np.array( [np.log(np.log(x[0])),np.log(x[1])] )
        else:
            raise ValueError("Transformation direction not recognized.")

        if not isinstance(likelihood_hessian,np.ndarray): # can't use likelihood_hessian == None because it is an array if supplied
            return z
        else:
            #Jacobian for transformation. See Reparameterization at https://en.wikipedia.org/wiki/Fisher_information 
            J = np.array([  [np.exp(np.exp(x[0])),0],\
                            [0, np.exp(x[1])]   ])

            if direction == "inverse":
                Ji = inv(J)                            
                z_cov = inv( Ji.transpose() @ likelihood_hessian @ Ji ) 
            elif direction == "forward":
                z_cov = inv( J.transpose() @ likelihood_hessian @ J ) 

            return z,z_cov 

    def plot_soiling_factor(self,simulation_inputs,posterior_predictive_distribution_samples=None,reflectance_data=None,figsize=None,\
        reflectance_std='measurements',save_path=None,fig_title=None,return_handles=False):
        sim_in = simulation_inputs
        samples = posterior_predictive_distribution_samples
        files = list(sim_in.time.keys())
        N_mirrors = np.array([self.helios.tilt[f].shape[0] for f in files])
        if np.all(N_mirrors==N_mirrors[0]):
            N_mirrors = N_mirrors[0]
        else:
            raise ValueError("Number of mirrors must be the same for each experiment to use this function.")

        if reflectance_data != None:
            # check to ensure that reflectance_data and simulation_input keys correspond to the same files
            _check_keys(sim_in,reflectance_data)
        
        N_experiments = sim_in.N_simulations
        ws_max = max([max(sim_in.wind_speed[f]) for f in files]) # max wind speed for setting y-axes
        mean_predictions        =  {f: np.array([]) for f in files} 
        CI_upper_predictions    =  {f: np.array([]) for f in files}
        CI_lower_predictions    =  {f: np.array([]) for f in files}
        
        fig,ax = plt.subplots(N_mirrors+1,N_experiments,figsize = figsize,sharex="col")
        fig.suptitle(fig_title, fontsize=16)
        ax_wind = []
        for ii in range(N_experiments):
            f = files[ii]
            N_times = self.helios.tilt[f].shape[1]
            mean_predictions[f]     =  np.zeros(shape=(N_mirrors,N_times)) 
            CI_upper_predictions[f] =  np.zeros(shape=(N_mirrors,N_times))
            CI_lower_predictions[f] =  np.zeros(shape=(N_mirrors,N_times))

            dust_conc = sim_in.dust_concentration[f]
            ws = sim_in.wind_speed[f]
            dust_type = sim_in.dust_type[f]
            ts = sim_in.time[f]
            if reflectance_data != None:
                tr = reflectance_data.times[f]

            for jj in range(0,N_mirrors):
                if jj == 0:
                    tilt_str = r"Experiment "+str(ii+1)+ r", tilt = ${0:.0f}^{{\circ}}$"
                else:
                    tilt_str = r"tilt = ${0:.0f}^{{\circ}}$"

                # get the axis handles
                if N_experiments == 1:
                    a = ax[jj] # experiment ii, mirror jj plot
                    a2 = ax[-1] # weather plot
                    am = ax[0] # plot to put legend on
                else:
                    a = ax[jj,ii]
                    a2 = ax[-1,ii]
                    am = ax[0,0]


                if reflectance_data is not None: # plot predictions and reflectance data
                    m = reflectance_data.average[f][:,jj]
                    m0 = m[0]

                    if reflectance_std == 'measurements':
                        s = reflectance_data.sigma[f][:,jj]
                    elif reflectance_std == 'mean':
                        s = reflectance_data.sigma_of_the_mean[f][:,jj]
                    else:
                        raise ValueError("reflectance_std="+reflectance_std+" not recognized. Must be either \"measurements\" or \"mean\" ")

                    # measurement plots
                    error_two_sigma = 1.96*s
                    a.errorbar(tr,m,yerr=error_two_sigma,label="Measurement mean")

                    # mean prediction plot
                    if samples == None: # use soiling factor in helios
                        ym = m0*self.helios.soiling_factor[f][jj,:]
                        a.plot(sim_in.time[f],ym,label='Reflectance Prediction',color='black')
                    else:
                        y = m0*samples[f][jj,:,:]
                        ym = y.mean(axis=1)
                        a.plot(sim_in.time[f],ym,label='Reflectance Prediction (Bayesian)',color='red')

                    tilt = reflectance_data.tilts[f][jj]
                    if all(tilt==tilt[0]):
                        a.set_title(tilt_str.format(tilt[0]))
                    else:
                        a.set_title(tilt_str.format(tilt.mean())+" (average)")
                else: # plot soiling factor predictions only
                    m0 = 1.0
                    if samples == None: 
                        ym = self.helios.soiling_factor[f][jj,:]  # no m0 is set to 1 since there are no measurements. Output is soiling factor only.  
                        a.plot(sim_in.time[f],ym,label='Soiling Factor Prediction',color='black')
                    else:
                        y = samples[f][jj,:,:]
                        ym = y.mean(y,axis=1)
                        a.plot(sim_in.time[f],ym,label='Soiling Factor Prediction (Bayesian)',color='red')
                    
                    tilt = self.helios.tilt[f][jj,:]
                    if all(tilt==tilt[0]):
                        a.set_title(tilt_str.format(tilt[0]))            
                    else:
                        a.set_title(tilt_str.format(tilt.mean())+" (average)")

                if samples==None and len(self.helios.soiling_factor_prediction_variance)>0: # add +/- 2 sigma limits to the predictions, is sigma_dep is set
                    # var_predict = self.helios.delta_soiled_area_variance[f][jj,:]
                    var_predict = self.helios.soiling_factor_prediction_variance[f][jj,:]
                    sigma_predict = np.sqrt( var_predict)
                    Lp = ym - m0*1.96*sigma_predict
                    Up = ym + m0*1.96*sigma_predict
                    a.fill_between(ts,Lp,Up,color='black',alpha=0.1,label=r'$\pm 2\sigma$ CI')
                elif samples != None: # use percentiles of posterior predictive samples for confidence intervals
                    Lp = np.percentile(y,2.5,axis=1)
                    Up = np.percentile(y,97.5,axis=1)
                    a.fill_between(ts,Lp,Up,color='red',alpha=0.1,label=r'95% Bayesian CI')
                
                a.xaxis.set_major_locator(mdates.DayLocator(interval=1)) # sets x ticks to day interval               
                
                if reflectance_data!=None: # reflectance is computed at reflectometer incidence angle
                    ang = reflectance_data.reflectometer_incidence_angle[f]
                    s = a.set_ylabel(r"$\rho(t)$ at "+str(ang)+"$^{{\circ}}$")
                else: # reflectance is computed at heliostat incidence angle. Put average incidence angle on axis label
                    ang = np.mean( self.helios.incidence_angle[f] )
                    s = a.set_ylabel(r"soiling factor at "+str(ang)+"$^{{\circ}}$ \n (average)")

                # set mean and CIs for output
                try:
                    mean_predictions[f][jj,:] = ym
                    CI_upper_predictions[f][jj,:] = Up
                    CI_lower_predictions[f][jj,:] = Lp 
                except:
                    mean_predictions[f][jj,:] = ym
            
            am.legend()
            label_str = dust_type + r" (mean = {0:.2f} $\mu g$/$m^3$)" 
            a2.plot(ts,dust_conc,label=label_str.format(dust_conc.mean()), color='blue')
            a2.xaxis.set_major_locator(mdates.DayLocator(interval=1)) # sets x ticks to day interval
            myFmt = mdates.DateFormatter('%d-%m-%Y')
            a2.xaxis.set_major_formatter(myFmt)
            a2.tick_params(axis ='y', labelcolor = 'blue')

            a2a = a2.twinx()
            p = a2a.plot(ts,ws,color='green',label="Wind Speed ({0:.2f} m/s)".format(ws.mean()))
            ax_wind.append(a2a)
            a2a.tick_params(axis ='y', labelcolor = 'green')
            a2a.set_ylim((0,ws_max))
            
            if ii == 0: # ylabel for TSP on leftmost plot only
                fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
                a2.set_ylabel(fs.format(dust_type),color='blue')
            if ii == N_experiments-1: # ylabel for wind speed on rightmost plot only
                a2a.set_ylabel('Wind Speed (m/s)', color='green') 
            
            a2.set_title(label_str.format(dust_conc.mean())+", Wind Speed ({0:.2f} m/s)".format(ws.mean()),fontsize=10)
        
        if N_experiments > 1:

            # share y axes for all reflectance measurments
            ymax = max([x.get_ylim()[1] for x in ax[0:-1,:].flatten()])
            ymin = min([x.get_ylim()[0] for x in ax[0:-1,:].flatten()])
            for a in ax[0:-1,:].flatten(): 
                a.set_ylim(ymin,1)

            # share y axes for weather variables of the same type
            ymax_dust = max([x.get_ylim()[1] for x in ax[-1,:]])
            ymax_wind = max([x.get_ylim()[1] for x in ax_wind])
            for a in ax[-1,:]:
                a.set_ylim(0,1.1*ymax_dust)
            for a in ax_wind:
                a.set_ylim(0,1.1*ymax_wind)
        else:
            ymax = max([x.get_ylim()[1] for x in ax[0:-1].flatten()])
            ymin = min([x.get_ylim()[0] for x in ax[0:-1].flatten()])
            for a in ax[0:-1]:
                a.set_ylim(ymin,ymax)

        fig.autofmt_xdate()
        if save_path != None:
            fig.savefig(save_path)

        fig.tight_layout()
        if return_handles:
            return fig,ax,mean_predictions,CI_lower_predictions,CI_upper_predictions
        else:
            return mean_predictions,CI_lower_predictions,CI_upper_predictions

    def update_model_parameters(self,x):
        if isinstance(x,list) or isinstance(x,np.ndarray) :
            self.hrz0 = x[0]
            if len(x)>1:
                self.sigma_dep = x[1]
        else:
            self.hrz0 = x
            self.sigma_dep = None

    def plot_traces(self,idata,estimates=None,priors=None):
        # plot traces and vertical lines for each estimate
        ax_trace = az.plot_trace(idata)
        L = ax_trace.shape[0]

        if not np.all(priors == None):
            names = list(priors.keys())
            for ii in range(L):
                a = ax_trace[ii,0]
                n = names[ii]
                XL = a.get_xlim(),1e-16
                x = np.linspace(XL[0],XL[1],1000)
                a.plot(x,priors[n].pdf(x),color='blue',label=n) 
        
        if not np.all(estimates == None):
            E = len(estimates)
            cmap = get_cmap('viridis')
            for ii in range(E):
                for jj in range(L):
                    a = ax_trace[jj,0]
                    if len(estimates[ii]) > jj:
                        a.axvline(estimates[ii][jj],color=cmap(ii/E),linestyle=':',label=n)

        plt.show(block=False)
