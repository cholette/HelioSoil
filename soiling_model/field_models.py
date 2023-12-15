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
from soiling_model.utilities import _print_if,_ensure_list,\
                                    _extinction_function,_same_ext_coeff,\
                                    _import_option_helper,_parse_dust_str,\
                                    _check_keys
from textwrap import dedent
from pysolar import solar, radiation
import datetime
import pytz
from scipy.optimize import minimize_scalar
from scipy.integrate import cumtrapz
from scipy.spatial.distance import cdist
import copy
from copylot import CoPylot
from soiling_model.base_models import physical_base,constant_mean_base, sun

class central_tower_plant:
    def __init__(self):
        self.receiver = {   'receiver_type': [],
                            'tower_height': [],
                            'panel_height': [],
                            'width_diameter': [],
                            'orientation_elevation': [],
                            'thermal_losses': [],
                            'thermal_max': [],
                            'thermal_min': []  }
        self.power_block_efficiency = []
        self.aim_point_strategy = []
        
    def import_plant(self,file_params):
        table = pd.read_excel(file_params,index_col="Parameter")
        
        self.receiver['receiver_type'] = table.loc['receiver_type'].Value
        self.receiver['width_diameter'] = float(table.loc['receiver_width_diameter'].Value) # Width/Diameter for Flat/Circular receiver
        if self.receiver['receiver_type'] == 'Flat plate':
            try:
                self.receiver['orientation_elevation'] = float(table.loc['receiver_orientation_elevation'].Value)
            except:
                raise ValueError("receiver_orientation_elevation for Flat plate not specified in parameters file")
        self.receiver['tower_height'] = float(table.loc['receiver_tower_height'].Value)
        self.receiver['panel_height'] = float(table.loc['receiver_height'].Value)
        self.receiver['thermal_losses'] = float(table.loc['receiver_thermal_losses'].Value)
        self.receiver['thermal_min'] = float(table.loc['minimum_receiver_power'].Value)
        self.receiver['thermal_max'] = float(table.loc['maximum_receiver_power'].Value)
        self.power_block_efficiency = float(table.loc['power_block_efficiency'].Value)
        self.aim_point_strategy = table.loc['heliostat_aim_point_strategy'].Value

class field_common_methods:
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
                sra = copy.deepcopy(helios.delta_soiled_area[f][hh,:])  # use copy.deepcopy otherwise when modifying sra to compute temp_soil2, also helios.delta_soiled_area is modified
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

class field_model(physical_base,field_common_methods):
    def __init__(self,file_params,file_SF,num_sectors=None):
        super().__init__()
        super().import_site_data_and_constants(file_params)

        self.sun = sun()
        self.sun.import_sun(file_params)
        
        self.helios.import_helios(file_params,file_SF,num_sectors=num_sectors)
        if not(isinstance(self.helios.stow_tilt,float)) and not(isinstance(self.helios.stow_tilt,int)):
            self.helios.stow_tilt = None
    
    def compute_acceptance_angles(self,plant,verbose=True):
        # Approximate acceptance half-angles using simple geometric approximation from: 
        #    Sutter, Montecchi, von Dahlen, Fernández-García, and M. Röger, 
        #    “The effect of incidence angle on the reflectance of solar mirrors,” 
        #     Solar Energy Materials and Solar Cells, vol. 176, pp. 119–133, 
        #     Mar. 2018, doi: 10.1016/j.solmat.2017.11.029.
        # This approach neglects blocking, shading, and the use of a single acceptance angle assumes that 
        # the acceptance zone is conical. Tower height is presumed to be to the center. 

        tower_height = plant.receiver['tower_height']
        panel_height = plant.receiver['panel_height']
        files = list(self.helios.tilt.keys())
        n_helios,n_times = self.helios.tilt[0].shape
        self.helios.acceptance_angles = {f: np.zeros((n_helios,)) for f in files}
        for f in files:
            for ii,h in enumerate(zip(self.helios.x,self.helios.y)):
                d = np.sqrt(h[0]**2 + h[1]**2)
                h1 = tower_height                       # middle of panel
                h2 = tower_height + panel_height/2.0    # top of panel
                a1 = np.arctan(h1/d)
                a2 = np.arctan(h2/d)
                self.helios.acceptance_angles[f][ii] = (a2-a1) # [rad]

        max_accept = max([self.helios.acceptance_angles[f].max() for f in files])
        min_accept = min([self.helios.acceptance_angles[f].min() for f in files])
        _print_if(f"Acceptance angle range: ({min_accept*1e3:.1f}, {max_accept*1e3:.1f})",verbose)
            
class simplified_field_model(constant_mean_base,field_common_methods):
    def __init__(self,file_params,file_SF,num_sectors=None):
        super().__init__()
        super().import_site_data_and_constants(file_params)

        self.sun = sun()
        self.sun.import_sun(file_params)
        
        self.helios.import_helios(file_params,file_SF,num_sectors=num_sectors)
        if not(isinstance(self.helios.stow_tilt,float)) and not(isinstance(self.helios.stow_tilt,int)):
            self.helios.stow_tilt = None