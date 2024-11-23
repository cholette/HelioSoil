import numpy as np
import pandas as pd
import copy
tol = np.finfo(float).eps # machine floating point precision
from soiling_model.base_models import simulation_inputs
from soiling_model.field_models import field_model,simplified_field_model,central_tower_plant
from soiling_model.utilities import _print_if,simple_annual_cleaning_schedule

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

class optimization_problem():
    def __init__(   self,params,solar_field,weather_files,climate_file,num_sectors,\
                    dust_type=None,n_az=10,n_el=10,second_surface=True,verbose=True,
                    model_type='semi-physical',ext_options={'grid_size_x':100}):
        self.truck = {  'operator_salary':[],
                        'operators_per_truck_per_day':[],
                        'purchase_cost':[],
                        'maintenance_costs':[],
                        'useful_life': [],
                        'fuel_cost': [],
                        'water_cost': []
                    }
        self.electricity_price = []
        self.plant_other_maintenace = [] 

        pl = central_tower_plant()
        pl.import_plant(params)
        sd = simulation_inputs(weather_files,dust_type=dust_type)
        if model_type.lower() == 'semi-physical':
            fm = field_model(params,solar_field,num_sectors=num_sectors)
            fm.sun_angles(sd)
            fm.helios_angles(pl,second_surface=second_surface)
            fm.compute_acceptance_angles(pl)    
            fm.helios.compute_extinction_weights(sd,fm.loss_model,verbose=True,options=ext_options)
            fm.deposition_flux(sd)
            fm.adhesion_removal(sd)
            fm.calculate_delta_soiled_area(sd)
        elif model_type.lower() == 'simplified':
            fm = simplified_field_model(params,solar_field,num_sectors=num_sectors)
            fm.sun_angles(sd)
            fm.helios_angles(pl,second_surface=second_surface)
            fm.calculate_delta_soiled_area(sd)
        else:
            raise ValueError("Model type not recognized. Must be either semi-physical or simplified.")

        fm.optical_efficiency(pl,sd,climate_file,n_az=n_az,n_el=n_el,verbose=verbose)

        self.field_model = fm
        self.simulation_data = sd
        self.plant = pl

def periodic_schedule_tcc(opt,n_trucks,n_cleans,\
    n_sectors_per_truck=1,verbose=True):
    """
    Computes the total cleaning costs (TCC) for a solar field optimization problem, including both degradation costs and direct cleaning costs.

    Args:
        opt (optimization_problem): An instance of the `soiling_model.cleaning_optimization.optimization_problem` class, containing the field model, plant, and simulation data.
        n_trucks (int): The number of cleaning trucks to use.
        n_cleans (int): The number of cleanings to perform per year.
        n_sectors_per_truck (int, optional): The number of sectors each truck can clean per cleaning. Defaults to 1.
        verbose (bool, optional): Whether to print detailed output. Defaults to True.

    Returns:
        dict: A dictionary containing the following keys:
            - 'total_cleaning_costs': The total cleaning costs (TCC)
            - 'degradation_costs': The degradation costs (C_deg)
            - 'direct_cleaning_costs': The direct cleaning costs (C_cl)
            - 'soiling_factor': The soiling factor for each heliostat
            - 'cleaning_actions': The cleaning schedule for each simulation file
            - 'soiling_induced_off_times': The number of time steps where the receiver is off due to soiling
            - 'soiling_induced_drops_below_upper_limit': The number of time steps where the clean receiver would be saturated but the actual receiver is not
    """

    assert isinstance(opt,optimization_problem),"First input must be a soiling_model.cleaning_optimization.optimization_problem instance. "
    field = opt.field_model
    plant = opt.plant
    files = list(opt.simulation_data.time.keys())
    N_files = len(files)
    
    # cleaning schedule, currently the same for all experiments/runs
    cleans = {k: [] for k in files}
    for f in files:
        n_helios = field.helios.tilt[f].shape[0]
        cleans[f] = simple_annual_cleaning_schedule(n_helios,n_trucks,n_cleans,\
            n_sectors_per_truck=n_sectors_per_truck)
    
    # compute reflectance losses (updates field.helios.soiling_factor)
    field.reflectance_loss(opt.simulation_data,cleans,verbose=verbose) 

    C_deg = np.zeros(N_files)
    C_cl = np.zeros(N_files)   
    TCC = np.zeros(N_files)     
    Aj = field.helios.sector_area
    Aj = Aj.reshape((len(Aj),1))
    eta_pb = plant.power_block_efficiency
    P = opt.electricity_price
    COM = opt.plant_other_maintenace
    Qloss = plant.receiver['thermal_losses']
    Qmin = plant.receiver['thermal_min']
    Qmax = plant.receiver['thermal_max']
    fmt_str = "Results for simulation {0:d}: \n  TCC: {1:.2e}\n  C_deg: {2:.2e}\n  C_cl: {3:.2e}"
    
    for fi in range(N_files):
        f = files[fi]
        sf = field.helios.soiling_factor[f].copy()
        opt_eff = field.helios.optical_efficiency[f].copy()

        # nans are when the sun is below the stowangle. Set the optical efficiency to zero during these times and
        # set sf to an arbitrary value just to ensure that we get zero instead of nan when summing.
        sf[np.isnan(sf)] = 0.01 
        opt_eff[np.isnan(opt_eff)] = 0
        
        # ensure that soiling factor is positive
        if np.any(sf<=0):
            ind = np.where(sf<=0)[0]
            _print_if("Warning: soiling factor is <= zero for {0:d} heliostats.".format(len(ind))+\
                "\n Setting these soiling factors equal to zero.",verbose)
            sf[ind] = 0
        
        # costs and efficiencies
        DNI = opt.simulation_data.dni[f]
        eta_nom = field.helios.nominal_reflectance
        eta_clean = eta_nom*opt_eff
        DT = opt.simulation_data.dt[f]
        alpha = eta_pb*(P-COM)*DT/3600.0
        
        clean_sector_reflected_power = DNI*Aj*eta_clean
        
        lost_power=_simulate_receiver_power(sf,clean_sector_reflected_power,Qmax*1e6,Qmin*1e6,Qloss*1e6)

        stow_elevation = field.sun.stow_angle
        sun_above_stow_elevation = (field.sun.elevation[f]>=stow_elevation)

        TCC[fi], C_cl[fi], C_deg[fi] = _cleaning_cost(lost_power,alpha,opt,cleans[f],n_trucks)
        
        # compute direct costs
        _print_if(fmt_str.format(fi,C_deg[fi]+C_cl[fi],C_deg[fi],C_cl[fi]),verbose)
    
    # [!] Day TCC
    night_idx = np.where(opt.simulation_data.time[f].dt.hour == 23)[0]
    tcc_days = np.zeros_like(night_idx)
    ccl_days = np.zeros_like(night_idx)
    cdeg_days = np.zeros_like(night_idx)
    for idx in  np.arange(len(night_idx)):
        tcc_days[idx], ccl_days[idx], cdeg_days[idx] = _cleaning_cost(lost_power[:night_idx[idx]], alpha,opt,cleans[0][:,:night_idx[idx]],n_trucks)
    
    results = { 'total_cleaning_costs':TCC,
                'degradation_costs': C_deg,
                'direct_cleaning_costs': C_cl,
                'soiling_factor':field.helios.soiling_factor.copy(),
                'arealoss':field.helios.arealoss,
                'cleaning_actions':cleans,
                'day_tcc':tcc_days,
                'day_cdeg':cdeg_days,
                'day_ccl':ccl_days} #'day_tcc', day_tcc
    return results

def rollout_heuristic_tcc(opt,n_trucks,n_sectors_per_truck=1,initial_arealoss=None):
    def _get_arealoss(soil_rate, cleaning_schedule=None):
        """
        Calculates the soiling factor based on the area loss, cleaning schedule, and incidence factor.
        
        Args:
            soil_rate (numpy.ndarray): The area loss over time.
            cleaning_schedule (numpy.ndarray): The cleaning schedule, where 1 indicates a cleaning event.
            incidence_factor (float): The incidence factor to apply to the cumulative loss.
        
        Returns:
            numpy.ndarray: The soiling factor over time.
        """
        if cleaning_schedule is None:
            cleaning_schedule = np.zeros_like(soil_rate)
        if len(soil_rate.shape) == 1:
            soil_rate = soil_rate.reshape((1, -1))
            cleaning_schedule = cleaning_schedule.reshape((1, -1))
        elif len(soil_rate.shape) == 2:
            pass
        else:
            raise ValueError("soil_rate must be a 1D or 2D array")
        
        area_loss = np.zeros_like(soil_rate)
        for i in range(soil_rate.shape[0]):
            cumulative = 0
            for j in range(soil_rate.shape[1]):
                if cleaning_schedule[i, j] == 1:
                    cumulative = 0
                else:
                    cumulative += soil_rate[i, j]
                area_loss[i, j] = cumulative
        
        return area_loss
    
    def _clean_arealoss(area_loss, cleaning_schedulue, cleaning_tech_curve=1.0):
        area_loss[cleaning_schedulue==1] = area_loss[cleaning_schedulue==1] * (1.0 - cleaning_tech_curve)
        return area_loss
    
    def _get_soilingfactor(area_loss,incidence_factor):
        soiling_factor = 1 - (area_loss * incidence_factor)
        soiling_factor[soiling_factor < 0] = 0
        soiling_factor[np.isnan(soiling_factor)] = 0
        return soiling_factor
                
    def _assign_indexing(current_day, day_of_year, n_day_horizon):
        idx = {'current': None, 'horizon': None, 'cleaning': None}
        idx['current'] = np.nonzero(day_of_year==current_day)[0]
        idx['horizon'] = np.nonzero((day_of_year >= np.mod(current_day, np.max(day_of_year) +1) +1) 
                                    & (day_of_year <= np.mod(current_day + n_day_horizon, np.max(day_of_year) +1) +1))[0]
        if len(idx['horizon']) == 0:
            idx['horizon'] = np.nonzero((day_of_year >= np.mod(current_day , np.max(day_of_year)+1)+1) 
                                        | (day_of_year <= np.mod(current_day + n_day_horizon, np.max(day_of_year) +1) +1))[0]
        idx['cleaning'] = idx['current'][-1]
        return idx
    
    def _sector_revenue(soilrate, cleaning_schedulue, incidence_factor, clean_reflected_irradiance, production_profit, sector_cleaningcost):
        # Area Loss
        clean_arealoss = _get_arealoss(soilrate, cleaning_schedulue)
        noclean_arealoss = _get_arealoss(soilrate, None)
        # Soiling Factor
        clean_soilingfactor = _get_soilingfactor(clean_arealoss, incidence_factor)
        noclean_soilingfactor = _get_soilingfactor(noclean_arealoss, incidence_factor)
        # Delta Soiling Factor
        delta_soilingfactor = clean_soilingfactor - noclean_soilingfactor
        delta_soilingfactor[np.isnan(delta_soilingfactor) | (delta_soilingfactor < 0)] = 0
        # Revenue
        delta_sector_irradiance = clean_reflected_irradiance * delta_soilingfactor
        return np.sum(delta_sector_irradiance,axis=1) * production_profit - sector_cleaningcost
    
    results = {'total_cleaning_cost': None,
               'degradation_cost': None,
               'cleaning_cost': None,
               'soiling_factor': None,
               'cleaning_actions': None,
               'actual_daycosts': None,
               'horizon_daycosts': None
               }
    # Setup constants
    f = 0
    doy = pd.to_datetime(opt.simulation_data.time[f]).dt.dayofyear # Day of year
    opt.field_model.helios.optical_efficiency[f][np.isnan(opt.field_model.helios.optical_efficiency[f])] = 0
    
    # sector_area = opt.field_model.helios.sector_area
                
    sector_area = opt.field_model.helios.sector_area.reshape(-1,1)
    
    clean_reflected_irradiance = opt.simulation_data.dni[f][np.newaxis,:] * sector_area * opt.field_model.helios.optical_efficiency[f] * opt.field_model.helios.nominal_reflectance
    production_profit = opt.plant.power_block_efficiency * (opt.electricity_price-opt.plant_other_maintenace) * opt.simulation_data.dt[f] / 3600.0
            
    sector_cleaningcost = (opt.truck['water_cost']+opt.truck['fuel_cost']) 
    
    n_sectors = opt.field_model.helios.tilt[f].shape[0]
    n_day_horizon = np.ceil(n_sectors / (n_sectors_per_truck*n_trucks))
    incidence_factor = opt.field_model.helios.inc_ref_factor[f]
    soil_rate = opt.field_model.helios.delta_soiled_area[f] # Actual soiling rate
    
    arealoss = np.zeros_like(soil_rate) # Actual area loss after rollout heuristic
    if initial_arealoss is None:
        initial_arealoss = np.zeros_like(soil_rate[:,0])
    cleaning_schedule = np.zeros_like(soil_rate) # Actual cleaning schedule after rollout heuristic
    
    for current_day in tqdm(np.unique(doy), desc="Processing days"): # Loop over each day of the year
        idx = _assign_indexing(current_day, doy, n_day_horizon) # get the indices of the current day and the prediction horizon
        
        # %% Soilfactor Trajectories
        # Current status
        if current_day == 1:
            arealoss[:,idx['current']] = np.cumsum(copy.deepcopy(soil_rate[:,idx['current']]),axis=1) + initial_arealoss.reshape(-1,1)
        else:
            arealoss[:,idx['current']] = np.cumsum(copy.deepcopy(soil_rate[:,idx['current']]),axis=1) + arealoss[:,idx['current'][0]-1].reshape(-1,1)
    
        # Soilrate Horizon
        horizon_soilrate = copy.deepcopy(soil_rate[:,idx['horizon']]) # Predictive area loss
        horizon_soilrate[:,0] += arealoss[:,idx['cleaning']]
        
        # %% Create 3D matrix for sector_revenue assuming one cleaning
        sector_revenue = np.zeros((n_sectors, int(n_day_horizon)))
        clean_idx_day = np.arange(int(n_day_horizon)) * 24
        for clean_day in np.arange(int(n_day_horizon)): # Determine sector revenue for each possible cleaning day in the horizon
            cleaning_schedule_day = np.zeros_like(horizon_soilrate)
            cleaning_schedule_day[:,clean_idx_day[clean_day]] = 1.0

            sector_revenue[:,clean_day] = _sector_revenue(horizon_soilrate, 
                                                          cleaning_schedule_day, 
                                                          incidence_factor[:,idx['horizon']], 
                                                          clean_reflected_irradiance[:,idx['horizon']], 
                                                          production_profit, 
                                                          sector_cleaningcost)
        
        # %% Priority Ranking
        cleaning_schedule_horizon = np.zeros((n_sectors, int(n_day_horizon)), dtype=int)
        
        available_heliostats = np.ones(n_sectors, dtype=bool)
        available_days = np.ones(int(n_day_horizon), dtype=bool)
        best_sector_revenue = np.zeros_like(sector_revenue)
        # available_days[np.sum(sector_revenue,axis=0) <= 0] = False
        
        for _ in range(n_sectors * int(n_day_horizon)): # Generate horizon cleaning schedule_
            valid_sector_revenue = sector_revenue * available_heliostats[:, np.newaxis] * available_days[np.newaxis, :]
            valid_options = (valid_sector_revenue) > 0

            if np.any(valid_options):
                # Find the sector with the highest revenue and add it to the cleaning schedule
                max_revenue_index = np.unravel_index(np.argmax(valid_sector_revenue), sector_revenue.shape)
                best_sector_revenue[max_revenue_index] = sector_revenue[max_revenue_index]
                sector_revenue[max_revenue_index] = 0
                
                cleaning_schedule_horizon[max_revenue_index] = 1
                cleaned_heliostat = max_revenue_index[0]
                cleaned_days =  np.where(cleaning_schedule_horizon[cleaned_heliostat,:] == 1)[0]
                
                # Check cleaning resource constraints
                available_heliostats[cleaned_heliostat] != np.sum(cleaning_schedule_horizon[cleaned_heliostat,:])==n_day_horizon
                available_days[np.sum(cleaning_schedule_horizon,axis=0) >= n_trucks * n_sectors_per_truck] = False
                
                # Update the heliostat sector revenue with the new fixed cleaning schedule
                valid_cleaning_days = available_heliostats[cleaned_heliostat] * available_days
                valid_cleaning_days[cleaned_days] = False # Remove the cleaned day from the available days
                
                if np.max(valid_cleaning_days):
                    possible_cleaning_days = np.where(valid_cleaning_days)[0]
                    for day_to_clean in possible_cleaning_days:
                        cleaning_schedule_day = np.zeros_like(horizon_soilrate[cleaned_heliostat,:])
                        cleaning_schedule_day[cleaned_days*24] = 1
                        try:
                            cleaning_schedule_day[day_to_clean*24] = 1
                        except IndexError:
                            print('f')
                        sector_revenue[cleaned_heliostat,day_to_clean] = _sector_revenue(horizon_soilrate[cleaned_heliostat,:].reshape(1,-1), 
                                                                                         cleaning_schedule_day.reshape(1,-1), 
                                                                                         incidence_factor[cleaned_heliostat,idx['horizon']].reshape(1,-1), 
                                                                                         clean_reflected_irradiance[cleaned_heliostat,idx['horizon']].reshape(1,-1), 
                                                                                         production_profit, 
                                                                                         sector_cleaningcost)
            elif available_days[0] == False:
                break
            else:
                # if opt.record_daily_costs == 1:
                #     cleaning_schedule_horizon
                #     print('f')
                break
            
        # %% Update conditions after cleaning
        # TODO: Fix indexing issues between horizon cleaning times and actual cleaning times (about 1 off?)
        cleaning_schedule[:,idx['cleaning']] = cleaning_schedule_horizon[:,0]
        arealoss[:,idx['cleaning']] = _clean_arealoss(arealoss[:,idx['cleaning']],cleaning_schedule[:,idx['cleaning']])
        
        if opt.record_daily_costs == 1:
            if current_day == 1:
                day_costs = {}
                day_total_cleaningcost = []
                day_cleaningcost = []
                day_degradationcost = []
                day_arealoss = []
            
            day_al = np.sum(arealoss[:,idx['current']],axis=1)
            day_soilingfactor = _get_soilingfactor(arealoss[:,:idx['cleaning']], incidence_factor[:,:idx['cleaning']])
            day_lostpower = _simulate_receiver_power(day_soilingfactor,
                                                        clean_reflected_irradiance[:,:idx['cleaning']],
                                                        opt.plant.receiver['thermal_max']*1e6,
                                                        opt.plant.receiver['thermal_min']*1e6,
                                                        opt.plant.receiver['thermal_losses']*1e6)
            day_tcc, day_ccl, day_cdeg = _cleaning_cost(day_lostpower, production_profit, opt, cleaning_schedule[:,:idx['cleaning']], n_trucks)
            day_total_cleaningcost.append(day_tcc)
            day_cleaningcost.append(day_ccl)
            day_degradationcost.append(day_cdeg)
            day_arealoss.append(day_al)
            
    # %% Simulate Plant Performance
    soiling_factor = np.zeros_like(arealoss)
    soiling_factor = _get_soilingfactor(arealoss,incidence_factor)
    lost_power = _simulate_receiver_power(soiling_factor, 
                                            clean_reflected_irradiance,
                                            opt.plant.receiver['thermal_max']*1e6,
                                            opt.plant.receiver['thermal_min']*1e6,
                                            opt.plant.receiver['thermal_losses']*1e6)
    
    results['total_cleaning_cost'] , results['cleaning_cost'], results['degradation_cost'] = _cleaning_cost(lost_power, production_profit, opt, cleaning_schedule, n_trucks)
    
    if opt.record_daily_costs == 1:
        day_costs = {'total_cleaning_cost': day_total_cleaningcost,
                     'cleaning_cost': day_cleaningcost,
                     'degradation_cost': day_degradationcost,
                     'arealoss': day_arealoss}
        results['day_costs'] = day_costs
    
    results['soiling_factor'] = soiling_factor
    results['cleaning_actions'] = cleaning_schedule
    return results

def _simulate_receiver_power(soiling_factor, clean_sector_power, receiver_max, receiver_min, receiver_losses, verbose=False):
    def __receiver_operation(sector_power, receiver_max, receiver_min):
        receiver_power = np.sum(sector_power,axis=0)
        receiver_saturation = (receiver_power > receiver_max)
        receiver_off = (receiver_power < receiver_min)
        receiver_power[receiver_saturation] = receiver_max
        receiver_power[receiver_off] = 0
        return receiver_power, receiver_off
        
    dirty_sector_power = copy.deepcopy(clean_sector_power) * soiling_factor
    
    clean_receiver_power, clean_receiver_off = __receiver_operation(clean_sector_power,receiver_max,receiver_min)
    dirty_receiver_power, dirty_receiver_off = __receiver_operation(dirty_sector_power,receiver_max,receiver_min)
    
    lost_power = clean_receiver_power-dirty_receiver_power
    lost_power[dirty_receiver_off & (clean_receiver_off==False)] -= receiver_losses  # subtract losses so we lose only net power

    return lost_power

def _cleaning_cost(lost_power, production_profit, opt, cleaning_schedule, n_trucks):
    """
        Calculates the total cleaning cost, including the cost of degradation and the variable and fixed costs of cleaning.
        
        Args:
            lost_power (numpy.ndarray): The amount of power lost due to soiling.
            production_profit (float): The profit per unit of power produced.
            opt (object): An options object containing various parameters related to the cleaning process.
            cleaning_schedule (numpy.ndarray): A 2D array indicating which sectors were cleaned on each day.
            n_trucks (int): The number of trucks used for cleaning.
        
        Returns:
            tuple:
                - total_cleaning_cost (float): The total cost of cleaning, including degradation.
                - cost_cleaning (float): The variable and fixed costs of the cleaning process.
                - cost_degradation (float): The cost of power degradation due to soiling.
    """
    cost_degradation = production_profit * np.sum(lost_power)
    
    n_sectors_cleaned = np.sum(cleaning_schedule)
    depreceiation_cost = opt.truck['purchase_cost']/opt.truck['useful_life']
    operator_cost = opt.truck['operators_per_truck_per_day']*opt.truck['operator_salary']
    cost_cleaning_fix = (depreceiation_cost+operator_cost+opt.truck['maintenance_costs'])*n_trucks
    cost_cleaning_var = (opt.truck['water_cost'] + opt.truck['fuel_cost'])*n_sectors_cleaned
    cost_cleaning = cost_cleaning_fix + cost_cleaning_var
    
    total_cleaning_cost = cost_degradation + cost_cleaning
    
    return total_cleaning_cost, cost_cleaning, cost_degradation