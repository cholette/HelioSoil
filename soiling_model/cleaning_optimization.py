import numpy as np
import pandas as pd
import copy
tol = np.finfo(float).eps # machine floating point precision
from functools import lru_cache
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.dates import DateFormatter

from soiling_model.base_models import simulation_inputs
from soiling_model.field_models import field_model,simplified_field_model,central_tower_plant
from soiling_model.utilities import _print_if,simple_annual_cleaning_schedule

class optimization_problem():
    def __init__(   self,params,solar_field,weather_files,climate_file,num_sectors=None,\
                    cleaning_rate:float=None,dust_type=None,n_az=10,n_el=10,second_surface=True,verbose=True,
                    model_type='semi-physical',ext_options={'grid_size_x':100}, n_modules = 1):
        self.electricity_price = []
        self.plant_other_maintenace = [] 

        pl = central_tower_plant()
        pl.import_plant(params)
        sd = simulation_inputs(weather_files,dust_type=dust_type)
        if model_type.lower() == 'semi-physical':
            fm = field_model(params,solar_field,cleaning_rate=cleaning_rate, n_modules=n_modules)
            fm.sun_angles(sd)
            fm.helios_angles(pl,second_surface=second_surface)
            fm.compute_acceptance_angles(pl)    
            fm.helios.compute_extinction_weights(sd,fm.loss_model,verbose=False,options=ext_options)
            fm.deposition_flux(sd)
            fm.adhesion_removal(sd)
            fm.calculate_delta_soiled_area(sd)
        elif model_type.lower() == 'simplified':
            fm = simplified_field_model(params,solar_field,cleaning_rate=cleaning_rate)
            fm.sun_angles(sd)
            fm.helios_angles(pl,second_surface=second_surface)
            fm.calculate_delta_soiled_area(sd)
        else:
            raise ValueError("Model type not recognized. Must be either semi-physical or simplified.")

        fm.optical_efficiency(pl,sd,climate_file,n_az=n_az,n_el=n_el,verbose=verbose)

        self.field_model = fm
        self.simulation_data = sd
        self.plant = pl

def find_truck_bounds(opt, verbose=True):
    """Find optimal truck bounds by analyzing cost trends.
    
    Args:
        opt (optimization_problem): Optimization problem instance
        verbose (bool): Print progress
        
    Returns:
        tuple: (min_trucks, max_trucks, cost_data)
    """
    print('Finding truck search bounds...')
    total_sectors = opt.field_model.helios.truck.sectors[0] * opt.field_model.helios.truck.sectors[1] * opt.field_model.helios.truck.sectors[2]
    sectors_per_truck = opt.field_model.helios.truck.n_sectors_per_truck
    
    # Start with single truck and increase until costs rise
    costs = []
    trucks = []
    n_trucks = 1
    cost_increased = False
    
    while True:
        # Calculate maximum cleanings possible with these trucks
        max_cleans = int(np.min([365 / (total_sectors / (n_trucks * sectors_per_truck)),365.0]))
        
        # Simulate with maximum cleaning rate
        schedule = periodic_schedule_tcc(opt, n_trucks, max_cleans, verbose=False)
        total_cost = schedule['total_cleaning_costs'][0]
        
        costs.append(total_cost)
        trucks.append(n_trucks)
        
        if verbose:
            print(f"Trucks: {n_trucks}, Max cleanings: {max_cleans}, Total cleaning cost: ${total_cost:,.2f}")
        
        # Check if costs have started increasing
        if len(costs) > 1 and costs[-1] > costs[-2]:
            if not cost_increased:
                # First time costs increase, check one more truck
                cost_increased = True
                n_trucks += 1
                continue
            else:
                # Costs increased for second time, we can break
                break
            
        n_trucks += 1
    
    # Find optimal number of trucks (minimum cost)
    optimal_trucks = trucks[np.argmin(costs)]
    
    # Set bounds +/- 2 trucks from optimal, respecting minimum of 1
    min_trucks = max(1, optimal_trucks - 2)
    max_trucks = optimal_trucks + 2
        
    return min_trucks, max_trucks, {'trucks': trucks, 'costs': costs}
    
def optimize_periodic_schedule(opt, file=0, verbose=True):
    """Optimizes the periodic cleaning schedule for a solar field optimization problem.
    
    This function performs a coarse grid search to find the optimal number of cleaning trucks and cleanings per year that minimize the total cleaning costs, including both degradation costs and direct cleaning costs.
    
    Args:
        opt (optimization_problem): An instance of the `soiling_model.cleaning_optimization.optimization_problem` class, containing the field model, plant, and simulation data.
        file (int, optional): The index of the simulation file to optimize for. Defaults to 0.
        verbose (bool, optional): Whether to print detailed output. Defaults to True.
    
    Returns:
        dict: A dictionary containing the following keys:
            - 'optimal_trucks': The optimal number of cleaning trucks.
            - 'optimal_cleans': The optimal number of cleanings per year.
            - 'optimal_results': The results of the optimal cleaning schedule.
            - 'bounds_data': The data used to determine the search bounds.
            - 'all_results'[nTrucks,nCleans]: A dictionary of all the results from the grid search.
    """
    # Initialize progress bar first
    progress_bar = tqdm(total=100, desc="Optimizing periodic schedule", leave=True)
    progress_bar.update(0)
    
    print("Optimizing periodic schedule...")
    # Find optimal truck bounds
    min_trucks, max_trucks, bounds_data = find_truck_bounds(opt, verbose)
    progress_bar.update(10)  # Update after finding bounds

    # Calculate remaining bounds
    total_sectors = opt.field_model.helios.truck.sectors[0] * opt.field_model.helios.truck.sectors[1] * opt.field_model.helios.truck.sectors[2]
    sectors_per_truck = opt.field_model.helios.truck.n_sectors_per_truck
    days_per_year = 365
    
    # Calculate cleaning frequency bounds
    min_cleans = int(np.ceil(total_sectors / (max_trucks * sectors_per_truck * days_per_year)))
    max_cleans = int(np.min([365 / (total_sectors / (max_trucks * sectors_per_truck)),365.0]))

    if verbose:
        print(f"Periodic Schedule Simulations Running Between:")
        print(f"Trucks: {min_trucks} to {max_trucks}")
        print(f"Cleanings per year: {min_cleans} to {max_cleans}\n")
        
    # Perform coarse grid search
    best_cost = np.inf
    best_trucks = None
    best_cleans = None
    best_schedule = None
    Nt = np.arange(min_trucks, max_trucks + 1)
    Nc = np.arange(min_cleans, max_cleans + 1)
    
    # Initialize results storage
    results = {}
    
    # Calculate total iterations for progress tracking
    total_iterations = sum(int(365 / (total_sectors / (n * sectors_per_truck))) - min_cleans + 1 
                         for n in Nt)
    current_iteration = 0
    remaining_progress = 90  # Remaining progress percentage for optimization loop
    
    # Main optimization loop
    for i, n_trucks in enumerate(Nt):
        nt_max_n_cleans = int(365 / (total_sectors / (n_trucks * sectors_per_truck)))
        nt_adjusted_nc = np.arange(min_cleans, nt_max_n_cleans + 1)
        for j, n_cleans in enumerate(nt_adjusted_nc):
            schedule = periodic_schedule_tcc(opt, n_trucks, n_cleans, verbose=False)
            total_cost = schedule['total_cleaning_costs'][file]
            results[(n_trucks, n_cleans)] = schedule
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_trucks = n_trucks
                best_cleans = n_cleans
                best_results = schedule
            
            # Update progress
            current_iteration += 1
            progress = int((current_iteration / total_iterations) * remaining_progress)
            progress_bar.update(progress - progress_bar.n + (100-remaining_progress))  # Update relative to previous progress
    
    progress_bar.close()

    if verbose:
        print(f"\nOptimal solution found:")
        print(f"Number of trucks: {best_trucks}")
        print(f"Cleanings per year: {best_cleans}")
        print(f"Total cleaning cost: ${best_cost:,.2f}/year")
        print(f"Direct cleaning cost: ${best_results['direct_cleaning_costs'][file]:,.2f}/year")
        print(f"Degradation cost: ${best_results['degradation_costs'][file]:,.2f}/year")

    return {
        'optimal_trucks': best_trucks,
        'optimal_cleans': best_cleans,
        'optimal_results': best_results,
        'bounds_data': bounds_data,
        'all_results': results
    }
    
def periodic_schedule_tcc(opt, n_trucks, n_cleans=None, verbose=True):
    """Computes the total cleaning costs (TCC) for a solar field optimization problem, including both degradation costs and direct cleaning costs.

    Args:
        opt (optimization_problem): An instance of the `soiling_model.cleaning_optimization.optimization_problem` class, containing the field model, plant, and simulation data.
        n_trucks (int): The number of cleaning trucks to use.
        n_cleans (int, optional): The number of cleanings to perform per year. If None or too high, 
                                uses maximum possible cleanings for given trucks
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
    # Calculate maximum possible cleanings
    total_sectors = opt.field_model.helios.truck.sectors[0] * opt.field_model.helios.truck.sectors[1] * opt.field_model.helios.truck.sectors[2]
    sectors_per_truck = opt.field_model.helios.truck.n_sectors_per_truck
    max_cleans = int(365 / (total_sectors / (n_trucks * sectors_per_truck)))
    
    # Check and adjust n_cleans if needed
    if n_cleans is None or n_cleans > max_cleans:
        if verbose:
            if n_cleans is None:
                print(f"No cleaning frequency provided. Using maximum: {max_cleans} cleanings/year")
            else:
                print(f"Requested cleanings ({n_cleans}) exceeds maximum possible ({max_cleans})")
                print(f"Adjusting to maximum: {max_cleans} cleanings/year")
        n_cleans = max_cleans
        
    # cleaning schedule, currently the same for all experiments/runs
    cleans = {k: [] for k in files}
    for f in files:
        n_helios = field.helios.tilt[f].shape[0]
        cleans[f] = simple_annual_cleaning_schedule(n_helios,n_trucks,n_cleans,\
            n_sectors_per_truck=field.helios.truck.n_sectors_per_truck, n_modules = opt.field_model.helios.truck.sectors[2])
    
    # compute reflectance losses (updates field.helios.soiling_factor)
    field.reflectance_loss(opt.simulation_data,cleans,opt.field_model.helios.truck.sectors[2],verbose=verbose) 

    C_deg = np.zeros(N_files)
    C_cl = np.zeros(N_files)   
    TCC = np.zeros(N_files)     
    Aj = field.helios.sector_area
    Aj = Aj.reshape((len(Aj),1))
    eta_pb = plant.plant['power_block_efficiency']
    P = plant.plant['electricity_price'] / 1e6
    COM = plant.plant['plant_other_maintenance'] /1e6
    Qloss = plant.receiver['thermal_losses']
    Qmin = plant.receiver['thermal_min']
    Qmax = plant.receiver['thermal_max']
    fmt_str = "Results for simulation {0:d}: \n  Total Cleaning Cost: {1:.2e} [$/yr]\n  Degradation Costs: {2:.2e} [$/yr] \n  Direct Cleaning Costs: {3:.2e} [$/yr]"
    
    for fi in range(N_files):
        f = files[fi]
        sf = field.helios.soiling_factor[f].copy()
        opt_eff = field.helios.optical_efficiency[f].copy()

        # nans are when the sun is below the stowangle. Set the optical efficiency to zero during these times and
        # set sf to an arbitrary value just to ensure that we get zero instead of nan when summing.
        sf[np.isnan(sf)] = 1 
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
        
        lost_power, receiver_saturation= _simulate_receiver_power(sf,clean_sector_reflected_power,Qmax*1e6,Qmin*1e6,Qloss*1e6)

        stow_elevation = field.sun.stow_angle
        sun_above_stow_elevation = (field.sun.elevation[f]>=stow_elevation)

        TCC[fi], C_cl[fi], C_deg[fi] = _cleaning_cost(lost_power*opt.field_model.helios.truck.sectors[2],alpha,opt,cleans[f],n_trucks)
        
        # compute direct costs
        _print_if(fmt_str.format(fi,C_deg[fi]+C_cl[fi],C_deg[fi],C_cl[fi]),verbose)
    
    # [!] Day TCC
    night_idx = np.where(opt.simulation_data.time[f].dt.hour == 23)[0]
    tcc_days = np.zeros_like(night_idx)
    ccl_days = np.zeros_like(night_idx)
    cdeg_days = np.zeros_like(night_idx)
    for idx in  np.arange(len(night_idx)):
        tcc_days[idx], ccl_days[idx], cdeg_days[idx] = _cleaning_cost(lost_power[:night_idx[idx]], alpha,opt,cleans[0][:,:night_idx[idx]],n_trucks)
    
    results = { 'n_trucks': n_trucks,
                'n_cleans': n_cleans,
                'total_cleaning_costs':TCC,
                'degradation_costs': C_deg,
                'direct_cleaning_costs': C_cl,
                'soiling_factor':field.helios.soiling_factor.copy(),
                'arealoss':field.helios.arealoss,
                'cleaning_actions':cleans,
                'soiling_induced_off_times':np.sum(~receiver_saturation['clean_field'] & receiver_saturation['soiled_field']),
                'soiling_induced_drops_below_upper_limit':np.sum(receiver_saturation['clean_field'] & ~receiver_saturation['soiled_field']),
                'day_tcc':tcc_days,
                'day_cdeg':cdeg_days,
                'day_ccl':ccl_days} #'day_tcc', day_tcc
    return results
def optimize_rollout_schedule(opt, file=0, verbose=True, max_trucks=20, initial_arealoss=None):
    """Optimizes the rollout cleaning schedule by finding optimal number of trucks.
    
    Args:
        opt (optimization_problem): Optimization problem instance
        file (int): File index to optimize for
        verbose (bool): Print progress
        max_trucks (int): Maximum number of trucks to consider
        
    Returns:
        dict: Dictionary containing:
            - optimal_trucks: Number of trucks that minimizes cost
            - optimal_results: Results from optimal configuration
            - bounds_data: Data from bounds search
            - all_results: Complete results from all configurations
    """
    # Initialize progress bar
    progress_bar = tqdm(total=100, desc="Optimizing rollout schedule", leave=True)
    progress_bar.update(0)
    
    print("Optimizing rollout schedule...")
    # Find initial bounds
    min_trucks, max_trucks, bounds_data = find_truck_bounds(opt, verbose)
    progress_bar.update(5)
    if verbose:
        print(f"Periodic Schedule Simulations Running Between:")
        print(f"Trucks: {min_trucks} to {max_trucks}")
    # Initialize results storage
    results = {}
    best_cost = np.inf
    best_trucks = None
    best_results = None
    
    # Main optimization loop
    while True:
        # Evaluate costs for current bounds
        trucks_to_evaluate = list(range(min_trucks, max_trucks + 1))
        total_iterations = len(trucks_to_evaluate)
        current_iteration = 0
        
        for n_trucks in trucks_to_evaluate:
            if n_trucks not in results:
                # Run rollout simulation
                schedule = rollout_heuristic_tcc(opt, n_trucks, initial_arealoss=initial_arealoss, method='greedy')
                total_cost = schedule['total_cleaning_costs'][file]
                results[n_trucks] = schedule
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_trucks = n_trucks
                    best_results = schedule
                
                if verbose:
                    print(f"Trucks: {n_trucks}, Total cleaning cost: ${total_cost:,.2f}")
            
            # Update progress
            current_iteration += 1
            progress = 5 + int((current_iteration / total_iterations) * 95)
            progress_bar.update(progress - progress_bar.n)
        
        # Check if optimal solution is at boundaries
        costs = [results[n]['total_cleaning_costs'][file] for n in trucks_to_evaluate]
        min_cost_idx = np.argmin(costs)
        
        # If minimum is at lower bound, extend lower
        if min_cost_idx == 0 and min_trucks > 1:
            max_trucks = trucks_to_evaluate[2]  # Keep 2 trucks above previous minimum
            min_trucks = max(1, min_trucks - 2)  # Extend 2 trucks lower
            continue
            
        # If minimum is at upper bound, extend upper
        if min_cost_idx == len(trucks_to_evaluate) - 1:
            min_trucks = trucks_to_evaluate[-3]  # Keep 2 trucks below previous maximum
            max_trucks = max_trucks + 2  # Extend 2 trucks higher
            continue
            
        # If minimum is internal, we're done
        break
    
    progress_bar.update(100 - progress_bar.n)
    progress_bar.close()

    if verbose:
        print(f"\nOptimal solution found:")
        print(f"Number of trucks: {best_trucks}")
        print(f"Total cleaning cost: ${best_cost:,.2f}/year")
        print(f"Direct cleaning cost: ${best_results['direct_cleaning_costs'][file]:,.2f}/year")
        print(f"Degradation cost: ${best_results['degradation_costs'][file]:,.2f}/year")

    return {
        'optimal_trucks': best_trucks,
        'optimal_results': best_results,
        'bounds_data': bounds_data,
        'all_results': results
    }
    
def rollout_heuristic_tcc(opt, n_trucks, initial_arealoss=None, method:str='greedy'):
    """Simulates the cleaning optimization process for a solar power plant using a rollout heuristic approach.
    
    Args:
        opt (OptimizationConfig): Configuration object containing simulation parameters and plant data
        n_trucks (int): Number of cleaning trucks available
        initial_arealoss (numpy.ndarray, optional): Initial area loss for each heliostat sector. Defaults to None.
        method (str): The optimization method to use, either 'greedy' or 'dynamic'.
    
    Returns:
        dict: Results dictionary containing:
            - total_cleaning_costs (float): Total annual cleaning cost [$/yr]
            - degradation_costs (float): Annual degradation cost [$/yr] 
            - direct_cleaning_costs (float): Annual direct cleaning cost [$/yr]
            - soiling_factor (numpy.ndarray): Soiling factor for each heliostat [-]
            - arealoss (numpy.ndarray): Cumulative area loss for each heliostat sector
            - cleaning_actions (numpy.ndarray): Cleaning schedule showing when sectors are cleaned
            - soiling_induced_off_times (int): Count of time steps receiver had to shut off due to soiling
            - soiling_induced_drops_below_upper_limit (int): Count of time steps power dropped below upper limit
            - day_costs (numpy.ndarray): Daily total cleaning costs
    """ 
    # # Helper functions
    def _clean_arealoss(area_loss, cleaning_schedule, cleaning_tech_curve=1.0):
        """Calculates the updated area loss after applying a cleaning event, taking into account the cleaning technology curve.
        
        Args:
            area_loss (numpy.ndarray): The area loss over time.
            cleaning_schedule (numpy.ndarray): The cleaning schedule, where 1 indicates a cleaning event.
            cleaning_tech_curve (float, optional): The cleaning technology efficiency factor to apply to the area loss after a cleaning event. 
                Defaults to perfect cleaning (eta_cl=100 %).
        
        Returns:
            numpy.ndarray: The updated area loss over time, with the cleaning events applied.
        """
        area_loss[cleaning_schedule==1] = area_loss[cleaning_schedule==1] * (1.0 - cleaning_tech_curve)
        return area_loss
                
    def _assign_indexing(current_day, day_of_year, n_day_horizon):
        """Assigns the indexing for the current day and prediction horizon based on the day of year and the number of days in the horizon.
        
        Args:
            current_day (int): The current day of the year.
            day_of_year (numpy.ndarray): The day of year for each data point.
            n_day_horizon (int): The number of days in the prediction horizon.
        
        Returns:
            dict: A dictionary containing the indices for the current day and prediction horizon.
        """
        
        idx = {'current': None, 'horizon': None, 'cleaning': None}
        idx['current'] = np.nonzero(day_of_year==current_day)[0]
        idx['horizon'] = np.nonzero((day_of_year >= np.mod(current_day, np.max(day_of_year) +1) +1) 
                                    & (day_of_year <= np.mod(current_day + n_day_horizon, np.max(day_of_year) +1) +1))[0]
        if len(idx['horizon']) == 0:
            idx['horizon'] = np.nonzero((day_of_year >= np.mod(current_day , np.max(day_of_year)+1)+1) 
                                        | (day_of_year <= np.mod(current_day + n_day_horizon, np.max(day_of_year) +1) +1))[0]
        idx['cleaning'] = idx['current'][-1]
        return idx
    
    def _dynamic_cleaning_schedule(sector_revenue, n_trucks, n_sectors_per_truck, n_sectors, n_day_horizon):
        """Determines optimal cleaning schedule horizon using dynamic programming.
        
        Args:
            sector_revenue (np.ndarray): Revenue matrix for each sector and day (n_sectors x n_day_horizon)
            n_trucks (int): Number of cleaning trucks available
            n_sectors_per_truck (int): Number of sectors each truck can clean per day
            n_sectors (int): Total number of sectors
            n_day_horizon (int): Number of days in planning horizon
        
        Returns:
            tuple: (optimal_schedule, best_revenue)
        TODO: Update to work
        """
        # Calculate maximum number of sectors that can be cleaned per day
        max_cleanings_per_day = n_trucks * n_sectors_per_truck
        # Initialize dynamic programming cache and decision tracking
        dp = {}
        decisions = {}

        @lru_cache(maxsize=None)
        def get_state_value(day, cleanings_left, cleaned_mask):
            # Base case: reached end of planning horizon
            if day >= n_day_horizon:
                return 0.0
                
            # Check if state has been computed before
            state = (day, cleanings_left, cleaned_mask)
            if state in dp:
                return dp[state]
            
            # Initialize with value of not cleaning any sectors today
            best_value = get_state_value(day + 1, max_cleanings_per_day, cleaned_mask)
            best_sectors = []
            
            # Only proceed if we have cleaning resources available
            if cleanings_left > 0:
                # Create list of uncleaned sectors with their revenues
                sector_revenues = [(s, sector_revenue[s, day]) for s in range(n_sectors)
                                if not (cleaned_mask & (1 << s))]
                # Sort sectors by revenue in descending order
                sector_revenues.sort(key=lambda x: x[1], reverse=True)
                
                # Consider only sectors we can clean with remaining resources
                sectors_to_try = sector_revenues[:cleanings_left]
                current_sectors = []
                current_value = 0
                
                # Evaluate each potential sector to clean
                for sector, rev in sectors_to_try:
                    # Only clean if revenue is positive
                    if rev > 0:
                        # Update mask to mark sector as cleaned
                        new_mask = cleaned_mask | (1 << sector)
                        current_sectors.append(sector)
                        current_value += rev
                        
                        # Calculate total value including future days
                        value = current_value + get_state_value(
                            day + 1,
                            max_cleanings_per_day,
                            new_mask
                        )
                        
                        # Update best solution if current is better
                        if value > best_value:
                            best_value = value
                            best_sectors = current_sectors.copy()
            
            # Cache results for this state
            dp[state] = best_value
            decisions[state] = best_sectors
            return best_value

        # Begin optimization from day 0
        initial_value = get_state_value(0, max_cleanings_per_day, 0)
        
        # Convert optimal decisions into cleaning schedule matrix
        schedule = np.zeros((n_sectors, n_day_horizon), dtype=int)
        cleaned_mask = 0
        
        # Fill schedule based on optimal decisions
        for day in range(n_day_horizon):
            sectors = decisions.get((day, max_cleanings_per_day, cleaned_mask), [])
            for sector in sectors:
                schedule[sector, day] = 1
                cleaned_mask |= (1 << sector)
        
        # Clear cache to free memory
        get_state_value.cache_clear()
        return schedule, initial_value    
    
    def _greedy_cleaning_schedule(sector_revenue, soilrate, incidence_factor, clean_reflected_irradiance, n_trucks, n_sectors_per_truck, n_sectors, n_day_horizon):
        """Determines cleaning schedule using a greedy approach.
        
        Args:
            sector_revenue (np.ndarray): Revenue matrix for each sector and day (n_sectors x n_day_horizon)
            n_trucks (int): Number of cleaning trucks available
            n_sectors_per_truck (int): Number of sectors each truck can clean per day
            n_sectors (int): Total number of sectors
            n_day_horizon (int): Number of days in planning horizon
        
        Returns:
            tuple: (cleaning_schedule_horizon, best_sector_revenue.sum())
        
        """
        cleaning_schedule_horizon = np.zeros((n_sectors, int(n_day_horizon)), dtype=int)
        cleaning_schedule_horizon_day = np.zeros_like(soilrate)
        
        available_heliostats = np.ones(n_sectors, dtype=bool)
        available_days = np.ones(int(n_day_horizon), dtype=bool)
        best_sector_revenue = np.zeros_like(sector_revenue)
        
        for _ in range(n_sectors * int(n_day_horizon)):
            valid_sector_revenue = sector_revenue * available_heliostats[:, np.newaxis] * available_days[np.newaxis, :]
            valid_options = (valid_sector_revenue) > 0

            if np.any(valid_options):
                # Find the sector with the highest revenue and add it tot he cleaning schedule
                max_revenue_index = np.unravel_index(np.argmax(valid_sector_revenue), sector_revenue.shape)
                best_sector_revenue[max_revenue_index] = sector_revenue[max_revenue_index]
                
                cleaning_schedule_horizon[max_revenue_index] = 1
                cleaning_schedule_horizon_day[max_revenue_index[0],max_revenue_index[1] * 24] = 1
                
                cleaned_heliostat = max_revenue_index[0]
                cleaned_day = max_revenue_index[1]
                cleaned_days = np.where(cleaning_schedule_horizon[cleaned_heliostat,:] == 1)[0]
                sector_revenue[cleaned_heliostat,cleaned_day] = 0
                # Recalculate revenues for all future days for this heliostat
                for future_day in range(cleaned_day+1, int(n_day_horizon)):
                    # Create temporary schedule for revenue calculation
                    temp_schedule = copy.copy(cleaning_schedule_horizon_day)
                    temp_schedule[cleaned_heliostat, future_day*24] = 1
                    
                    # Calculate new revenue considering locked in cleaning schedule
                    sector_revenue[cleaned_heliostat, future_day] = _sector_revenue(
                        soilrate[cleaned_heliostat,:].reshape(1,-1),
                        temp_schedule[cleaned_heliostat,:].reshape(1,-1),
                        incidence_factor[cleaned_heliostat,:].reshape(1,-1),
                        clean_reflected_irradiance[cleaned_heliostat,:].reshape(1,-1),
                        production_profit,
                        sector_cleaningcost,
                        existing_schedule=cleaning_schedule_horizon_day[cleaned_heliostat,:].reshape(1,-1)
                    )
                
                available_heliostats[cleaned_heliostat] != np.sum(cleaning_schedule_horizon[cleaned_heliostat,:])==n_day_horizon
                available_days[np.sum(cleaning_schedule_horizon,axis=0) >= n_trucks * n_sectors_per_truck] = False
                
                if available_days[0] == False: # Break the greedy algoirmth once we have the schedule for the current day filled.
                    break
            else:
                break
        
        return cleaning_schedule_horizon, best_sector_revenue.sum()
    
    # # Initialize results output
    files = list(opt.simulation_data.time.keys())
    N_files = len(files)    
    
    results = {'total_cleaning_costs': np.zeros(N_files), # [$/yr] TCC: Total cleaning cost
               'n_trucks': np.zeros(N_files), # [-] Number of trucks
               'degradation_costs': np.zeros(N_files), # [$/yr] DC: Degradation cost
               'direct_cleaning_costs': np.zeros(N_files), # [$/yr] CC: Cleaning cost
               'soiling_factor': {}, # [-] SF: Soiling factor 
               'cleaning_actions': {}, # [-] CA: Cleaning schedule (sectors, days)
               'soiling_induced_off_times': [], # [number of timesteps] where receiver off due to soiling
               'soiling_induced_drops_below_upper_limit': [], # [number of timesteps]  
               'day_costs': {}, # [dict] containing daily economic metrics
               }
    
    for f in files:
        # Setup constants
        doy = pd.to_datetime(opt.simulation_data.time[f]).dt.dayofyear # Day of year
        opt.field_model.helios.optical_efficiency[f][np.isnan(opt.field_model.helios.optical_efficiency[f])] = 0
        
        sector_area = np.repeat(opt.field_model.helios.sector_area.reshape(-1,1), repeats=opt.field_model.helios.truck.sectors[2], axis=0)
        
        clean_reflected_irradiance = opt.simulation_data.dni[f][np.newaxis,:] * sector_area * np.repeat(opt.field_model.helios.optical_efficiency[f], repeats=opt.field_model.helios.truck.sectors[2], axis=0) * opt.field_model.helios.nominal_reflectance
        production_profit = opt.plant.plant['power_block_efficiency'] * (opt.plant.plant['electricity_price']/1e6-opt.plant.plant['plant_other_maintenance']/1e6) * opt.simulation_data.dt[f] / 3600.0
                
        sector_cleaningcost = (opt.field_model.helios.truck.consumable_costs['total']) 
        
        n_sectors = opt.field_model.helios.truck.sectors[0] * opt.field_model.helios.truck.sectors[1] * opt.field_model.helios.truck.sectors[2]
        n_day_horizon = np.ceil(n_sectors / (opt.field_model.helios.truck.n_sectors_per_truck*n_trucks))
        incidence_factor = np.repeat(opt.field_model.helios.inc_ref_factor[f].copy(), repeats=opt.field_model.helios.truck.sectors[2], axis=0)
        soil_rate = np.repeat(opt.field_model.helios.delta_soiled_area[f], repeats=opt.field_model.helios.truck.sectors[2], axis=0) # Actual soiling rate
        
        arealoss = np.zeros_like(soil_rate) # Actual area loss after rollout heuristic
        if initial_arealoss is None:
            initial_arealoss = np.zeros_like(soil_rate[:,0])
        cleaning_schedule = np.zeros_like(soil_rate) # Actual cleaning schedule after rollout heuristic
        
        for current_day in tqdm(np.unique(doy),desc=f"Simulating Day ({method.capitalize()})", leave=False): # Loop over each day of the year
            idx = _assign_indexing(current_day, doy, n_day_horizon) # get the indices of the current day and the prediction horizon
            
            # Soilfactor Trajectories
            # Current status
            if current_day == 1:
                arealoss[:,idx['current']] = np.cumsum(copy.deepcopy(soil_rate[:,idx['current']]),axis=1) + initial_arealoss.reshape(-1,1)
            else:
                arealoss[:,idx['current']] = np.cumsum(copy.deepcopy(soil_rate[:,idx['current']]),axis=1) + arealoss[:,idx['current'][0]-1].reshape(-1,1)
        
            # Soilrate Horizon
            horizon_soilrate = copy.deepcopy(soil_rate[:,idx['horizon']]) # Predictive area loss rate
            horizon_soilrate[:,0] += arealoss[:,idx['cleaning']] # Add the last index of the current day i.e. posible 'cleaning' index arealoss to the first time period of the horizon
            
            # Create 3D matrix for sector_revenue assuming one cleaning
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
            
            # Choose optimization method
            if method.lower() == 'greedy':
                cleaning_schedule_horizon, _ = _greedy_cleaning_schedule(sector_revenue, 
                                                                         horizon_soilrate,
                                                                         incidence_factor[:,idx['horizon']],
                                                                         clean_reflected_irradiance[:,idx['horizon']], 
                                                                         n_trucks, 
                                                                         opt.field_model.helios.truck.n_sectors_per_truck, 
                                                                         n_sectors, int(n_day_horizon)
                )
            elif method.lower() == 'dynamic':
                cleaning_schedule_horizon, _ = _dynamic_cleaning_schedule(
                    sector_revenue, n_trucks, opt.field_model.helios.truck.n_sectors_per_truck, n_sectors, int(n_day_horizon)
                )
            else:
                raise ValueError("Method must be either 'greedy' or 'dynamic'")
            
            # Update conditions after cleaning
            cleaning_schedule[:,idx['cleaning']] = cleaning_schedule_horizon[:,0]
            arealoss[:,idx['cleaning']] = _clean_arealoss(
                arealoss[:,idx['cleaning']],
                cleaning_schedule[:,idx['cleaning']]
            )
            
            if hasattr(opt, 'record_daily_costs') and opt.record_daily_costs == 1:
                if current_day == 1:
                    day_costs = {}
                    day_total_cleaningcost = []
                    day_cleaningcost = []
                    day_degradationcost = []
                    day_arealoss = []
                
                day_al = np.sum(arealoss[:,idx['current']],axis=1)
                day_soilingfactor = _get_soilingfactor(arealoss[:,:idx['cleaning']], incidence_factor[:,:idx['cleaning']])
                day_lostpower, _ = _simulate_receiver_power(day_soilingfactor,
                                                            clean_reflected_irradiance[:,:idx['cleaning']],
                                                            opt.plant.receiver['thermal_max']*1e6,
                                                            opt.plant.receiver['thermal_min']*1e6,
                                                            opt.plant.receiver['thermal_losses']*1e6)
                day_tcc, day_ccl, day_cdeg = _cleaning_cost(day_lostpower, production_profit, opt, cleaning_schedule[:,:idx['cleaning']], n_trucks)
                day_total_cleaningcost.append(day_tcc)
                day_cleaningcost.append(day_ccl)
                day_degradationcost.append(day_cdeg)
                day_arealoss.append(day_al)
                
        # Simulate Plant Performance
        soiling_factor = _get_soilingfactor(arealoss,incidence_factor)
        lost_power, receiver_saturation = _simulate_receiver_power(soiling_factor, 
                                                clean_reflected_irradiance,
                                                opt.plant.receiver['thermal_max']*1e6,
                                                opt.plant.receiver['thermal_min']*1e6,
                                                opt.plant.receiver['thermal_losses']*1e6)
        
        results['total_cleaning_costs'][f] , results['direct_cleaning_costs'][f], results['degradation_costs'][f] = _cleaning_cost(lost_power, production_profit, opt, cleaning_schedule, n_trucks)
        
        if hasattr(opt, 'record_daily_costs') and opt.record_daily_costs == 1:
            day_costs = {'total_cleaning_costs'[f]: day_total_cleaningcost,
                        'direct_cleaning_costs'[f]: day_cleaningcost,
                        'degradation_costs'[f]: day_degradationcost,
                        'arealoss'[f]: day_arealoss}
            results['day_costs'][f] = day_costs
    
    soiling_factor_avg = np.zeros([opt.field_model.helios.truck.sectors[0] * opt.field_model.helios.truck.sectors[1], soiling_factor.shape[1]])
    for m in range(opt.field_model.helios.x.shape[0]):
        hel_idx = np.arange(m, opt.field_model.helios.x.shape[0] * opt.field_model.helios.truck.sectors[2], opt.field_model.helios.x.shape[0])
        hel_array = soiling_factor[hel_idx, :]
        soiling_factor_avg[m, :] = np.mean(hel_array, axis = 0)

    results['n_trucks'] = n_trucks
    results['soiling_induced_off_times'] = np.sum(~receiver_saturation['clean_field'] & receiver_saturation['soiled_field'])
    results['soiling_induced_drops_below_upper_limit'] = np.sum(receiver_saturation['clean_field'] & ~receiver_saturation['soiled_field'])            
    results['soiling_factor'][f] = soiling_factor_avg


    results['cleaning_actions'][f] = cleaning_schedule
    return results

# %% Cleaning optimization Utility Functions
def _get_arealoss(soil_rate, cleaning_schedule=None):
    """Calculates the cumulative area loss based on the area loss, cleaning schedule, and incidence factor.
    
    Args:
        soil_rate (numpy.ndarray): The area loss over time.
        cleaning_schedule (numpy.ndarray): The cleaning schedule, where 1 indicates a cleaning event.
    
    Returns:
        numpy.ndarray: The cumulative area loss over time [heliostat_sector, time_step].
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
    for i in range(soil_rate.shape[0]): # Loop through each heliostat sector
        cumulative = 0
        for j in range(soil_rate.shape[1]): # Loop through each time step
            if cleaning_schedule[i, j] == 1:
                cumulative = 0 # Reset the cumulative area loss if a cleaning event occurs 
            else:
                cumulative += soil_rate[i, j]
            area_loss[i, j] = cumulative
    return area_loss

def _get_soilingfactor(cumulative_area_loss,incidence_factor):
    """Calculates the soiling factor based on the area loss and incidence factor.
        
        Args:
            area_loss (numpy.ndarray): The cumulative area loss over time for each heliostat sector.
            incidence_factor (float): The incidence factor to apply to the cumulative loss.
        
        Returns:
            numpy.ndarray: The soiling factor over time.
    """                
    soiling_factor = 1 - (cumulative_area_loss * incidence_factor)
    soiling_factor[soiling_factor < 0] = 0
    soiling_factor[np.isnan(soiling_factor)] = 1
    return soiling_factor

def _sector_revenue(soilrate, cleaning_schedule, incidence_factor, clean_reflected_irradiance, production_profit, sector_cleaningcost, existing_schedule=None):
    """
    Calculates the sector revenue based on the soiling rate, cleaning schedule, incidence factor, clean reflected irradiance, production profit, and sector cleaning cost.
    
    Args:
        soilrate (numpy.ndarray): The soiling rate over time.
        cleaning_schedule (numpy.ndarray): The cleaning schedule, where 1 indicates a cleaning event.
        incidence_factor (float): The incidence factor to apply to the cumulative loss.
        clean_reflected_irradiance (numpy.ndarray): The clean reflected irradiance.
        production_profit (float): The production profit per unit of irradiance.
        sector_cleaningcost (float): The cost of cleaning per sector.
        existing_schedule(numpy.ndarray): The existing cleaning schedule to compare against, where 1 indicates a cleaning event.
    
    Returns:
        float: The total sector revenue.
    """
    # Area Loss
    clean_arealoss = _get_arealoss(soilrate, cleaning_schedule)
    noclean_arealoss = _get_arealoss(soilrate, existing_schedule)
    # Soiling Factor
    clean_soilingfactor = _get_soilingfactor(clean_arealoss, incidence_factor)
    noclean_soilingfactor = _get_soilingfactor(noclean_arealoss, incidence_factor)
    # Delta Soiling Factor
    delta_soilingfactor = clean_soilingfactor - noclean_soilingfactor
    delta_soilingfactor[np.isnan(delta_soilingfactor) | (delta_soilingfactor < 0)] = 0
    # Revenue
    delta_sector_irradiance = clean_reflected_irradiance * delta_soilingfactor
    return np.sum(delta_sector_irradiance,axis=1) * production_profit - sector_cleaningcost
    
def _simulate_receiver_power(soiling_factor, clean_sector_power, receiver_max, receiver_min, receiver_losses, verbose=False):
    """Calculates the power output of a receiver given the soiling factor and clean sector power.
        
        Args:
            soiling_factor (numpy.ndarray): The soiling factor for each sector.
            clean_sector_power (numpy.ndarray): The clean power output for each sector.
            receiver_max (float): The maximum power output of the receiver.
            receiver_min (float): The minimum power output of the receiver.
            receiver_losses (float): The power losses in the receiver.
            verbose (bool, optional): Whether to print verbose output.
        
        Returns:
            tuple:
                - lost_power (numpy.ndarray): The power lost due to soiling.
                - receiver_saturation (dict): A dictionary containing the saturation and off states of the receiver for both clean and soiled conditions.
    """
    def __receiver_operation(sector_power, receiver_max, receiver_min):
        """Calculates the power output of a receiver given the sector power, receiver maximum and minimum power limits.
        
        Args:
            sector_power (numpy.ndarray): The power output of each sector.
            receiver_max (float): The maximum power output of the receiver.
            receiver_min (float): The minimum power output of the receiver.
        
        Returns:
            tuple:
                - receiver_power (numpy.ndarray): The power output of the receiver.
                - receiver_off (numpy.ndarray): A boolean array indicating which sectors are below the receiver minimum power.
                - receiver_saturation (numpy.ndarray): A boolean array indicating which sectors are above the receiver maximum power.
        """
        receiver_power = np.sum(sector_power,axis=0)
        receiver_saturation = (receiver_power > receiver_max)
        receiver_off = (receiver_power < receiver_min)
        receiver_power[receiver_saturation] = receiver_max
        receiver_power[receiver_off] = 0
        return receiver_power, receiver_off, receiver_saturation
        
    dirty_sector_power = copy.deepcopy(clean_sector_power) * soiling_factor
    receiver_saturation = {}
    clean_receiver_power, clean_receiver_off, receiver_saturation['clean_field'] = __receiver_operation(clean_sector_power,receiver_max,receiver_min)
    dirty_receiver_power, dirty_receiver_off, receiver_saturation['soiled_field'] = __receiver_operation(dirty_sector_power,receiver_max,receiver_min)
    
    lost_power = clean_receiver_power-dirty_receiver_power
    lost_power[dirty_receiver_off & (clean_receiver_off==False)] -= receiver_losses  # subtract losses so we lose only net power

    return lost_power, receiver_saturation

def _cleaning_cost(lost_power, production_profit, opt, cleaning_schedule, n_trucks):
    """Calculate total cleaning cost including degradation and operational costs.
        
    Args:
        lost_power (numpy.ndarray): Power lost due to soiling
        production_profit (float): Profit per unit power produced
        opt (object): Optimization object containing truck parameters
        cleaning_schedule (numpy.ndarray): 2D array of cleaning events
        n_trucks (int): Number of trucks used
        
    Returns:
        tuple:
            - total_cleaning_cost (float): Total annual cleaning cost [$/yr]
            - cost_cleaning (float): Direct cleaning costs [$/yr]
            - cost_degradation (float): Degradation costs [$/yr]
    """   
    # Soil degradation cost
    cost_degradation = production_profit * np.sum(lost_power)  # [$/year]
    
    # Fixed cleaning costs
    depreciation_cost = opt.field_model.helios.truck._params.cost_purchase / opt.field_model.helios.truck._params.useful_life  # [$/truck/year]
    operator_cost = opt.field_model.helios.truck._params.salary_operator  # [$/truck/year]
    maintenance_cost = opt.field_model.helios.truck._params.cost_maintenance  # [$/truck/year]
    cost_cleaning_fix = (depreciation_cost + operator_cost + maintenance_cost) * n_trucks  # [$/year]
    
    # Variable cleaning costs
    n_sectors_cleaned = np.sum(cleaning_schedule)
    cost_cleaning_var =opt.field_model.helios.truck.consumable_costs['total'] * n_sectors_cleaned  # [$/year]
    
    # s
    cost_cleaning = cost_cleaning_fix + cost_cleaning_var  # [$/year]
    total_cleaning_cost = cost_degradation + cost_cleaning  # [$/year]
    
    return total_cleaning_cost, cost_cleaning, cost_degradation

def plot_optimization_results(results: dict, file: int = 0,save_path: str = None) -> tuple:
    """Plot optimization results showing cost surface for truck and cleaning combinations.
    """    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data and organize into matrix
    trucks = sorted(list(set(k[0] for k in results['all_results'].keys())))
    cleans = sorted(list(set(k[1] for k in results['all_results'].keys())))
    
    # Create cost matrix
    TCC = np.full((len(trucks), len(cleans)), np.nan)
    for (n_truck, n_clean), result in results['all_results'].items():
        i = trucks.index(n_truck)
        j = cleans.index(n_clean)
        TCC[i,j] = result['total_cleaning_costs'][file]
    
    # Generate colors for each truck count
    colors = plt.cm.viridis(np.linspace(0, 1, len(trucks)))
    
    # Plot cost curves for each truck count and their optimal points
    for i, n_trucks in enumerate(trucks):
        valid_costs = ~np.isnan(TCC[i,:])
        if np.any(valid_costs):
            # Plot line
            ax.plot(np.array(cleans)[valid_costs], TCC[i,valid_costs]/1e6, 
                   color=colors[i], label=f'{n_trucks} Trucks', linewidth=2)
            
            # Plot optimal point for this truck count with matching color
            opt_idx = np.nanargmin(TCC[i,:])
            if not np.isnan(TCC[i,opt_idx]):
                ax.plot(cleans[opt_idx], TCC[i,opt_idx]/1e6, 
                       marker='*', markersize=15, color=colors[i], linestyle='None')
    
    # Plot overall optimal point
    opt_idx = np.nanargmin(TCC)
    opt_row, opt_col = np.unravel_index(opt_idx, TCC.shape)
    label_str = f"Optimal ({trucks[opt_row]} trucks, {cleans[opt_col]} cleans)"
    ax.plot(cleans[opt_col], TCC[opt_row,opt_col]/1e6, 
           color=colors[opt_row], marker='*', markersize=20, 
           label=label_str, linestyle='None')
    
    # First add the legend in upper right
    legend = ax.legend(fontsize=12, loc='upper right')
    
    # Get the position of the legend
    legend_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    
    # Position text box just below the legend
    text_x = 0.98# Same x position as legend
    text_y = legend_bbox.y0 - 0.05 # Slightly below legend
    
    # Add cost breakdown text box
    textstr = '\n'.join((
        f'Optimal Configuration:',
        f'Trucks: {trucks[opt_row]}, Cleanings/year: {cleans[opt_col]}',
        f'Total Cleaning Cost: ${TCC[opt_row,opt_col]/1e6:.2f}M/year',
        f'Direct Cleaning: ${results["optimal_results"]["direct_cleaning_costs"][file]/1e6:.2f}M/year',
        f'Degradation: ${results["optimal_results"]["degradation_costs"][file]/1e6:.2f}M/year'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(text_x, text_y, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    max_cost = np.nanmax(TCC)/1e6  # Convert to millions
    ax.set_ylim(0, max_cost * 0.5)
    
    # Customize plot
    ax.set_xlabel("Number of Field Cleans per Year", fontsize=14)
    ax.set_ylabel("Total Cleaning Cost [M$/yr]", fontsize=14)
    ax.set_title("Optimization of Field Cleaning Operations", fontsize=14, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, which='major', linestyle='-', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    ax.minorticks_on()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_soiling_factor(results_schedule: dict, file: int = 0, save_path: str = None) -> tuple:
    """Plot soiling factor evolution and cleaning schedule results.
    
    Args:
        results_schedule (dict): Results from periodic_schedule_tcc
        file (int): File index to plot
        save_path (str, optional): Path to save figure
        
    Returns:
        tuple: (fig, ax) Matplotlib figure and axes objects
    """    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract soiling factor data
    sf = results_schedule['soiling_factor'][file]
    sf[sf==1.0] = np.nan
    # Plot field average
    ax.plot(sf.mean(axis=0), label="Field Average", linewidth=2)
    
    # Add reference lines
    ax.axhline(y=np.nanmean(sf), color='red', ls='--', label="Average")
    
    # Calculate and plot initial/final values
    valid_indices = ~np.isnan(sf[0,:])
    initial_sf = np.nanmean(sf[:,valid_indices][:,0])
    final_sf = np.nanmean(sf[:,valid_indices][:,-1])
    
    ax.axhline(y=initial_sf, color='green', ls='--', label="Initial")
    ax.axhline(y=final_sf, color='blue', ls='--', label="Final")
    
    # Add cleaning count information
    cleans = results_schedule['cleaning_actions'][file]
    total_cleans = cleans.sum()
    
    avg_cleans = total_cleans / len(sf)
    # Calculate y-axis limits
    ymin = np.floor(np.nanmin(sf.mean(axis=0)) * 10) / 10  # Round down to nearest 0.05
    ax.set_ylim(ymin, 1.0)  # Set y-axis from calculated minimum to 100%
    
    
    # Customize plot
    ax.set_xlabel("Hour of the year", fontsize=14)
    ax.set_ylabel("Soiling factor [-]", fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=12)
    
    # Add title with key metrics
    title = (f"Trucks {results_schedule['n_trucks']}, Avg. Cleanings {avg_cleans:.1f}\n"
            f"Total Cleanings: {total_cleans:.0f}, Mean Soiling Factor: {np.nanmean(sf):.3f}\n"
            f"Initial SF: {initial_sf:.3f}, Final SF: {final_sf:.3f}, Delta SF: {final_sf - initial_sf:.3f}\n"
            f"Receiver off periods: {results_schedule['soiling_induced_off_times']:.0f}, "
            f"Below saturation: {results_schedule['soiling_induced_drops_below_upper_limit']:.0f}")
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_cleaning_schedule(opt, results_schedule: dict, file: int = 0, save_path: str = None) -> tuple:
    """Plot cleaning schedule showing when each sector is cleaned.
    
    Args:
        results_schedule (dict): Results from periodic_schedule_tcc
        file (int): File index to plot
        save_path (str, optional): Path to save figure
        
    Returns:
        tuple: (fig, ax) Matplotlib figure and axes objects
    """    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract cleaning schedule data
    cleaning_actions = results_schedule['cleaning_actions'][file]
    n_sectors, n_timesteps = cleaning_actions.shape
    time_daily = pd.to_datetime(opt.simulation_data.time[file]).dt.floor('D')
    
    df_cleaning_actions = pd.DataFrame(cleaning_actions.T, index=time_daily).resample('D').max()
    
    # Calculate cleaning statistics
    cleanings_per_sector = cleaning_actions.sum(axis=1)
    total_cleanings = int(cleanings_per_sector.sum())
    avg_cleanings = total_cleanings / n_sectors
    
    ax = plt.subplot()
    for col in df_cleaning_actions.columns:
        mask = df_cleaning_actions[col] == 1
        ax.scatter(df_cleaning_actions.index[mask], df_cleaning_actions[col][mask] * int(col+1), marker='.', color='blue')
        
    # Customize plot
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Sector", fontsize=14)
    ax.set_xlim(df_cleaning_actions.index[0], df_cleaning_actions.index[-1])
    plt.title(f'Cleaning Schedule (blue dot = clean)\nTrucks: {results_schedule['n_trucks']}, Avg Cleanings: {avg_cleanings:.1f}\nTotal Cleanings: {total_cleanings}', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.set_major_formatter(DateFormatter('%m'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_soiled_optical_efficiency(opt, results_schedule: dict, file: int = 0, save_path: str = None) -> tuple:
    """Plot optical efficiency evolution including soiling effects and operational limits.
    
    Args:
        opt (optimization_problem): Optimization problem instance
        results_schedule (dict): Results from periodic_schedule_tcc
        file (int): File index to plot
        save_path (str, optional): Path to save figure
        
    Returns:
        tuple: (fig, ax) Matplotlib figure and axes objects
    """
    def add_limits(ax, ulim, llim, DMW):
        """Add operational limits to the plot."""
        xl = ax.get_xlim()
        xloc = xl[0] + 0.0*(xl[1]-xl[0])
        for ii in range(len(ulim)):
            if ii == 0:
                ax.axhline(y=ulim[ii], color='gray', ls='--', 
                          label='Receiver capacity limit')
            else:
                ax.axhline(y=ulim[ii], color='gray', ls='--', label=None)

            if ulim[ii] < 1.0:
                ax.text(xloc, ulim[ii], f"DNI={DMW[ii]*1e6:.0f}", 
                       va='bottom')

            if ii == 0:
                ax.axhline(y=llim[ii], color='gray', ls=':', 
                          label='Receiver lower limit')
            else:
                ax.axhline(y=llim[ii], color='gray', ls=':', label=None)

            if llim[ii] > 0.2:
                ax.text(xloc, llim[ii], f"DNI={DMW[ii]*1e6:.0f}", 
                       va='bottom')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    sf = results_schedule['soiling_factor'][file]
    eta_clean = opt.field_model.helios.optical_efficiency[file] * \
                opt.field_model.helios.nominal_reflectance
    eta_soiled = sf * eta_clean
    
    # Calculate limits
    receiver_area = opt.field_model.helios.sector_area.sum()
    DNI_MW = 1e-6*np.array([200,300,500,700,900,1000])
    u_limits = opt.plant.receiver['thermal_max'] / DNI_MW / receiver_area
    l_limits = opt.plant.receiver['thermal_min'] / DNI_MW / receiver_area
    
    # Plot optical efficiency
    ax.plot(opt.simulation_data.time[file],np.nanmean(eta_soiled, axis=0), 
            label="Field Average (above stow angle only)", 
            linewidth=2)
    
    # Add mean line
    ax.axhline(y=np.nanmean(eta_soiled), color='red', 
               ls='--', label="Average")
    
    # Add operational limits
    add_limits(ax, u_limits, l_limits, DNI_MW)
    
    # Add info text box
    textstr = '\n'.join((
        f'Configuration:',
        f'Trucks: {results_schedule["n_trucks"]}',
        f'Field Cleans: {results_schedule["n_cleans"]}',
        f'Average Soiling Factor: {np.nanmean(sf):.3f}',
        f'Hours Off: {results_schedule["soiling_induced_off_times"]}',
        f'Hours Below Limit: {results_schedule["soiling_induced_drops_below_upper_limit"]}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', 
            bbox=props)
    
    # Customize plot
    ax.set_xlabel("Month", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylabel("Field average efficiency w/ soiling", fontsize=14)
    ax.set_title("Field Optical Efficiency with Soiling Effects", 
                 fontsize=14, pad=20)
    ax.xaxis.set_major_formatter(DateFormatter('%m'))
    ax.grid(True)
    ax.legend(fontsize=12)
    ax.set_ylim((0, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig, ax
    