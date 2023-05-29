import numpy as np
tol = np.finfo(float).eps # machine floating point precision
from soiling_model.base_models import *
from soiling_model.utilities import _print_if,simple_annual_cleaning_schedule

class optimization_problem():
    def __init__(self,params,solar_field,weather_files,climate_file,num_sectors,\
        dust_type=None,n_az=10,n_el=10,second_surface=True,verbose=True):
        self.truck = {  'operator_salary':[],
                        'operators_per_truck_per_day':[],
                        'purchase_cost':[],
                        'maintenance_costs':[],
                        'useful_life': [],
                        'fuel_cost': [],
                        'water_cost': []
                    }
        self.electricty_price = []
        plant_other_maintenace = [] 

        pl = plant()
        pl.import_plant(params)

        fm = field_model(params,solar_field,num_sectors=num_sectors)

        if fm.loss_model=="mie":
            ValueError("Field simulation using loss_model==""mie"" not yet available.")

        sd = simulation_inputs(weather_files,dust_type=dust_type)
        fm.sun_angles(sd)
        fm.helios_angles(pl,second_surface=second_surface)
        fm.deposition_flux(sd)
        fm.adhesion_removal(sd)
        fm.helios.compute_extinction_weights(sd,fm.loss_model,verbose=True)
        fm.calculate_delta_soiled_area(sd)
        fm.optical_efficiency(pl,sd,climate_file,n_az=n_az,n_el=n_el,verbose=verbose)

        self.field_model = fm
        self.simulation_data = sd
        self.plant = pl

def periodic_schedule_tcc(opt,n_trucks,n_cleans,\
    n_sectors_per_truck=1,verbose=True):

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
    Aj = field.helios.sector_area
    Aj = Aj.reshape((len(Aj),1))
    eta_pb = plant.power_block_efficiency
    P = opt.electricty_price
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
        
        # reflected power when perfectly clean
        clean_sector_reflected_power = DNI*Aj*eta_clean
    
        clean_receiver_power = np.sum(clean_sector_reflected_power,axis=0)
        clean_receiver_saturated = (clean_receiver_power>Qmax*1e6)
        clean_receiver_off = (clean_receiver_power<Qmin*1e6)
        clean_receiver_power[clean_receiver_saturated]=Qmax*1e6
        clean_receiver_power[clean_receiver_off] = 0

        dirty_receiver_power = np.clip(np.sum(clean_sector_reflected_power*sf,axis=0),None,Qmax*1e6)
        dirty_receiver_saturated = (dirty_receiver_power>Qmax*1e6)
        dirty_receiver_off = (dirty_receiver_power<Qmin*1e6)
        dirty_receiver_power[dirty_receiver_saturated]=Qmax*1e6
        dirty_receiver_power[dirty_receiver_off] = 0

        stow_elevation = field.sun.stow_angle
        sun_above_stow_elevation = (field.sun.elevation[f]>=stow_elevation)

        _print_if("Number of time steps where sun is above stow elevation:{0:.1f} ({1:.1f} degrees)".format(sun_above_stow_elevation.sum(),stow_elevation),verbose)
        _print_if("Number of time steps where clean receiver would be on: {0:.0f}".format( np.sum(~clean_receiver_off) ),verbose)
        _print_if("Number of time steps where actual receiver is on: {0:.0f}".format( np.sum(~dirty_receiver_off) ),verbose)
        
        _print_if("Number of time steps where clean receiver would be saturated: {0:.0f}".format(clean_receiver_saturated.sum()),verbose)
        _print_if("Number of time steps where actual receiver is saturated: {0:.0f}".format(dirty_receiver_saturated.sum()),verbose)

        lost_power = clean_receiver_power-dirty_receiver_power
        lost_power[dirty_receiver_off & (clean_receiver_off==False)] -= Qloss*1e6  # subtract losses so we lose only net power
        C_deg[fi] = alpha*np.sum(lost_power) 

        # compute direct costs
        number_of_sectors_cleaned = cleans[f].sum()
        depreciation_cost = opt.truck['purchase_cost']/opt.truck['useful_life']
        operator_cost = opt.truck['operators_per_truck_per_day']*opt.truck['operator_salary']
        C_cl_fix = (depreciation_cost+operator_cost+opt.truck['maintenance_costs'])*n_trucks
        C_cl_var = (opt.truck['water_cost'] + opt.truck['fuel_cost'])*number_of_sectors_cleaned
        C_cl[fi] = C_cl_fix + C_cl_var
        _print_if(fmt_str.format(fi,C_deg[fi]+C_cl[fi],C_deg[fi],C_cl[fi]),verbose)

    TCC = (C_cl + C_deg)        
    results = { 'total_cleaning_costs':TCC,
                'degradation_costs': C_deg,
                'direct_cleaning_costs': C_cl,
                'soiling_factor':field.helios.soiling_factor,
                'cleaning_actions':cleans}
    return results
