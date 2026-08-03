# %% Analysis of Raygen Heliostats data (Carwarp)

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
main_directory = os.path.abspath(os.path.join(script_dir, ".."))
for root, dirs, files in os.walk(main_directory):
    sys.path.append(root)

import numpy as np
import pandas as pd
import soiling_model.base_models as smb
import soiling_model.fitting as smf
import soiling_model.utilities as smu
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from paper_specific_utilities import plot_for_paper, plot_for_heliostats, daily_soiling_rate, \
                                     fit_quality_plots, summarize_fit_quality,plot_experiment_PA, \
                                     daily_soiling_tilt_all_data
import scipy.stats as sps
from collections import defaultdict
from datetime import datetime
import warnings
import re

# %% Analysis setup

# CHOOSE RAYGEN LOCATION - heliostat analysis is only supported for Carwarp
raygen_site = "Carwarp"

# CHOOSE PM FRACTION TO USE FOR FITTING
# dust_type = "PM10" # choose PM fraction to use for analysis --> PMT, PM10, PM2.5
dust_type = "PM2.5" # choose PM fraction to use for analysis --> PMT, PM10, PM2.5

# CHOOSE WHETHER TO USE DAILY AVERAGE OF REFLECTANCE VALUES OR ARVO DATA ONLY
DAILY_AVERAGE = True

# CHOOSE TRAIN EXPERIMENTS
train_experiments = [0] # indices for training experiments from 0 to len(files)-1

# CHOOSE TRAIN MIRRORS (default = True, otherwise choose individual mirrors)
train_mirror_default = True # Set the default train mirrors for Carwarp ["ON_M1_T00"] and Yadnarie ["ONE_M2_T00"]
# train_mirrors_custom = ["OAV_M1_T00"] # WORKS ONLY IF train_mirror_defaul is False
train_mirrors_custom = ["OAV_M2_T60"] # WORKS ONLY IF train_mirror_defaul is False
# train_mirrors_custom = ["ONW_M4_T00","OSE_M4_T00","ONE_M2_T00"] # WORKS ONLY IF train_mirror_defaul is False
# train_mirrors_custom = ["ONW_M2_T00","ONE_M2_T00","OSW_M2_T00"] # WORKS ONLY IF train_mirror_defaul is False

# Heliostat analysis is forced on for this script (only supported for Carwarp data)
hel_analysis = True
if raygen_site == "Carwarp":
    HELIOSTATS = hel_analysis
else:
    HELIOSTATS = False

# default to False, set to True to save outputs (and override any previous saved output)
save_output = False

# %% Set data input

site_paths = {
    "Carwarp": {
        "sp_save_file": f"{main_directory}/results/sp_fitting_results_mildura_hel",
        "cm_save_file": f"{main_directory}/results/cm_fitting_results_mildura_hel",
        "d": f"{main_directory}/data/mildura/",
        "parameter_file": "parameters_mildura_experiments.xlsx",
        "compass": 1, # cardinal directions are N, E, S, W
        "default_train_mirror" : "ON_M1_T00",
        "all_intervals": np.array([
            [np.datetime64('2024-01-30T10:00:00'), np.datetime64('2024-02-05T08:00:00')], # January 2024 (first experiments --- nothing to remove)
            [np.datetime64('2024-06-06T17:00:00'), np.datetime64('2024-06-11T08:00:00')]], # June 2024 (second experiments --- already removed rainy data - consider adding them back)
            dtype='datetime64[m]')
    },
    "Yadnarie": {
        "sp_save_file": f"{main_directory}/results/sp_fitting_results_yadnarie",
        "cm_save_file": f"{main_directory}/results/cm_fitting_results_yadnarie",
        "d": f"{main_directory}/data/yadnarie/",
        "parameter_file": "parameters_yadnarie_experiments.xlsx",
        "compass": 2, # cardinal directions are NE, SE, SW, NW
        "default_train_mirror" : "ONE_M2_T00",
        "all_intervals": np.array([
            [np.datetime64('2024-11-11T20:30:00'), np.datetime64('2024-11-16T07:30:00')], # November 2024 # First set of mirrors
            [np.datetime64('2024-11-12T11:40:00'), np.datetime64('2024-11-16T07:30:00')]], # November 2024 # Second set of mirrors
            dtype='datetime64[m]')
    } }

if raygen_site in site_paths:
    sp_save_file = site_paths[raygen_site]['sp_save_file']
    cm_save_file = site_paths[raygen_site]['cm_save_file']
    d = site_paths[raygen_site]['d']
    parameter_file = d+site_paths[raygen_site]['parameter_file']
    train_mirrors = [site_paths[raygen_site]['default_train_mirror']] if train_mirror_default else train_mirrors_custom
    all_intervals = site_paths[raygen_site]['all_intervals']
    print('Analysis of ' + raygen_site + ' data')
else:
    raise ValueError(f"Invalid site '{raygen_site}'. Choose a valid site for analysis.")

pad = 0.05

reflectometer_incidence_angle = 15 # [deg] angle of incidence of reflectometer
reflectometer_acceptance_angle = 12.5e-3 # [rad] half acceptance angle of reflectance measurements
second_surf = True # True if using the second-surface model. Otherwise, use first-surface

time_to_remove_at_end = [0,0,0,0,0]  # time to be removed for each experiment, in hours
k_factor = "import" # None sets equal to 1.0, "import" imports from the file


# %% Get file list and time intervals. Import training data.

files,all_intervals,exp_mirrors,all_mirrors = smu.get_training_data(d,"experiment_",time_to_remove_at_end=time_to_remove_at_end)
orientation = [ [s[1:1+site_paths[raygen_site]['compass']] for s in mirrors] for mirrors in exp_mirrors]

testing_intervals = all_intervals

Nfiles = len(files)
extract = lambda x,ind: [x[ii] for ii in ind]
files_train = extract(files,train_experiments)
training_intervals = extract(all_intervals,train_experiments)
testing_intervals = list(all_intervals)
t = [t for t in train_experiments]
plot_title = "Training: "+str(train_mirrors)+", Exp: "+str(t)

# %% Import training data
imodel = smf.semi_physical(parameter_file)
imodel_constant = smf.constant_mean_deposition(parameter_file)
sim_data_train = smb.simulation_inputs( files_train,
                                        k_factors=k_factor,
                                        dust_type=dust_type
                                        )
reflect_data_train = smb.reflectance_measurements(  files_train,
                                                    sim_data_train.time,
                                                    number_of_measurements=6.0,
                                                    reflectometer_incidence_angle=reflectometer_incidence_angle,
                                                    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
                                                    import_tilts=True,
                                                    column_names_to_import=train_mirrors
                                                    )
# %% Trim training data
sim_data_train,reflect_data_train = smu.trim_experiment_data(   sim_data_train,
                                                                reflect_data_train,
                                                                training_intervals
                                                            )

sim_data_train,reflect_data_train = smu.trim_experiment_data(   sim_data_train,
                                                                reflect_data_train,
                                                                training_intervals
                                                            )


# %% Plot training data

for ii,experiment in enumerate(train_experiments):
    if any("augusta".lower() in value.lower() for value in sim_data_train.file_name.values()):
        fig,ax = plot_experiment_PA(sim_data_train,reflect_data_train,ii,figsize=(10,15))
    else:
        fig,ax = smu.plot_experiment_data(sim_data_train,reflect_data_train,ii,figsize=(10,15))

    # fig.suptitle(f"Training Data for file {files[experiment]}")
    # fig,ax = smu.wind_rose(sim_data_train,ii)
    # ax.set_title(f"Wind for file {files[experiment]}")

# %% Load total simulation data

if HELIOSTATS==True:
    files,_,exp_mirrors,all_mirrors = smu.get_training_data(d,"hel_experiment_",time_to_remove_at_end=time_to_remove_at_end)


sim_data_total = smb.simulation_inputs( files,
                                        k_factors=k_factor,
                                        dust_type=dust_type
                                        )

# %% Load total reflectance data
if HELIOSTATS:
    n_meas = 36.0
else:
    n_meas = 6.0

reflect_data_total = smb.reflectance_measurements(  files,
                                                    sim_data_total.time,
                                                    number_of_measurements=n_meas,
                                                    reflectometer_incidence_angle=reflectometer_incidence_angle,
                                                    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
                                                    import_tilts=True,
                                                    column_names_to_import=None
                                                    )

# %% compute daily_averaged values of reflectance to avoid morning-afternoon (not understood) recoveries
if DAILY_AVERAGE:
    reflect_data_total = smu.daily_average(reflect_data_total,
                                           sim_data_total.time,
                                           dt=None)

# %% Trim data and plot
sim_data_total,reflect_data_total = smu.trim_experiment_data(   sim_data_total,
                                                                reflect_data_total,
                                                                all_intervals
                                                            )

for ii,experiment in enumerate(sim_data_total.dt.keys()):
    if any("augusta".lower() in value.lower() for value in sim_data_total.file_name.values()):
            fig,ax = plot_experiment_PA(sim_data_total,reflect_data_total,ii,figsize=(10,15))
    else:
        fig,ax = smu.plot_experiment_data(sim_data_total,reflect_data_total,ii,figsize=(10,15))
    # fig.suptitle(f"Testing Data for file {files[experiment]}")
    # fig,ax = smu.wind_rose(sim_data_total,ii)
    # ax.set_title(f"Wind for file {files[experiment]}")

# %% Daily average of reflectance values and trimming of simulation inputs (Training data)

if DAILY_AVERAGE:
    reflect_data_train = smu.daily_average(reflect_data_train,sim_data_train.time,sim_data_train.dt)    # compute daily_averaged values of reflectance to avoid morning-afternoon (not understood) recoveries
    # sim_data_train , _ = smu.trim_experiment_data(      sim_data_train,                                 # trim the correspoding simulation inputs to align with the new reflectance values (start and end time can be modified by the average)
    #                                                 reflect_data_train,
    #                                                 "reflectance_data")

# %% Daily average of reflectance values and trimming of simulation inputs (Total data)

if DAILY_AVERAGE:
    reflect_data_total = smu.daily_average(reflect_data_total,sim_data_total.time,sim_data_total.dt)
    # sim_data_total , _ = smu.trim_experiment_data(      sim_data_total,
    #                                                 reflect_data_total,
    #                                                 "reflectance_data" )

# %% Plot training data after daily averaging
if DAILY_AVERAGE:
    for ii,experiment in enumerate(train_experiments):
        if any("augusta".lower() in value.lower() for value in sim_data_train.file_name.values()):
            fig,ax = plot_experiment_PA(sim_data_train,reflect_data_train,ii)
        else:
            fig,ax = smu.plot_experiment_data(sim_data_train,reflect_data_train,ii)

        fig.suptitle(f"Averaged Training Data for file {files_train[0]}")
        # fig,ax = smu.wind_rose(sim_data_train,ii)
        # ax.set_title(f"Wind for file {files[experiment]}"

# %% Plot total data after daily averaging
if DAILY_AVERAGE:
    for ii,experiment in enumerate(files):
        if any("augusta".lower() in value.lower() for value in sim_data_total.file_name.values()):
            fig,ax = plot_experiment_PA(sim_data_total,reflect_data_total,ii)
        else:
            fig,ax = smu.plot_experiment_data(sim_data_total,reflect_data_total,ii)

# %% PLOT EXPERIMENTAL DATA

for f in range(len(reflect_data_total.average)):
    lgd_size=10
    fig,ax = plt.subplots(nrows=4,figsize=(12,15),gridspec_kw={'hspace': 0.3})

    ave = reflect_data_total.average[f]
    t = reflect_data_total.times[f]
    # if t[0]
    std = reflect_data_total.sigma[f]
    lgd_label = [re.sub(r"_T\d{2,3}", "", lg.replace("O", "").replace("_M", "")) for lg in exp_mirrors[f]]
    for ii in range(ave.shape[1]):
        if (lgd_label[ii]=='W1' and pd.Timestamp(t[0].astype('datetime64[ns]')).month==1) or (lgd_label[ii]=='W2' and pd.Timestamp(t[0].astype('datetime64[ns]')).month==6) or lgd_label[ii]=='NE1':
            ax[0].errorbar(t,ave[:,ii],yerr=1.96*std[:,ii],label=lgd_label[ii],linestyle='dashed',marker='o',capsize=4.0)
        elif 'AV' in lgd_label[ii]:
            continue
        else:
            ax[0].errorbar(t,ave[:,ii],yerr=1.96*std[:,ii],label=lgd_label[ii],marker='o',capsize=4.0)
    ax[0].grid(True)
    # label_str = r"Reflectance at {0:.0f} $^{{\circ}}$".format(reflect_data_total.reflectometer_incidence_angle[0])
    label_str = "Measured Reflectance"
    ax[0].set_ylabel(label_str)
    ax[0].legend(fontsize=lgd_size,loc='center right',bbox_to_anchor=(1.15,0.5))
    month_plot = datetime.fromisoformat(str(reflect_data_total.times[f][0])).strftime('%B')
    year_plot = datetime.fromisoformat(str(reflect_data_total.times[f][0])).year
    title_plot = raygen_site + " Experiments Summary - " + month_plot + " " + str(year_plot)
    if DAILY_AVERAGE:
        title_plot += ' - Daily Average'
    if HELIOSTATS:
        title_plot += ' - HELIOSTATS'
    plt.suptitle(title_plot,fontsize = 20,x=0.5,y=0.92)

    ax[1].plot(sim_data_total.time[f],sim_data_total.dust_concentration[f],color='brown',label="Measurements")#     if f == 0 else '')
    # f==0 and ...
    ax[1].axhline(y=sim_data_total.dust_concentration[f].mean(),color='brown',ls='--',label = r"Average = {0:.2f}".format(sim_data_total.dust_concentration[f].mean()))
    label_str = r'{0:s} [$\mu g\,/\,m^3$]'.format(sim_data_total.dust_type[0])
    ax[1].set_ylabel(label_str,color='brown',fontsize=20)
    ax[1].tick_params(axis='y', labelcolor='brown')
    ax[1].grid(True)
    ax[1].legend(fontsize=lgd_size)
    # ax[1].set_ylim(0,1.1*(sim_data_total.dust_concentration[f][~np.isnan(sim_data_total.dust_concentration[f])].max()))
    ax[1].set_ylim(0,30)

    ax[2].plot(sim_data_total.time[f],sim_data_total.wind_speed[f],color='green',label="Measurements")#     if f == 0 else '')
    # f==0 and ...
    ax[2].axhline(y=sim_data_total.wind_speed[f].mean(),color='green',ls='--',label = r"Average = {0:.2f}".format(sim_data_total.wind_speed[f].mean()))
    label_str = r'Wind Speed [$m\,/\,s$]'
    ax[2].set_ylabel(label_str,color='green')
    ax[2].tick_params(axis='y', labelcolor='green')
    ax[2].grid(True)
    ax[2].legend(fontsize=lgd_size)

    ax[3].plot(sim_data_total.time[f],sim_data_total.relative_humidity[f],color='blue',label="Measurements")#     if f == 0 else '')
    # f==0 and ...
    ax[3].axhline(y=sim_data_total.relative_humidity[f][~np.isnan(sim_data_total.relative_humidity[f])].mean(),color='blue',ls='--',label = r"Average = {0:.1f}".format(sim_data_total.relative_humidity[f][~np.isnan(sim_data_total.relative_humidity[f])].mean()))
    label_str = r'Relative Humidity [%]'
    ax[3].set_ylabel(label_str,color='blue')
    ax[3].tick_params(axis='y', labelcolor='blue')
    ax[3].grid(True)
    ax[3].legend(fontsize=lgd_size)
    ax[3].set_ylim(0,100)

    [axis.tick_params(axis='x', rotation=15) for axis in ax]

    plt.show()

# %% PLOT average REFLECTANCE LOSSES, RELATIVE HUMIDITY, and PM10 between measurements
for ii in range(len(reflect_data_total.average)):

    # Convert arrays to pandas DatetimeIndex for easier slicing
    sim_times = pd.to_datetime(sim_data_total.time[ii])  # Simulation time
    sim_humidity = sim_data_total.relative_humidity[ii]  # Simulation relative humidity
    sim_dust_conc = sim_data_total.dust_concentration[ii] # Simulation dust concentration
    reflect_times = pd.to_datetime(reflect_data_total.times[ii].astype('datetime64[ns]'))  # Reflectance times

    # Initialize an empty list to store average values
    relative_humidity_averages = []
    dust_conc_averages = []

    # Compute average relative humidity for each interval
    for start, end in zip(reflect_times[:-1], reflect_times[1:]):
        # Select humidity values between start and end times
        mask = (sim_times >= start) & (sim_times < end)
        average_humidity = np.mean(sim_humidity[mask])  # Compute average
        dust_conc = np.mean(sim_dust_conc[mask])  # Compute average
        relative_humidity_averages.append(average_humidity)
        dust_conc_averages.append(dust_conc)

    # Convert results to a NumPy array (optional)
    relative_humidity_averages = np.array(relative_humidity_averages)
    dust_conc_averages = np.array(dust_conc_averages)

    # Compute average reflectance losses for horizontal mirrors (check it is true for January experiments)
    ref_loss_ave = np.mean(reflect_data_total.delta_ref[ii][1:, [2, -1]] * 1e2, axis=1)

    # Print the results
    print("Relative Humidity Averages:", relative_humidity_averages)
    print("Dust Concentration Averages", dust_conc_averages)
    print("Reflectance Loss Averages", ref_loss_ave)

    fig, ax1 = plt.subplots()
    ax1.plot(reflect_data_total.times[ii][1:],ref_loss_ave, color='tab:blue')
    ax1.set_xlabel('Time')  # x-axis label
    ax1.set_ylabel('Reflectance Loss, %' , color='tab:blue')  # y-axis label for the first plot

    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(reflect_data_total.times[ii][1:],relative_humidity_averages, color='tab:green')
    ax2.set_ylabel('Relative Humidity, %', color='tab:green')  # y-axis label for the second plot
    ax2.tick_params(axis='y', labelcolor='tab:green')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
    ax3.plot(reflect_data_total.times[ii][1:],dust_conc_averages, color='tab:orange', label='Dust Concentration')
    ax3.set_ylabel('Dust Concentration, µg/m³', color='tab:orange')
    ax3.tick_params(axis='y', labelcolor='tab:orange')

    plt.show()

# %% Plot reflectance losses in each interval for train mirrors (with mirror rig), or horizontal heliostat (H58)
if HELIOSTATS==True:
    train_mirrors = ['H58']
for m,mir in enumerate(train_mirrors):
    for exp in sim_data_total.time.keys():
        diff_array_times = -np.diff(reflect_data_total.times[exp])
        diff_days = -diff_array_times.astype('timedelta64[s]').astype('int')/3600/24
        if mir in reflect_data_total.mirror_names[exp]:
            idx_mir = reflect_data_total.mirror_names[exp].index(mir)
        else:
            warnings.warn(f"{mir} not found in mirror names. Exiting loop.")
            break
        diff_ref = -np.diff(reflect_data_total.average[exp][:,idx_mir],axis=0)
        diff_rates = diff_ref*100/diff_days  # contain soiling rates for mir in training_mirrors

        df_dust = pd.DataFrame({'Time': sim_data_total.time[exp],
                                'Value':sim_data_total.dust_concentration[exp]})
        retiming_vector = pd.to_datetime(reflect_data_total.times[exp].astype('datetime64[ns]'))
        df_dust['Interval'] = pd.cut(df_dust['Time'], bins=retiming_vector, right=False, labels=False)
        df_dust_retime = pd.DataFrame({
            'Time': retiming_vector[1:],
            'Mean_Dust_Conc': df_dust.groupby('Interval')['Value'].mean().values,
            'Soiling_Rate': diff_rates
            })
        df_sorted = df_dust_retime.sort_values(by='Mean_Dust_Conc')
        print(df_dust_retime)
        print(df_sorted)
        df_filtered = df_sorted.dropna(subset=['Soiling_Rate'])  # remove NaNs that would make the plot incomplete
        plt.plot(df_filtered['Mean_Dust_Conc'], df_filtered['Soiling_Rate'], marker='o', linestyle='-', color='b')
        # plt.plot(df_sorted['Mean_Dust_Conc'], df_sorted['Soiling_Rate'], marker='o', linestyle='-', color='b')
        # plt.hist(df_sorted['Soiling_Rate'], bins='auto', edgecolor='black')
        plt.xlabel(f"Mean_{dust_type}, µg/m3")
        plt.ylabel('Soiling_Rate, %/day')
        plt.title(f"Soiling Rates for {files[exp]}")
        plt.show()

# %% Set mirror angles and get extinction weights for fitting (using train data)
imodel.helios_angles(sim_data_train,reflect_data_train,second_surface=second_surf)
imodel.helios.compute_extinction_weights(sim_data_train,imodel.loss_model,verbose=True,options={"grid_size_x":100,"grid_size_mu":10000})
imodel.helios.plot_extinction_weights(sim_data_train,fig_kwargs={})
ext_weights = imodel.helios.extinction_weighting[0].copy()

imodel_constant.helios_angles(sim_data_train,reflect_data_train,second_surface=second_surf)
file_inds = np.arange(len(files_train))
imodel_constant = smu.set_extinction_coefficients(imodel_constant,ext_weights,file_inds)

# %% Fit semi-physical model
log_param_hat,log_param_cov = imodel.fit_mle(   sim_data_train,
                                                reflect_data_train,
                                                transform_to_original_scale=False)

s = np.sqrt(np.diag(log_param_cov))
param_ci = log_param_hat + 1.96*s*np.array([[-1],[1]])
lower_ci = imodel.transform_scale(param_ci[0,:])
upper_ci = imodel.transform_scale(param_ci[1,:])
param_hat = imodel.transform_scale(log_param_hat)
hrz0_mle,sigma_dep_mle = param_hat
print(f'hrz0: {hrz0_mle:.2e} [{lower_ci[0]:.2e},{upper_ci[0]:.2e}]')
print(f'\sigma_dep: {sigma_dep_mle:.2e} [{lower_ci[1]:.2e},{upper_ci[1]:.2e}] [p.p./day]')

imodel.update_model_parameters(param_hat)
imodel.save(f"{sp_save_file}_{dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}",
            log_p_hat=log_param_hat,
            log_p_hat_cov=log_param_cov,
            training_simulation_data=sim_data_train,
            training_reflectance_data=reflect_data_train)

# %% Fit constant mean model
log_param_hat_con,log_param_cov_con = imodel_constant.fit_mle(  sim_data_train,
                                                                reflect_data_train,
                                                                transform_to_original_scale=False)
s_con = np.sqrt(np.diag(log_param_cov_con))
param_ci_con = log_param_hat_con + 1.96*s_con*np.array([[-1],[1]])
lower_ci_con = imodel_constant.transform_scale(param_ci_con[0,:])
upper_ci_con = imodel_constant.transform_scale(param_ci_con[1,:])
param_hat_con = imodel_constant.transform_scale(log_param_hat_con)
mu_tilde,sigma_dep_con = param_hat_con
print(f'mu_tilde: {mu_tilde:.2e} [{lower_ci_con[0]:.2e},{upper_ci_con[0]:.2e}] [p.p./day]')
print(f'\sigma_dep (constant mean model): {sigma_dep_con:.2e} [{lower_ci_con[1]:.2e},{upper_ci_con[1]:.2e}] [p.p./day]')

imodel_constant.update_model_parameters(param_hat_con)
imodel_constant.save(f"{cm_save_file}_{dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}",
                     log_p_hat=log_param_hat_con,
                     log_p_hat_cov=log_param_cov_con,
                     training_simulation_data=sim_data_train,
                     training_reflectance_data=reflect_data_train)

# %% Compute daily averaged data of training data and updated imodel (DOES THIS MAKE SENSE?? IT IS DOING TWICE THE SAME JOB)
if DAILY_AVERAGE:
    sim_data_train , reflect_data_train = smu.trim_experiment_data( sim_data_train,      # trim the correspoding simulation inputs to align with the new reflectance values (start and end time can be modified by the average)
                                                                    reflect_data_train,
                                                                    "reflectance_data")

# %% updated imodel with new daily-averaged training data
if DAILY_AVERAGE:
    imodel.helios_angles(sim_data_train,reflect_data_train,second_surface=second_surf)
    file_inds = np.arange(len(files_train))
    imodel = smu.set_extinction_coefficients(imodel,ext_weights,file_inds)  # reuse weights computed above instead of recomputing
    imodel_constant.helios_angles(sim_data_train,reflect_data_train,second_surface=second_surf)
    imodel_constant = smu.set_extinction_coefficients(imodel_constant,ext_weights,file_inds)
    imodel_constant.save(f"{cm_save_file}_{dust_type.replace('.', '-')}{'_update_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}")

# %% plot SP on training data
_,_,_ = imodel.plot_soiling_factor( sim_data_train,
                            reflectance_data=reflect_data_train,
                            reflectance_std='mean',
                            save_path=f"{sp_save_file}_TRAINING_{dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}",
                            # fig_title="On Training Data (semi-physical)",
                            orientation_strings=orientation,
                            figsize=[12,8])

# %% plot CM on training data
_,_,_ = imodel_constant.plot_soiling_factor(    sim_data_train,
                                                reflectance_data=reflect_data_train,
                                                reflectance_std='mean',
                                                save_path=f"{cm_save_file}_TRAINING_{dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}",
                                                # fig_title="On Training Data",
                                                orientation_strings=orientation,
                                                figsize = [12,8]  )

# %% Compute daily averaged data of total data (DOES THIS MAKE SENSE?? IT IS DOING TWICE THE SAME JOB)
if DAILY_AVERAGE:
    sim_data_total , reflect_data_total = smu.trim_experiment_data( sim_data_total,      # trim the correspoding simulation inputs to align with the new reflectance values (start and end time can be modified by the average)
                                                                    reflect_data_total,
                                                                    "reflectance_data")

# %% Compute average measured reflectance losses for each tilt (only for mirror rig experiments)

# if not HELIOSTATS:
#     ref_data_summary = smu.soiling_rates_summary(reflect_data_total,sim_data_total)
#     ref_data_summary

# %% Performance of semi-physical model on total data
imodel.helios_angles(sim_data_total,reflect_data_total,second_surface=second_surf)
file_inds = np.arange(len(reflect_data_total.file_name))
imodel = smu.set_extinction_coefficients(imodel,ext_weights[0].reshape(1, -1),file_inds)
#%% Plot semi-physical model results
if HELIOSTATS==True:
    fig,ax,ref_hel_sp = plot_for_heliostats( imodel,
                                    reflect_data_total,
                                    sim_data_total,
                                    train_experiments,
                                    train_mirrors,
                                    orientation,
                                    legend_shift=(0.04,0),
                                    yticks=(0.97,0.98,0.99,1.0))#0.97,0.98,
    fig.set_size_inches(10, 20)

    df_hel_sp = smu.loss_hel_table_from_sim(ref_hel_sp,sim_data_total)
    if save_output:
        df_hel_sp.to_csv(f"{sp_save_file}_HEL_{dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}.csv", index=False)

else:
    fig,ax,ref_simulation_sp = plot_for_paper( imodel,
                                reflect_data_total,
                                sim_data_total,
                                train_experiments,
                                train_mirrors,
                                orientation,
                                legend_shift=(0.04,0),
                                # yticks=(0.75,0.80,0.85,0.90,0.95,1.00))
                                yticks=(0.94,0.95,0.96,0.97,0.98,0.99,1.00)) #

    df_sim_sp = smu.loss_table_from_sim(ref_simulation_sp,sim_data_total)
    if save_output:
        df_sim_sp.to_csv(f"{sp_save_file}_{dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}.csv", index=False)

if save_output:
    fig.suptitle('Semi-Physical Model', fontsize=16, fontweight='bold', y=1.045)
    fig.savefig(f"{sp_save_file}_{dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}.pdf", bbox_inches='tight')
    fig.savefig(f"{sp_save_file}_{dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}.png", bbox_inches='tight')

plt.show()

# %%

# df_plot_sp =smu.loss_table_from_fig(ref_simulation_sp,sim_data_total) # NEED TO BE FIXED


# %% Performance of constant-mean model on total data
imodel_constant.helios_angles(sim_data_total,reflect_data_total,second_surface=second_surf)

if HELIOSTATS:
    fig,ax,ref_hel_cm = plot_for_heliostats(   imodel_constant,
                                    reflect_data_total,
                                    sim_data_total,
                                    train_experiments,
                                    train_mirrors,
                                    orientation,
                                    legend_shift=(0.04,0),
                                    yticks=(0.97,0.98,0.99,1.02))
    df_hel_cm = smu.loss_hel_table_from_sim(ref_hel_cm,sim_data_total)
    if save_output:
        df_hel_cm.to_csv(f"{cm_save_file}_HEL_{dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}.csv", index=False)

else:
    fig,ax,ref_simulation_cm = plot_for_paper(    imodel_constant,
                                reflect_data_total,
                                sim_data_total,
                                train_experiments,
                                train_mirrors,
                                orientation,
                                legend_shift=(0.04,0),
                                # yticks=(0.75,0.80,0.85,0.90,0.95,1.00))
                                yticks=(0.94,0.95,0.96,0.97,0.98,0.99,1.00)) #

    df_sim_cm = smu.loss_table_from_sim(ref_simulation_cm,sim_data_total)
    if save_output:
        df_sim_cm.to_csv(f"{cm_save_file}_{dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}.csv", index=False)

if save_output:
    fig.suptitle('Constant-Mean Model', fontsize=16, fontweight='bold', y=1.045)
    fig.savefig(f"{cm_save_file}_{dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}.pdf", bbox_inches='tight')
    fig.savefig(f"{cm_save_file}_{dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}.png", bbox_inches='tight')

plt.show()

# %% ============================================================
# YADNARIE "WHAT-IF" HELIOSTAT SCENARIO
# =================================================================
# Yadnarie has no real heliostat field data (no "hel_experiment_" files). This section asks:
# "if the same heliostats (i.e. the same tilt-vs-time history measured at Carwarp) were deployed at
# Yadnarie, what soiling would they experience under Yadnarie's own dust/wind conditions?"
#
# The deposition/removal model is refit from scratch on Yadnarie's own mirror-rig training data and
# parameter file (site-specific dust/mirror properties), then the fitted model is driven with Yadnarie's
# own environmental time series (dust concentration, wind speed, humidity) while the heliostat tilt
# history is borrowed from the real Carwarp heliostats (tiled to cover however long the Yadnarie
# experiments run). There is no real heliostat reflectance measurement at Yadnarie to compare against,
# so the prediction is simulated from a clean start (rho0=None).

yad_site_cfg = site_paths["Yadnarie"]
yad_d = yad_site_cfg["d"]
yad_parameter_file = yad_d+yad_site_cfg["parameter_file"]
yad_train_mirrors = [yad_site_cfg["default_train_mirror"]]
yad_dust_type = "PM10" # Yadnarie analysis uses PM10, independent of the main site's dust_type

# %% Get Yadnarie file list and time intervals. Import Yadnarie training data.
yad_files,yad_all_intervals,yad_exp_mirrors,yad_all_mirrors = smu.get_training_data(yad_d,"experiment_",time_to_remove_at_end=time_to_remove_at_end)

# The yadnarie folder contains superseded/duplicate extractions alongside the corrected file for each
# campaign (e.g. "experiment_20241111_20241116.xlsx" vs "..._this_is_the_right_one.xlsx" for the same
# dates), so a plain "experiment_" prefix match picks up 5 files instead of the 2 real campaigns
# (November 2024, February 2025). Keep only the files marked as the corrected one for each campaign.
yad_keep = [ii for ii,f in enumerate(yad_files) if "this_is_the_right_one" in f]
yad_files = [yad_files[ii] for ii in yad_keep]
yad_all_intervals = yad_all_intervals[yad_keep]
yad_exp_mirrors = [yad_exp_mirrors[ii] for ii in yad_keep]

yad_files_train = extract(yad_files,train_experiments)
yad_training_intervals = extract(yad_all_intervals,train_experiments)

# %% Fit semi-physical & constant-mean models on Yadnarie's own mirror-rig training data
imodel_yad = smf.semi_physical(yad_parameter_file)
imodel_constant_yad = smf.constant_mean_deposition(yad_parameter_file)

sim_data_train_yad = smb.simulation_inputs( yad_files_train,
                                        k_factors=k_factor,
                                        dust_type=yad_dust_type
                                        )
reflect_data_train_yad = smb.reflectance_measurements(  yad_files_train,
                                                    sim_data_train_yad.time,
                                                    number_of_measurements=6.0,
                                                    reflectometer_incidence_angle=reflectometer_incidence_angle,
                                                    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
                                                    import_tilts=True,
                                                    column_names_to_import=yad_train_mirrors
                                                    )

sim_data_train_yad,reflect_data_train_yad = smu.trim_experiment_data(   sim_data_train_yad,
                                                                reflect_data_train_yad,
                                                                yad_training_intervals
                                                            )

if DAILY_AVERAGE:
    reflect_data_train_yad = smu.daily_average(reflect_data_train_yad,sim_data_train_yad.time,dt=None)
    sim_data_train_yad , reflect_data_train_yad = smu.trim_experiment_data( sim_data_train_yad,
                                                                    reflect_data_train_yad,
                                                                    "reflectance_data")

imodel_yad.helios_angles(sim_data_train_yad,reflect_data_train_yad,second_surface=second_surf)
imodel_yad.helios.compute_extinction_weights(sim_data_train_yad,imodel_yad.loss_model,verbose=True,options={"grid_size_x":100,"grid_size_mu":10000})
ext_weights_yad = imodel_yad.helios.extinction_weighting[0].copy()

imodel_constant_yad.helios_angles(sim_data_train_yad,reflect_data_train_yad,second_surface=second_surf)
yad_train_file_inds = np.arange(len(yad_files_train))
imodel_constant_yad = smu.set_extinction_coefficients(imodel_constant_yad,ext_weights_yad,yad_train_file_inds)

log_param_hat_yad,log_param_cov_yad = imodel_yad.fit_mle(   sim_data_train_yad,
                                                reflect_data_train_yad,
                                                transform_to_original_scale=False)
param_hat_yad = imodel_yad.transform_scale(log_param_hat_yad)
imodel_yad.update_model_parameters(param_hat_yad)
imodel_yad.save(f"{sp_save_file}_YADNARIE_{yad_dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+yad_train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}",
            log_p_hat=log_param_hat_yad,
            log_p_hat_cov=log_param_cov_yad,
            training_simulation_data=sim_data_train_yad,
            training_reflectance_data=reflect_data_train_yad)

log_param_hat_con_yad,log_param_cov_con_yad = imodel_constant_yad.fit_mle(  sim_data_train_yad,
                                                                reflect_data_train_yad,
                                                                transform_to_original_scale=False)
param_hat_con_yad = imodel_constant_yad.transform_scale(log_param_hat_con_yad)
imodel_constant_yad.update_model_parameters(param_hat_con_yad)
imodel_constant_yad.save(f"{cm_save_file}_YADNARIE_{yad_dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+yad_train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}",
                     log_p_hat=log_param_hat_con_yad,
                     log_p_hat_cov=log_param_cov_con_yad,
                     training_simulation_data=sim_data_train_yad,
                     training_reflectance_data=reflect_data_train_yad)

# %% Load Yadnarie's total environmental data (dust, wind, humidity) across all its experiments
sim_data_total_yad = smb.simulation_inputs( yad_files,
                                        k_factors=k_factor,
                                        dust_type=yad_dust_type
                                        )
sim_data_total_yad,_ = smu.trim_experiment_data( sim_data_total_yad,
                                                None,
                                                yad_all_intervals
                                            )

# %% Plot losses as a function of (constant) tilt (Yadnarie)
tilts = [0,10,30,45,60,75,90]
M = 1000
cm_save_file_full_yad = f"{cm_save_file}_YADNARIE_{yad_dust_type.replace('.', '-')}{'_train_'+str(train_experiments)}{'_'+yad_train_mirrors[0][-3:]}{'_daily' if DAILY_AVERAGE else ''}"
ave,low,high,med = [],[],[],[]
for t in tilts:
    sims,a,a2 = daily_soiling_tilt_all_data(sim_data_total_yad,
                                            cm_save_file_full_yad,
                                            M = M,
                                            dust_type=yad_dust_type,
                                            tilt=t)

    xl,xu = np.percentile(sims,[5,95],axis=None)
    ave.append(sims.mean())
    med.append(np.median(sims))
    low.append(xl)
    high.append(xu)

low = np.array(low)
high = np.array(high)
ave = np.array(ave)
med = np.array(med)

# trimmed estimation: exclude days with a PM10 reading above the threshold
max_daily_pm10 = 121.8
ave_t,low_t,high_t,med_t = [],[],[],[]
for t in tilts:
    sims,a,a2 = daily_soiling_tilt_all_data(sim_data_total_yad,
                                            cm_save_file_full_yad,
                                            M = M,
                                            dust_type=yad_dust_type,
                                            tilt=t,
                                            max_daily_pm10=max_daily_pm10)

    xl,xu = np.percentile(sims,[5,95],axis=None)
    ave_t.append(sims.mean())
    med_t.append(np.median(sims))
    low_t.append(xl)
    high_t.append(xu)

low_t = np.array(low_t)
high_t = np.array(high_t)
ave_t = np.array(ave_t)
med_t = np.array(med_t)

fig,ax = plt.subplots()
ax.plot(tilts,ave,label='all data',color="blue")
ax.plot(tilts,med,label='median',color="blue",linestyle='--')
ax.fill_between(x=tilts,y1=low,y2=high,color="blue",alpha=0.2)
ax.plot(tilts,ave_t,label='trimmed',color="red")
ax.plot(tilts,med_t,label='median',color="red",linestyle='--')
ax.fill_between(x=tilts,y1=low_t,y2=high_t,color="red",alpha=0.2)
ax.set_xlabel('Tilt (degrees)')
ax.set_ylabel('Percentage points of loss \n (p.p./day)')
ax.set_xlim((0,90))
ax.set_title('Yadnarie')
ax.legend()

# %% Build the assumed heliostat tilt history: reuse Carwarp's real heliostat diurnal tilt pattern
# (requires the Carwarp HELIOSTATS section above to have already run, so imodel.helios.tilt holds the
# real hel_experiment_ tilt history). Rather than tiling it out to cover each Yadnarie experiment's full
# (often much longer, multi-day) duration -- which just produces an uninformative repeating sawtooth --
# each Yadnarie experiment's environmental window is capped to n_tilt_cycles repeats of the Carwarp
# pattern's own length. Increase n_tilt_cycles to look at more repeated days if useful.
n_tilt_cycles = 1

hel_tilt_source = imodel.helios.tilt[0]  # [N_heliostats, N_pattern], from the real Carwarp heliostats
n_pattern = hel_tilt_source.shape[1]
n_target = n_tilt_cycles*n_pattern

yad_windowed_intervals = []
for f in sim_data_total_yad.time.keys():
    t = sim_data_total_yad.time[f]
    n_times_f = min(len(t),n_target)
    yad_windowed_intervals.append(np.array([t.iloc[0],t.iloc[n_times_f-1]]))
yad_windowed_intervals = np.array(yad_windowed_intervals,dtype='datetime64[m]')

sim_data_total_yad,_ = smu.trim_experiment_data( sim_data_total_yad,
                                                None,
                                                yad_windowed_intervals
                                            )

def _tile_tilt_to_length(tilt_pattern,n_times):
    n_pattern = tilt_pattern.shape[1]
    n_reps = int(np.ceil(n_times/n_pattern))
    return np.tile(tilt_pattern,(1,n_reps))[:,:n_times]

imodel_yad_hel = smf.semi_physical(yad_parameter_file)
imodel_yad_hel.update_model_parameters(param_hat_yad)

imodel_constant_yad_hel = smf.constant_mean_deposition(yad_parameter_file)
imodel_constant_yad_hel.update_model_parameters(param_hat_con_yad)

if second_surf:
    inc_ref_factor_yad_hel = 2/np.cos(np.radians(reflectometer_incidence_angle))
else:
    inc_ref_factor_yad_hel = (1+np.sin(np.radians(reflectometer_incidence_angle)))/np.cos(np.radians(reflectometer_incidence_angle))

for mdl in (imodel_yad_hel,imodel_constant_yad_hel):
    mdl.helios.tilt = {}
    mdl.helios.elevation = {}
    mdl.helios.incidence_angle = {}
    mdl.helios.inc_ref_factor = {}
    mdl.helios.acceptance_angles = {}
    for f in sim_data_total_yad.time.keys():
        n_times = len(sim_data_total_yad.time[f])
        tilt_f = _tile_tilt_to_length(hel_tilt_source,n_times)
        mdl.helios.tilt[f] = tilt_f
        mdl.helios.elevation[f] = 90-tilt_f
        mdl.helios.incidence_angle[f] = reflectometer_incidence_angle
        mdl.helios.inc_ref_factor[f] = inc_ref_factor_yad_hel
        mdl.helios.acceptance_angles[f] = [reflectometer_acceptance_angle]*tilt_f.shape[0]

    yad_hel_file_inds = np.arange(len(sim_data_total_yad.time))
    smu.set_extinction_coefficients(mdl,ext_weights_yad,yad_hel_file_inds)

# %% Predict & plot the assumed heliostats' performance under Yadnarie's environment (simulated from clean)
hel_names_yad = reflect_data_total.mirror_names[0]  # names of the real Carwarp heliostats the tilt history was borrowed from

mean_pred_yad_sp,ci_lower_yad_sp,ci_upper_yad_sp = imodel_yad_hel.plot_soiling_factor( sim_data_total_yad,
                            reflectance_data=None,
                            save_path=f"{sp_save_file}_YADNARIE_SCENARIO_{yad_dust_type.replace('.', '-')}{'_daily' if DAILY_AVERAGE else ''}" if save_output else None,
                            figsize=[12,15],
                            mirror_names=hel_names_yad)

mean_pred_yad_cm,ci_lower_yad_cm,ci_upper_yad_cm = imodel_constant_yad_hel.plot_soiling_factor( sim_data_total_yad,
                            reflectance_data=None,
                            save_path=f"{cm_save_file}_YADNARIE_SCENARIO_{yad_dust_type.replace('.', '-')}{'_daily' if DAILY_AVERAGE else ''}" if save_output else None,
                            figsize=[12,15],
                            mirror_names=hel_names_yad)

plt.show()

# %% Tabulate the predictions (mean, and 95% CI bounds) for each borrowed heliostat, experiment and time
def _predictions_to_frame(mean_pred,ci_lower,ci_upper,model_name):
    rows = []
    for f in sim_data_total_yad.time.keys():
        times_f = sim_data_total_yad.time[f].values
        for h,hel_name in enumerate(hel_names_yad):
            rows.append(pd.DataFrame({
                "Model": model_name,
                "Experiment": f,
                "Heliostat": hel_name,
                "Time": times_f,
                "Mean_Prediction": mean_pred[f][h,:],
                "CI_Lower": ci_lower[f][h,:],
                "CI_Upper": ci_upper[f][h,:]
            }))
    return pd.concat(rows,ignore_index=True)

df_pred_yad_sp = _predictions_to_frame(mean_pred_yad_sp,ci_lower_yad_sp,ci_upper_yad_sp,"Semi-Physical")
df_pred_yad_cm = _predictions_to_frame(mean_pred_yad_cm,ci_lower_yad_cm,ci_upper_yad_cm,"Constant-Mean")
df_pred_yad = pd.concat([df_pred_yad_sp,df_pred_yad_cm],ignore_index=True)
print(df_pred_yad)

if save_output:
    df_pred_yad.to_csv(f"{sp_save_file}_YADNARIE_SCENARIO_PREDICTIONS_{yad_dust_type.replace('.', '-')}{'_daily' if DAILY_AVERAGE else ''}.csv", index=False)

# %%

# df_plot_cm =smu.loss_table_from_sim(ref_simulation_cm,sim_data_total) # NEED TO BE FIXED
