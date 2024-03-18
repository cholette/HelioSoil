# %% Analysis of Mount Isa data
main_directory = ".."
import os
os.sys.path.append(main_directory)

# %% modules
import numpy as np
import pandas as pd
import soiling_model.base_models as smb
import soiling_model.fitting as smf
import soiling_model.utilities as smu
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from paper_specific_utilities import plot_for_paper, daily_soiling_rate, \
                                    fit_quality_plots, summarize_fit_quality
import scipy.stats as sps

pad = 0.05
sp_save_file = f"{main_directory}/results/sp_fitting_results_port_augusta"
cm_save_file = f"{main_directory}/results/cm_fitting_results_port_augusta"
reflectometer_incidence_angle = 15 # [deg] angle of incidence of reflectometer
reflectometer_acceptance_angle = 12.5e-3 # [rad] half acceptance angle of reflectance measurements
second_surf = True # True if using the second-surface model. Otherwise, use first-surface
d = f"{main_directory}/data/port_augusta/"
time_to_remove_at_end = [0,0,0,0,0,0]
train_experiments = [0] # indices for training experiments from 0 to len(files)-1
train_mirrors = ["OSE_M2_T00"]#,"ONW_M5_T00"] # which mirrors within the experiments are used for 
# train_mirrors = ["OSE_M3_T30"]#,"ONW_M5_T00"] # which mirrors within the experiments are used for 
k_factor = "import" # None sets equal to 1.0, "import" imports from the file
dust_type = "TSP"

# %% Get file list and time intervals. Import training data.
parameter_file = d+"parameters_port_augusta_experiments.xlsx"
files,training_intervals,mirror_name_list,all_mirrors = \
    smu.get_training_data(  d,"PortAugusta_Data_",
                            time_to_remove_at_end=time_to_remove_at_end)
training_intervals[0][1] -= np.datetime64(36,'h')

orientation = [ [s[1]+s[2] for s in mirrors] for mirrors in mirror_name_list]

Nfiles = len(files)
extract = lambda x,ind: [x[ii] for ii in ind]
files_train = extract(files,train_experiments)
training_intervals = extract(training_intervals,train_experiments)
t = [t for t in train_experiments]
plot_title = "Training: "+str(train_mirrors)+", Exp: "+str(t)

# %% Import & plot training data
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
# Trim data and plot
sim_data_train,reflect_data_train = smu.trim_experiment_data(   sim_data_train,
                                                                reflect_data_train,
                                                                training_intervals 
                                                            )
                                                            
sim_data_train,reflect_data_train = smu.trim_experiment_data(   sim_data_train,
                                                                reflect_data_train,
                                                                "reflectance_data" 
                                                            )

def plot_experiment_PA(simulation_inputs,reflectance_data,experiment_index,figsize=(7,12)):
    sim_data = simulation_inputs
    reflect_data = reflectance_data
    f = experiment_index

    fig,ax = plt.subplots(nrows=3,sharex=True,figsize=figsize)
    # fmt = r"${0:s}^\circ$"
    fmt = "${0:s}$"
    ave = reflect_data.average[f]
    t = reflect_data.times[f]
    std = reflect_data.sigma[f]
    # names = ["M"+str(ii+1) for ii in range(ave.shape[1])]
    names = ["SE1",	"SE2",	"SE3",	"SE4",	"SE5",	"NW1",	"NW2",	"NW3",	"NW4",	"NW5"]

    for ii in range(ave.shape[1]):
        ax[0].errorbar(t,ave[:,ii],yerr=1.96*std[:,ii],label=fmt.format(names[ii]),marker='o',capsize=4.0)

    ax[0].grid(True) 
    label_str = r"Reflectance at {0:.1f} $^{{\circ}}$".format(reflect_data.reflectometer_incidence_angle[f]) 
    ax[0].set_ylabel(label_str)
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    # ax[0].set_ylim(0.86, 0.97)

    ax[1].plot(sim_data.time[f],sim_data.dust_concentration[f],color='brown',label="Measurements")
    ax[1].axhline(y=sim_data.dust_concentration[f].mean(),color='brown',ls='--',label = "Average")
    label_str = r'{0:s} [$\mu g\,/\,m^3$]'.format(sim_data.dust_type[0])
    ax[1].set_ylabel(label_str,color='brown')
    ax[1].tick_params(axis='y', labelcolor='brown')
    ax[1].grid(True)
    ax[1].legend()
    title_str = dust_type + r" (mean = {0:.2f} $\mu g$/$m^3$)" 
    ax[1].set_title(title_str.format(sim_data.dust_concentration[f].mean()),fontsize=10)
    # ax[1].set_ylim(0,70)

    # # Rain intensity, if available
    # if len(sim_data.rain_intensity)>0: # rain intensity is not an empty dict
    #     ax[2].plot(sim_data.time[f],sim_data.rain_intensity[f])
    # else:
    #     rain_nan = np.nan*np.ones(sim_data.time[f].shape)
    #     ax[2].plot(sim_data.time[f],rain_nan)
    
    # ax[2].set_ylabel(r'Rain [mm/hour]',color='blue')
    # ax[2].tick_params(axis='y', labelcolor='blue')
    # YL = ax[2].get_ylim()
    # ax[2].set_ylim((0,YL[1]))
    # ax[2].grid(True)

    ax[2].plot(sim_data.time[f],sim_data.wind_speed[f],color='green',label="Measurements")
    ax[2].axhline(y=sim_data.wind_speed[f].mean(),color='green',ls='--',label = "Average")
    label_str = r'Wind Speed [$m\,/\,s$]'
    ax[2].set_ylabel(label_str,color='green')
    ax[2].set_xlabel('Date')
    ax[2].tick_params(axis='y', labelcolor='green')
    ax[2].grid(True)
    ax[2].legend()
    title_str = "Wind Speed (mean = {0:.2f} m/s)".format(sim_data.wind_speed[f].mean())
    ax[2].set_title(title_str,fontsize=10)
    # ax[2].set_ylim(0,8)

    # if len(sim_data.relative_humidity)>0: 
    #     ax[4].plot(sim_data.time[f],sim_data.relative_humidity[f],color='black',label="measurements")
    #     ax[4].axhline(y=sim_data.relative_humidity[f].mean(),color='black',ls='--',label = "Average")
    # else:
    #     rain_nan = np.nan*np.ones(sim_data.time[f].shape)
    #     ax[4].plot(sim_data.time[f],rain_nan)
    
    # label_str = r'Relative Humidity [%]'
    # ax[4].set_ylabel(label_str,color='black')
    # ax[4].set_xlabel('Date')
    # ax[4].tick_params(axis='y', labelcolor='black')
    # ax[4].grid(True)
    # ax[4].legend()
    
    if len(sim_data.wind_direction)>0: 
        figwr,axwr = smu.wind_rose(sim_data,f)
        figwr.tight_layout()

    fig.autofmt_xdate()
    fig.tight_layout()

    return fig,ax
    


for ii,experiment in enumerate(train_experiments):
    fig,ax = plot_experiment_PA(sim_data_train,reflect_data_train,ii)
    # fig.suptitle(f"Training Data for file {files[experiment]}")
    # fig,ax = smu.wind_rose(sim_data_train,ii)
    # ax.set_title(f"Wind for file {files[experiment]}")

# %% Load and trim total data
sim_data_total = smb.simulation_inputs( files,
                                        k_factors=k_factor,
                                        dust_type=dust_type
                                        )

reflect_data_total = smb.reflectance_measurements(  files,
                                                    sim_data_total.time,
                                                    number_of_measurements=6.0,
                                                    reflectometer_incidence_angle=reflectometer_incidence_angle,
                                                    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
                                                    import_tilts=True,
                                                    column_names_to_import=None
                                                    )

# %% Trim data and plot                                                           
sim_data_total,reflect_data_total = smu.trim_experiment_data(   sim_data_total,
                                                                reflect_data_total,
                                                                "reflectance_data" 
                                                            )

for ii,experiment in enumerate(sim_data_total.dt.keys()):
    fig,ax = plot_experiment_PA(sim_data_total,reflect_data_total,ii)
    # fig.suptitle(f"Testing Data for file {files[experiment]}")
    fig,ax = smu.wind_rose(sim_data_total,ii)
    ax.set_title(f"Wind for file {files[experiment]}")

# %% Compute pair-average reflectance loss after clean state to avoid morning-evening recoveries

sim_data_total,reflect_data_total = smu.average_experiment_data(sim_data_total,reflect_data_total)

for ii in range(len(reflect_data_total.times)):
    if len(reflect_data_total.average[ii])>2:   # if less or equal to 2, the resulting array would only include the starting point
        fig,ax = plot_experiment_PA(sim_data_total,reflect_data_total,ii)


# %% Plot reflectance losses in each interval
for m,mir in enumerate(train_mirrors):
    for exp in sim_data_total.time.keys():
        diff_array_times = -np.diff(reflect_data_total.times[exp])
        diff_days = -diff_array_times.astype('timedelta64[s]').astype('int')/3600/24
        idx_mir = reflect_data_total.mirror_names[exp].index(mir)
        diff_ref = -np.diff(reflect_data_total.average[exp][:,idx_mir],axis=0)
        diff_rates = diff_ref*100/diff_days  # contain soiling rates for mir in training_mirrors

        df_dust = pd.DataFrame({'Time': sim_data_total.time[exp],
                                'Value':sim_data_total.dust_concentration[exp]})
        retiming_vector = pd.to_datetime(reflect_data_total.times[exp])
        df_dust['Interval'] = pd.cut(df_dust['Time'], bins=retiming_vector, right=False, labels=False)
        df_dust_retime = pd.DataFrame({
            'Time': retiming_vector[1:],
            'Mean_TSP': df_dust.groupby('Interval')['Value'].mean().values,
            'Soiling_Rate': diff_rates
            })
        df_sorted = df_dust_retime.sort_values(by='Mean_TSP')
        print(df_dust_retime)
        print(df_sorted)
        plt.plot(df_sorted['Mean_TSP'], df_sorted['Soiling_Rate'], marker='o', linestyle='-', color='b')
        # plt.hist(df_sorted['Soiling_Rate'], bins='auto', edgecolor='black')
        plt.xlabel('Mean_TSP, µg/m3')
        plt.ylabel('Soiling_Rate, %/day')
        plt.title(f"Soiling Rates for {files[exp]}")
        plt.show()

# %% Set mirror angles and get extinction weights
imodel.helios_angles(sim_data_train,reflect_data_train,second_surface=second_surf)
imodel.helios.compute_extinction_weights(sim_data_train,imodel.loss_model,verbose=True)
imodel.helios.plot_extinction_weights(sim_data_train,fig_kwargs={})
ext_weights = imodel.helios.extinction_weighting[0].copy()

imodel_constant.helios_angles(sim_data_train,reflect_data_train,second_surface=second_surf)
file_inds = np.arange(len(files_train))
imodel_constant = smu.set_extinction_coefficients(imodel_constant,ext_weights,file_inds)

# %% Fit semi-physical model & plot on training data
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
imodel.save(sp_save_file,
            log_p_hat=log_param_hat,
            log_p_hat_cov=log_param_cov,
            training_simulation_data=sim_data_train,
            training_reflectance_data=reflect_data_train)

_,_,_ = imodel.plot_soiling_factor( sim_data_train,
                            reflectance_data=reflect_data_train,
                            reflectance_std='mean',
                            save_path=f"{main_directory}/results/port_augusta_sp_training",
                            # fig_title="On Training Data (semi-physical)",
                            orientation_strings=orientation,
                            figsize=[12,6])

# %% Fit constant mean model & plot on training data
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
imodel_constant.save(cm_save_file,
                     log_p_hat=log_param_hat_con,
                     log_p_hat_cov=log_param_cov_con,
                     training_simulation_data=sim_data_train,
                     training_reflectance_data=reflect_data_train)

_,_,_ = imodel_constant.plot_soiling_factor(    sim_data_train,
                                        reflectance_data=reflect_data_train,
                                        reflectance_std='mean',
                                        save_path=f"{main_directory}/results/port_augusta_cm_training",
                                        # fig_title="On Training Data",
                                        orientation_strings=orientation,
                                        figsize = [12,8]  )


# %% Performance of semi-physical model on total data
imodel.helios_angles(sim_data_total,reflect_data_total,second_surface=second_surf)
file_inds = np.arange(len(reflect_data_total.file_name))
imodel = smu.set_extinction_coefficients(imodel,ext_weights,file_inds)

# %%
fig,ax = plot_for_paper(    imodel,reflect_data_total,
                            sim_data_total,
                            train_experiments,
                            train_mirrors,
                            orientation,
                            legend_shift=(0.04,0),
                            yticks=(0.88,0.92,0.96,1.0))

fig.savefig(sp_save_file+".pdf",bbox_inches='tight')

# %% Performance of constant-mean model on total data
imodel_constant.helios_angles(sim_data_total,reflect_data_total,second_surface=second_surf)

fig,ax = plot_for_paper(    imodel_constant,
                            reflect_data_total,
                            sim_data_total,
                            train_experiments,
                            train_mirrors,
                            orientation,
                            legend_shift=(0.04,0),
                            yticks=(0.92,0.94,0.96,0.98,1.0))

fig.savefig(cm_save_file+".pdf",bbox_inches='tight')

# %% High, Medium, Low daily loss distributions from total data
pers = [5,50,95.0,100]
labels = ['Low','Medium','High','Maximum']
colors = ['blue','green','purple','black']
fsz=16

sims,a,a2 = daily_soiling_rate(  sim_data_total,
                                cm_save_file,
                                M = 100000,
                                percents=pers,
                                dust_type=dust_type)
# xL,xU = np.percentile(sims,[0.1,99.9])
xL,xU = -0.25,3.0
lg = np.linspace(xL,xU,1000)
inc_factor = imodel.helios.inc_ref_factor[0]

fig,ax = plt.subplots()
for ii in range(sims.shape[1]):
    ax.hist(sims[:,ii],250,density=True,
            alpha=0.5,color=colors[ii],
            label=labels[ii])

    loc = inc_factor*mu_tilde*a[ii]
    s2 = (inc_factor*sigma_dep_con)**2 * a2[ii]
    dist = sps.norm(loc=loc*100,scale=np.sqrt(s2)*100)
    ax.plot(lg,dist.pdf(lg),color=colors[ii])
    print(f"Loss for {labels[ii]} scenario: {loc*100:.2f} +/- {1.96*100*np.sqrt(s2):.2f}")

ax.set_xlim((xL,xU))
ax.set_ylabel('Probability Density',fontsize=fsz+2)
ax.set_xlabel('Loss (percentage points)',fontsize=fsz+2)
ax.legend(fontsize=fsz)

fig.set_size_inches(5,4)
fig.savefig(f"{main_directory}/results/losses_port_augusta.pdf",dpi=300,bbox_inches='tight',pad_inches=0)

# %% Highest only

xL,xU = np.percentile(sims,[0.1,99.99])
lg = np.linspace(xL,xU,1000)

fig,ax = plt.subplots()
ii = sims.shape[1]-1
ax.hist(sims[:,ii],250,density=True,
        alpha=0.5,color=colors[ii],
        label=labels[ii])

loc = inc_factor*mu_tilde*a[ii]
s2 = (inc_factor*sigma_dep_con)**2 * a2[ii]
dist = sps.norm(loc=loc*100,scale=np.sqrt(s2)*100)
ax.plot(lg,dist.pdf(lg),color=colors[ii])

ax.set_xlim((xL,xU))
ax.set_ylabel('Probability Density',fontsize=fsz+2)
ax.set_xlabel('Loss (percentage points)',fontsize=fsz+2)
# ax.legend(fontsize=fsz)

fig.set_size_inches(5,4)
fig.savefig(f"{main_directory}/results/highest_losses_port_augusta.pdf",dpi=300,bbox_inches='tight',pad_inches=0)

# %% Fit quality plots (semi-physical)
mirror_idxs = list(range(len(all_mirrors)))
test_experiments = [f for f in list(range(len(files))) if f not in train_experiments]
train_mirror_idx = [m for m in mirror_idxs if all_mirrors[m] in train_mirrors]
test_mirror_idx = [m for m in mirror_idxs if all_mirrors[m] not in train_mirrors]

fig,ax = summarize_fit_quality( imodel,
                                reflect_data_total,
                                train_experiments,
                                train_mirror_idx,
                                test_mirror_idx,test_experiments,
                                min_loss=-0.2,
                                max_loss=3.25,
                                save_file=sp_save_file,
                                figsize=(10,10)
                                )
for a in ax:
    a.set_xticks([0,1,2,3])
    a.set_yticks([0,1,2,3])


fig,ax = plt.subplots(figsize=(6,6))
fit_quality_plots(imodel,
                  reflect_data_total,
                  test_experiments,
                  test_mirror_idx+train_mirror_idx,
                  ax=ax,
                  min_loss= -1.0,
                  max_loss=10.0,
                  include_fits=False,
                  data_ls='k*',
                  data_label="Testing data",
                  replot=True,
                  vertical_adjust=-0.1,
                  cumulative=True)

fit_quality_plots(imodel,
                  reflect_data_total,
                  train_experiments,
                  test_mirror_idx,
                  ax=ax,
                  min_loss= -1.0,
                  max_loss=10.0,
                  include_fits=False,
                  data_ls='g.',
                  data_label="Training (different tilts)",
                  replot=False,
                  vertical_adjust=-0.05,
                  cumulative=True)

fit_quality_plots(imodel,
                  reflect_data_total,
                  train_experiments,
                  train_mirror_idx,
                  ax=ax,
                  min_loss= -1.0,
                  max_loss=10.0,
                  include_fits=False,
                  data_ls='m.',
                  replot=False,
                  data_label="Training",
                  cumulative=True)

ax.set_xlabel("Measured cumulative loss",fontsize=16)
ax.set_ylabel("Predicted cumulative loss",fontsize=16)
ax.legend(loc='lower right')
fig.savefig(sp_save_file+"_cumulative_fit_quality.pdf",bbox_inches='tight')

fig,ax = plt.subplots(figsize=(6,6))
fit_quality_plots(imodel,
                  reflect_data_total,
                  test_experiments,
                  test_mirror_idx+train_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=10.0,
                  include_fits=False,
                  data_ls='k*',
                  data_label="Testing data",
                  replot=True,
                  vertical_adjust=-0.1,
                  cumulative=False)

fit_quality_plots(imodel,
                  reflect_data_total,
                  train_experiments,
                  test_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=10.0,
                  include_fits=False,
                  data_ls='g.',
                  data_label="Training (different tilts)",
                  replot=False,
                  vertical_adjust=-0.05,
                  cumulative=False)

fit_quality_plots(imodel,
                  reflect_data_total,
                  train_experiments,
                  train_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=10.0,
                  include_fits=False,
                  data_ls='m.',
                  replot=False,
                  data_label="Training",
                  cumulative=False)


ax.set_xlabel(r"Measured $\Delta$ loss",fontsize=16)
ax.set_ylabel(r"Predicted $\Delta$ loss",fontsize=16)
ax.legend(loc='lower right')
ax.set_title("Loss change prediction quality assessment (semi-physical)")

# %% Fit Quality plots (constant-mean)

mirror_idxs = list(range(len(all_mirrors)))
test_experiments = [f for f in list(range(len(files))) if f not in train_experiments]
train_mirror_idx = [m for m in mirror_idxs if all_mirrors[m] in train_mirrors]
test_mirror_idx = [m for m in mirror_idxs if all_mirrors[m] not in train_mirrors]

fig,ax = summarize_fit_quality( imodel_constant,
                                reflect_data_total,
                                train_experiments,
                                train_mirror_idx,
                                test_mirror_idx,test_experiments,
                                min_loss=-0.2,
                                max_loss=3.25,
                                save_file=sp_save_file,
                                figsize=(10,10)
                                )
for a in ax:
    a.set_xticks([0,1,2,3])
    a.set_yticks([0,1,2,3])


fig,ax = plt.subplots(figsize=(6,6))

fit_quality_plots(imodel_constant,
                  reflect_data_total,
                  test_experiments,
                  test_mirror_idx+train_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=8.0,
                  include_fits=False,
                  data_ls='k*',
                  data_label="Testing data",
                  replot=True,
                  vertical_adjust=-0.1,
                  cumulative=True)

fit_quality_plots(imodel_constant,
                  reflect_data_total,
                  train_experiments,
                  test_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=8.0,
                  include_fits=False,
                  data_ls='g.',
                  data_label="Training (different tilts)",
                  replot=False,
                  vertical_adjust=-0.05,
                  cumulative=True)

fit_quality_plots(imodel_constant,
                  reflect_data_total,
                  train_experiments,
                  train_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=8.0,
                  include_fits=False,
                  data_ls='m.',
                  replot=False,
                  data_label="Training",
                  cumulative=True)

ax.set_xlabel("Measured cumulative loss",fontsize=16)
ax.set_ylabel("Predicted cumulative loss",fontsize=16)
ax.legend(loc='lower right')
fig.savefig(cm_save_file+"_cumulative_fit_quality.pdf",bbox_inches='tight')

fig,ax = plt.subplots(figsize=(6,6))
fit_quality_plots(imodel_constant,
                  reflect_data_total,
                  test_experiments,
                  test_mirror_idx+train_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=3.5,
                  include_fits=False,
                  data_ls='k*',
                  data_label="Testing data",
                  replot=True,
                  vertical_adjust=-0.1,
                  cumulative=False)

fit_quality_plots(imodel_constant,
                  reflect_data_total,
                  train_experiments,
                  test_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=3.5,
                  include_fits=False,
                  data_ls='g.',
                  data_label="Training (different tilts)",
                  replot=False,
                  vertical_adjust=-0.05,
                  cumulative=False)

fit_quality_plots(imodel_constant,
                  reflect_data_total,
                  train_experiments,
                  train_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=3.5,
                  include_fits=False,
                  data_ls='m.',
                  replot=False,
                  data_label="Training",
                  cumulative=False)




ax.set_xlabel(r"Measured $\Delta$loss",fontsize=16)
ax.set_ylabel(r"Predicted $\Delta$loss",fontsize=16)
# %%
