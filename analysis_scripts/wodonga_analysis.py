# %% Analysis of Wodonga data
main_directory = ".."
import os
os.sys.path.append(main_directory)

# %% modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soiling_model.base_models as smb
import soiling_model.fitting as smf
import soiling_model.utilities as smu
from matplotlib import rcParams
from paper_specific_utilities import plot_for_paper, daily_soiling_rate, \
                                    fit_quality_plots, summarize_fit_quality
from copy import deepcopy
import scipy.stats as sps

rcParams['figure.figsize'] = (10, 7)

sp_save_file = f"{main_directory}/results/sp_fitting_results_wodonga"
cm_save_file = f"{main_directory}/results/cm_fitting_results_wodonga"
figure_format = ".pdf"
reflectometer_incidence_angle = 15 # angle of incidence of reflectometer
reflectometer_acceptance_angle = 12.5e-3 # half acceptance angle of reflectance measurements
second_surf = True # True if using the second-surface model. Otherwise, use first-surface
d = f"{main_directory}/data/wodonga/"
time_to_remove_at_end = [0,0,0,0,0,0]
train_experiments = [0] # indices for training experiments from 0 to len(files)-1
train_mirrors = ["OE_M1_T00"] # which mirrors within the experiments are used for 
k_factor = None # None sets equal to 1.0, "import" imports from the file
dust_type = "PM10"
use_fitted_dust_distributions = False
epsilon = 1e-3
M = 100 # number of Monte Carlo simulations for deposition rate histogram

# %% Get file list and time intervals. Import training data.
parameter_file = d+"parameters_wodonga_experiments.xlsx"
if use_fitted_dust_distributions:
    d+="fitted/"

files,all_intervals,exp_mirrors,all_mirrors = smu.get_training_data(d,"experiment_",time_to_remove_at_end=time_to_remove_at_end)
orientation = [ [s[1] for s in mirrors] for mirrors in exp_mirrors]

# Feb 2022 (first experiment --- remove last three days after rain started)
all_intervals[0][0] = np.datetime64('2022-02-20T16:20:00')
all_intervals[0][1] = np.datetime64('2022-02-23T17:40:00')

# April 2022 (remove nothing, first/last measurement)
all_intervals[1][0] = np.datetime64('2022-04-21T11:00:00')
all_intervals[1][1] = np.datetime64('2022-04-27T08:30:00')

# Feb 2023 (most recent experiment --- remove very dirty days)
all_intervals[2][0] = np.datetime64('2023-02-09T15:00:00')
all_intervals[2][1] = np.datetime64('2023-02-14T09:45:00')

testing_intervals = all_intervals
        
Nfiles = len(files)
extract = lambda x,ind: [x[ii] for ii in ind]
files_train = extract(files,train_experiments)
training_intervals = extract(all_intervals,train_experiments)
testing_intervals = list(all_intervals)
t = [t for t in train_experiments]
plot_title = "Training: "+str(train_mirrors)+", Exp: "+str(t)

# %% Import & plot training data
imodel = smf.SemiPhysical(parameter_file)
imodel_constant = smf.ConstantMeanDeposition(parameter_file)
sim_data_train = smb.SimulationInputs( files_train,
                                        k_factors=k_factor,
                                        dust_type=dust_type
                                        )
reflect_data_train = smb.ReflectanceMeasurements(  files_train,
                                                    sim_data_train.time,
                                                    number_of_measurements=9.0,
                                                    reflectometer_incidence_angle=reflectometer_incidence_angle,
                                                    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
                                                    import_tilts=True,
                                                    imported_column_names=train_mirrors
                                                    )
#%%
# Trim data and plot
sim_data_train,reflect_data_train = smu.trim_experiment_data(   sim_data_train,
                                                                reflect_data_train,
                                                                training_intervals 
                                                            )
                                                            
sim_data_train,reflect_data_train = smu.trim_experiment_data(   sim_data_train,
                                                                reflect_data_train,
                                                                "reflectance_data" 
                                                            )
for ii,experiment in enumerate(train_experiments):
    fig,ax = smu.plot_experiment_data(sim_data_train,reflect_data_train,ii)
    fig.suptitle(f"Training Data for file {files[experiment]}")

# %% Set mirror angles and get extinction weights
imodel.helios_angles(sim_data_train,reflect_data_train,second_surface=second_surf)
imodel.helios.compute_extinction_weights(sim_data_train,imodel.loss_model,
                                         verbose=True,options={'grid_size_x':1000})
fig_weights,ax_weights = imodel.helios.plot_extinction_weights(sim_data_train,fig_kwargs={'figsize':(5,7)})
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
print(f'hrz0: {hrz0_mle:.2e} [{lower_ci[0]:.2e}, {upper_ci[0]:.2e}]')
print(f'\sigma_dep: {sigma_dep_mle:.2e} [{lower_ci[1]:.2e},{upper_ci[1]:.2e}] [p.p./day]')

imodel.update_model_parameters(param_hat)
imodel.save(sp_save_file,
            log_p_hat=log_param_hat,
            log_p_hat_cov=log_param_cov,
            training_simulation_data=sim_data_train,
            training_reflectance_data=reflect_data_train)

_,_,_ = imodel.plot_soiling_factor( sim_data_train,
                            reflectance_data=reflect_data_train,
                            figsize=(10,10),
                            reflectance_std='measurements',
                            save_path=f"{main_directory}/results/wodonga_semi_physical_training",
                            fig_title="On Training Data (semi-physical)",
                            orientation_strings=orientation    )

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
print(f'mu_tilde: {mu_tilde:.2e} [{lower_ci_con[0]:.2e}, {upper_ci_con[0]:.2e}] [p.p./day]')
print(f'\sigma_dep (constant mean model): {sigma_dep_con:.2e} [{lower_ci_con[1]:.2e},{upper_ci_con[1]:.2e}] [p.p./day]')

imodel_constant.update_model_parameters(param_hat_con)
imodel_constant.save(cm_save_file,
                     log_p_hat=log_param_hat_con,
                     log_p_hat_cov=log_param_cov_con,
                     training_simulation_data=sim_data_train,
                     training_reflectance_data=reflect_data_train)

_,_,_ = imodel_constant.plot_soiling_factor(    sim_data_train,
                                                reflectance_data=reflect_data_train,
                                                figsize=(10,10),
                                                reflectance_std='mean',
                                                save_path=f"{main_directory}/results/wodonga_constant_mean_training",
                                                fig_title="On Training Data",
                                                orientation_strings=orientation    )

# %% Load and trim total data
sim_data_total = smb.SimulationInputs( files,
                                        k_factors=k_factor,
                                        dust_type=dust_type
                                        )

reflect_data_total = smb.ReflectanceMeasurements(  files,
                                                    sim_data_total.time,
                                                    number_of_measurements=9.0,
                                                    reflectometer_incidence_angle=reflectometer_incidence_angle,
                                                    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
                                                    import_tilts=True,
                                                    imported_column_names=None
                                                    )
sim_data_total,reflect_data_total = smu.trim_experiment_data(   sim_data_total,
                                                                reflect_data_total,
                                                                testing_intervals 
                                                            )

sim_data_total,reflect_data_total = smu.trim_experiment_data(   sim_data_total,
                                                                reflect_data_total,
                                                                "reflectance_data" 
                                                            )

sim_data_total,reflect_data_total = smu.trim_experiment_data(   sim_data_total,
                                                                reflect_data_total,
                                                                "simulation_inputs" 
                                                            )

# %% Plot Experiments
for ii,experiment in enumerate(sim_data_total.dt.keys()):
    fig,ax = smu.plot_experiment_data(sim_data_total,reflect_data_total,ii)
    fig.suptitle(f"Testing Data for file {files[experiment]}")

# %% Performance of semi-physical model on total data
imodel.helios_angles(sim_data_total,reflect_data_total,second_surface=second_surf)
file_inds = np.arange(len(files))
imodel = smu.set_extinction_coefficients(imodel,ext_weights,file_inds)

fig,ax,ref_data = plot_for_paper(    imodel,reflect_data_total,
                            sim_data_total,
                            train_experiments,
                            train_mirrors,
                            orientation,
                            legend_shift=(0,0),
                            rows_with_legend=[2],
                            num_legend_cols=4,
                            yticks=(0.93,0.95,0.98,1.0))

if use_fitted_dust_distributions:
    fig.savefig(sp_save_file+figure_format,dpi=300,bbox_inches='tight',pad_inches=0.1)
else:
    fig.savefig(sp_save_file+figure_format,dpi=300,bbox_inches='tight',pad_inches=0.1)

# %% Performance of constant-mean model on total data
if use_fitted_dust_distributions: 
    # set testing dust distributions to be the same as training for constant-mean model
    sim_data_total_constant = deepcopy(sim_data_total)
    obj = sim_data_total_constant.dust
    for a in dir(obj):
        if not a.startswith('__') and  not callable(getattr(obj, a)):
            for f in range(1,len(files)):
                getattr(obj,a)[f] = getattr(obj,a)[0]
else:
    sim_data_total_constant = sim_data_total

            
imodel_constant.helios_angles(sim_data_total_constant,
                              reflect_data_total,
                              second_surface=second_surf)

fig,ax,ref_output = plot_for_paper(    imodel_constant,
                            reflect_data_total,
                            sim_data_total_constant,
                            train_experiments,
                            train_mirrors,
                            orientation,
                            legend_shift=(0,0),
                            rows_with_legend=[2],
                            num_legend_cols=4,
                            yticks=(0.93,0.95,0.98,1.0))

if use_fitted_dust_distributions:
    fig.savefig(cm_save_file+figure_format,dpi=300,bbox_inches='tight',pad_inches=0.1)
else:
    fig.savefig(cm_save_file+figure_format,dpi=300,bbox_inches='tight',pad_inches=0.1)

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
fig.savefig(f"{main_directory}/results/losses_wodonga.pdf",dpi=300,bbox_inches='tight',pad_inches=0)

mirror_idxs = list(range(len(all_mirrors)))
test_experiments = [f for f in list(range(len(files))) if f not in train_experiments]
train_mirror_idx = [m for m in mirror_idxs if all_mirrors[m] in train_mirrors]
test_mirror_idx = [m for m in mirror_idxs if all_mirrors[m] not in train_mirrors]

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
                  test_mirror_idx,
                  ax=ax,
                  min_loss= -0.5,
                  max_loss=9.0,
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
                  min_loss= -0.5,
                  max_loss=9.0,
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
                  min_loss= -0.5,
                  max_loss=9.0,
                  include_fits=False,
                  data_ls='m.',
                  replot=False,
                  data_label="Training",
                  cumulative=True)

ax.set_xlabel("Measured cumulative loss",fontsize=16)
ax.set_ylabel("Predicted cumulative loss",fontsize=16)
fig.savefig(sp_save_file+"_cumulative_fit_quality.pdf",bbox_inches='tight')

fig,ax = plt.subplots(figsize=(6,6))
fit_quality_plots(imodel,
                  reflect_data_total,
                  test_experiments,
                  test_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=2.0,
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
                  max_loss=2.0,
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
                  max_loss=2.0,
                  include_fits=False,
                  data_ls='m.',
                  replot=False,
                  data_label="Training",
                  cumulative=False)


# ax.set_title("Loss change prediction quality assessment (semi-physical)")

# %% Fit Quality plots (constant-mean)
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
                  test_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=9.0,
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
                  max_loss=9.0,
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
                  max_loss=9.0,
                  include_fits=False,
                  data_ls='m.',
                  replot=False,
                  data_label="Training",
                  cumulative=True)

ax.set_xlabel("Measured cumulative loss",fontsize=16)
ax.set_ylabel("Predicted cumulative loss",fontsize=16)
fig.savefig(cm_save_file+"_cumulative_fit_quality.pdf",bbox_inches='tight')

fig,ax = plt.subplots(figsize=(6,6))
fit_quality_plots(imodel_constant,
                  reflect_data_total,
                  test_experiments,
                  test_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=2.0,
                  include_fits=False,
                  data_ls='k*',
                  data_label="Testing data",
                  replot=True,
                  vertical_adjust=-0.1,
                  cumulative=False)

fit_quality_plots(imodel_constant,
                  reflect_data_total,
                  train_experiments,
                  train_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=2.0,
                  include_fits=False,
                  data_ls='m.',
                  replot=False,
                  data_label="Training",
                  cumulative=False)

fit_quality_plots(imodel_constant,
                  reflect_data_total,
                  train_experiments,
                  test_mirror_idx,
                  ax=ax,
                  min_loss= -0.1,
                  max_loss=2.0,
                  include_fits=False,
                  data_ls='g.',
                  data_label="Training (different tilts)",
                  replot=True,
                  vertical_adjust=-0.05,
                  cumulative=False)


ax.set_title("Loss change prediction quality assessment (constant mean)")
ax.set_xlabel(r"Measured $\Delta$loss")
ax.set_ylabel(r"Predicted $\Delta$loss")
# %%
