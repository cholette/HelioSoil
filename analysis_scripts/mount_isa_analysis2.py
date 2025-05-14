# %% Analysis of Mount Isa data
main_directory = ".."
import os
os.sys.path.append(main_directory)

# %% modules
import numpy as np
import soiling_model.base_models as smb
import soiling_model.fitting as smf
import soiling_model.utilities as smu
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from paper_specific_utilities import plot_for_paper, daily_soiling_rate, \
                                    fit_quality_plots, summarize_fit_quality
import scipy.stats as sps

pad = 0.05
sp_save_file = f"{main_directory}/results/sp_fitting_results_mount_isa"
cm_save_file = f"{main_directory}/results/cm_fitting_results_mount_isa"
reflectometer_incidence_angle = 15 # [deg] angle of incidence of reflectometer
reflectometer_acceptance_angle = 12.5e-3 # [rad] half acceptance angle of reflectance measurements
second_surf = True # True if using the second-surface model. Otherwise, use first-surface
d = f"{main_directory}/data/mount_isa/"
time_to_remove_at_end = [0,0,0,0,0,0]
train_experiments = [0] # indices for training experiments from 0 to len(files)-1
train_mirrors = ["ON_M1_T00"] # which mirrors within the experiments are used for 
k_factor = "import" # None sets equal to 1.0, "import" imports from the file
dust_type = "TSP"

# %% Get file list and time intervals. Import training data.
parameter_file = d+"parameters_mildura_experiments.xlsx"
files,training_intervals,mirror_name_list,all_mirrors = \
    smu.get_training_data(  d,"Mildura_Data_",
                            time_to_remove_at_end=time_to_remove_at_end)
orientation = [ [s[1] for s in mirrors] for mirrors in mirror_name_list]

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
                                                    number_of_measurements=9.0,
                                                    reflectometer_incidence_angle=reflectometer_incidence_angle,
                                                    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
                                                    import_tilts=True,
                                                    imported_column_names=train_mirrors
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
for ii,experiment in enumerate(train_experiments):
    fig,ax = smu.plot_experiment_data(sim_data_train,reflect_data_train,ii)
    fig.suptitle(f"Training Data for file {files[experiment]}")
    # fig,ax = smu.wind_rose(sim_data_train,ii)
    # ax.set_title(f"Wind for file {files[experiment]}")

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
                            save_path=f"{main_directory}/results/mount_isa_sp_training",
                            # fig_title="On Training Data (semi-physical)",
                            orientation_strings=orientation)

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
                                        save_path=f"{main_directory}/results/mount_isa_cm_training",
                                        # fig_title="On Training Data",
                                        orientation_strings=orientation  )

# %% Load and trim total data
sim_data_total = smb.simulation_inputs( files,
                                        k_factors=k_factor,
                                        dust_type=dust_type
                                        )

reflect_data_total = smb.reflectance_measurements(  files,
                                                    sim_data_total.time,
                                                    number_of_measurements=9.0,
                                                    reflectometer_incidence_angle=reflectometer_incidence_angle,
                                                    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
                                                    import_tilts=True,
                                                    imported_column_names=None
                                                    )

# Trim data and plot                                                           
sim_data_total,reflect_data_total = smu.trim_experiment_data(   sim_data_total,
                                                                reflect_data_total,
                                                                "reflectance_data" 
                                                            )

for ii,experiment in enumerate(sim_data_total.dt.keys()):
    fig,ax = smu.plot_experiment_data(sim_data_total,reflect_data_total,ii)
    fig.suptitle(f"Testing Data for file {files[experiment]}")
    fig,ax = smu.wind_rose(sim_data_total,ii)
    ax.set_title(f"Wind for file {files[experiment]}")

# %% Performance of semi-physical model on total data
imodel.helios_angles(sim_data_total,reflect_data_total,second_surface=second_surf)
file_inds = np.arange(len(files))
imodel = smu.set_extinction_coefficients(imodel,ext_weights,file_inds)

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
fig.savefig(f"{main_directory}/results/losses_mount_isa.pdf",dpi=300,bbox_inches='tight',pad_inches=0)

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
fig.savefig(f"{main_directory}/results/highest_losses_mount_isa.pdf",dpi=300,bbox_inches='tight',pad_inches=0)

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