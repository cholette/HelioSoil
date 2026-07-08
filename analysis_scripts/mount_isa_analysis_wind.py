"""
Generalized soiling-model fitting/evaluation pipeline.

Configure DATA_DIR / FILE_PREFIX / PARAMETER_FILE_NAME to point at any dataset that
follows the heliosoil analysis-script convention (weather + Reflectance_Average/Sigma/
Tilts sheets per campaign file, one "parameters" workbook), and MODEL_TYPE to select
which soiling model to fit. The pipeline is otherwise identical across all three models:

  - Trains on TRAIN_EXPERIMENTS, using every mirror common to all campaigns (unless
    TRAIN_MIRRORS overrides this). Mirror-normal azimuths (used only by
    "constant_mean_wind") are parsed automatically from the O{orientation} mirror-name
    convention, e.g. "ON_M1_T00" -> North, "OSE_M2_T00" -> Southeast.
  - Evaluates the fitted model on every campaign (the training one, in-sample, plus
    the rest, out-of-sample).
  - Reports MBE/MAE/RMSE/R^2 (out-of-sample) and R^2 (in-sample) on predicted vs.
    measured reflectance.
  - Plots predicted vs. measured soiling factor across all campaigns using
    plot_for_paper (grouped by tilt, one curve per orientation).

Results are written to results/{site_name}/{MODEL_TYPE}/, where site_name is the
last path component of DATA_DIR.
"""

import os

import numpy as np
import pandas as pd
import heliosoil.base_models as smb
import heliosoil.fitting as smf
import heliosoil.utilities as smu
from heliosoil.horizontal_impaction import ConstantMeanWindDeposition
from heliosoil.paper_specific_utilities import (
    regression_performance_stats,
    plot_for_paper,
)

# ============================== Configuration ===============================
main_directory = smu.get_project_root()
DATA_DIR = f"{main_directory}/data/mount_isa/"
FILE_PREFIX = "MountIsa_Data_"
PARAMETER_FILE_NAME = "parameters_mount_isa_experiments.xlsx"

MODEL_TYPE = "constant_mean"  # "constant_mean" | "constant_mean_wind" | "semi_physical"

train_experiments = [0]  # indices (into the sorted file list) used for training
train_mirrors = None  # None -> use every mirror common to all campaigns
dust_type = "TSP"
k_factor = "import"  # None sets equal to 1.0, "import" imports from the file
number_of_measurements = 9.0
reflectometer_incidence_angle = 15  # [deg]
reflectometer_acceptance_angle = 12.5e-3  # [rad]
second_surf = True  # True: second-surface AOI model, False: first-surface
time_to_remove_at_end = 0  # scalar or per-file list, see smu.get_training_data
# ==============================================================================

MODEL_CLASSES = {
    "constant_mean": smf.ConstantMeanDeposition,
    "constant_mean_wind": ConstantMeanWindDeposition,
    "semi_physical": smf.SemiPhysical,
}
PARAM_NAMES = {
    "constant_mean": ["mu_tilde", "sigma_dep"],
    "constant_mean_wind": [
        "mu_tilde",
        "omega_windward",
        "omega_leeward",
        "sigma_dep",
        "sigma_dep_gamma",
    ],
    "semi_physical": ["hrz0", "sigma_dep"],
}
if MODEL_TYPE not in MODEL_CLASSES:
    raise ValueError(f"Unknown MODEL_TYPE {MODEL_TYPE!r}; choose one of {list(MODEL_CLASSES)}")

site_name = os.path.basename(os.path.normpath(DATA_DIR))
results_dir = f"{main_directory}/results/{site_name}/{MODEL_TYPE}"
os.makedirs(results_dir, exist_ok=True)
model_save_file = f"{results_dir}/fitting_results"


def orientation_code(mirror_name):
    """e.g. "ON_M1_T00" -> "N", "OSE_M2_T00" -> "SE" (letters between the leading
    "O" and the first underscore)."""
    return mirror_name.split("_")[0][1:]


def extract(x, ind):
    return [x[ii] for ii in ind]


# %% Get file list, time intervals, and the mirrors common to every campaign
parameter_file = os.path.join(DATA_DIR, PARAMETER_FILE_NAME)
files, training_intervals, mirror_name_list, all_mirrors = smu.get_training_data(
    DATA_DIR, FILE_PREFIX, time_to_remove_at_end=time_to_remove_at_end
)
if train_mirrors is None:
    train_mirrors = all_mirrors
test_experiments = [f for f in range(len(files)) if f not in train_experiments]

files_train = extract(files, train_experiments)
training_intervals_train = extract(training_intervals, train_experiments)

# %% Import & fit on the training campaign(s), using all common mirrors
imodel = MODEL_CLASSES[MODEL_TYPE](parameter_file)
sim_data_train = smb.SimulationInputs(files_train, k_factors=k_factor, dust_type=dust_type)
reflect_data_train = smb.ReflectanceMeasurements(
    files_train,
    sim_data_train.time,
    number_of_measurements=number_of_measurements,
    reflectometer_incidence_angle=reflectometer_incidence_angle,
    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
    import_tilts=True,
    imported_column_names=train_mirrors,
)
sim_data_train, reflect_data_train = smu.trim_experiment_data(
    sim_data_train, reflect_data_train, training_intervals_train
)
sim_data_train, reflect_data_train = smu.trim_experiment_data(
    sim_data_train, reflect_data_train, "reflectance_data"
)

imodel.helios_angles(sim_data_train, reflect_data_train, second_surface=second_surf)

ext_weights = None
if MODEL_TYPE == "semi_physical":
    # The physical model additionally needs Mie extinction weights, computed once
    # from the training dust distribution and re-applied (not recomputed) below.
    imodel.helios.compute_extinction_weights(sim_data_train, imodel.loss_model, verbose=True)
    ext_weights = imodel.helios.extinction_weighting[0].copy()

log_param_hat, log_param_cov = imodel.fit_mle(
    sim_data_train, reflect_data_train, transform_to_original_scale=False
)
# The parameter transform is model-specific (e.g. constant_mean_wind logs mu_tilde/
# sigma_dep/sigma_dep_gamma but leaves omega_windward/omega_leeward linear); the
# script only needs imodel.transform_scale to go from the fitted (possibly
# log-transformed) scale back to each parameter's natural scale.
s = np.sqrt(np.diag(log_param_cov))
param_ci = log_param_hat + 1.96 * s * np.array([[-1], [1]])
lower_ci = imodel.transform_scale(param_ci[0, :])
upper_ci = imodel.transform_scale(param_ci[1, :])
param_hat = imodel.transform_scale(log_param_hat)
param_names = PARAM_NAMES[MODEL_TYPE]
for i, name in enumerate(param_names):
    print(f"{name:16s} = {param_hat[i]:.3e}  [{lower_ci[i]:.3e}, {upper_ci[i]:.3e}]")

imodel.update_model_parameters(param_hat)
imodel.save(
    model_save_file,
    log_p_hat=log_param_hat,
    log_p_hat_cov=log_param_cov,
    training_simulation_data=sim_data_train,
    training_reflectance_data=reflect_data_train,
)

# %% Load all campaigns (same mirror set) and extend tilt/azimuth to all of them
sim_data_total = smb.SimulationInputs(files, k_factors=k_factor, dust_type=dust_type)
reflect_data_total = smb.ReflectanceMeasurements(
    files,
    sim_data_total.time,
    number_of_measurements=number_of_measurements,
    reflectometer_incidence_angle=reflectometer_incidence_angle,
    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
    import_tilts=True,
    imported_column_names=train_mirrors,
)
sim_data_total, reflect_data_total = smu.trim_experiment_data(
    sim_data_total, reflect_data_total, "reflectance_data"
)

imodel.helios_angles(sim_data_total, reflect_data_total, second_surface=second_surf)
if MODEL_TYPE == "semi_physical":
    imodel = smu.set_extinction_coefficients(imodel, ext_weights, np.arange(len(files)))
imodel.predict_soiling_factor(sim_data_total, rho0=reflect_data_total.rho0, verbose=False)

# %% Performance statistics: R^2 in-sample (training campaign), full stats out-of-sample
stats_in_sample = regression_performance_stats(imodel, reflect_data_total, train_experiments)
stats_out_of_sample = regression_performance_stats(imodel, reflect_data_total, test_experiments)

print(f"\nIn-sample  (experiment(s) {train_experiments}, N={stats_in_sample['N']}):")
print(f"  R2   = {stats_in_sample['R2']:.3f}")

print(f"\nOut-of-sample (experiment(s) {test_experiments}, N={stats_out_of_sample['N']}):")
print(f"  MBE  = {stats_out_of_sample['MBE']:.4f}")
print(f"  MAE  = {stats_out_of_sample['MAE']:.4f}")
print(f"  RMSE = {stats_out_of_sample['RMSE']:.4f}")
print(f"  R2   = {stats_out_of_sample['R2']:.3f}")

stats_df = pd.DataFrame(
    [
        {
            "split": "in_sample",
            "experiments": str(train_experiments),
            **stats_in_sample,
        },
        {
            "split": "out_of_sample",
            "experiments": str(test_experiments),
            **stats_out_of_sample,
        },
    ]
).set_index("split")
stats_df.to_csv(f"{results_dir}/performance_stats.csv")

# %% Plot predicted vs. measured soiling factor across all campaigns (grouped by
# tilt, one colored curve per orientation), following mount_isa_analysis.py's paper plot.
# rdat.mirror_names[e] == train_mirrors for every e (same imported_column_names used
# throughout), so `orientation` just needs to be built once, in that same order.
orientation = [[orientation_code(name) for name in train_mirrors] for _ in files]
fig, ax, ref_output = plot_for_paper(
    imodel,
    reflect_data_total,
    sim_data_total,
    train_experiments,
    train_mirrors,
    orientation,
    figsize=(16, 15),
)
fig.savefig(f"{results_dir}/all_campaigns.pdf", bbox_inches="tight")
