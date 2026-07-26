"""
Single-campaign soiling-model fit + report workflow.

Fits each selected model on the training campaign(s) and reports, per model:
  - fitted parameters (+ 95% CI) and in-sample/out-of-sample performance stats to CSV,
    plus a plot_for_paper figure (grouped by tilt, one curve per orientation) across
    all campaigns;
  - per-campaign, per-tilt measured-vs-predicted reflectance plots (TILT_ANGLES_TO_PLOT),
    each annotated with that tilt's MAE/RMSE, plus a per-campaign stats CSV.

Each (model_type, wind_components, component_dust_types) combination is a separate "run"
written to its own results/simulate/{site}/{run_name}/{label}/ folder, where label is the
model type or the wind model's derived name (e.g.
"constant-mean_gravitational_normal-wind", or "constant-mean_PM17xgravitational_PM10-PM2.5
xtangential-wind" when the mechanisms are driven by different dust channels). See
model_pipeline.py for the shared fitting kernel and the model-type/dust-channel notes,
and model_selection.py for the cross-validation model-comparison workflow.

Configure the run from the constants below, then run this script directly.
"""

import os
import logging

import pandas as pd
import matplotlib.pyplot as plt
from heliosoil.utilities import configure_logging
from heliosoil.paper_specific_utilities import plot_for_paper, plot_reflectance_by_tilt, regression_performance_stats

import model_pipeline as mp

# ============================== Configuration ===============================
CONFIG = mp.PipelineConfig(
    location="yadnarie",  # "mountisa" | "carwarp" | "yadnarie" | "qut" | "ablrf" | "wodonga"
    train_experiments=[0],  # indices (into the sorted file list) used for training
    train_mirrors=None,  # None -> model-aware default per run; explicit list overrides every run
    dust_type="PM2.5",  # "PM10" or "PM2.5"
    k_factor="import",  # None sets equal to 1.0, "import" imports from the file
    second_surf=True,  # True: second-surface AOI model, False: first-surface
    daily_average=True,  # True: daily-average reflectance, False: all measurements
    verbose=False,
)

RUN_NAME = None  # None -> "run-yy-mm-dd_hh-mm" timestamp; else a label for this run's results folder

MODEL_TYPE = "constant_mean_wind"  # "constant_mean" | "constant_mean_wind" | "semi_physical" | None (run all three)
WIND_COMPONENTS = [
    "turbulent_wind",
    "normal_wind",
    "tangential_wind",
]  # only used when MODEL_TYPE == "constant_mean_wind"; WIND_COMPONENTS=None -> sweep WIND_COMPONENT_COMBOS

# Dust channel driving each wind mechanism. None -> every mechanism uses CONFIG.dust_type
# (one PM type for the whole model). A dict assigns a dust spec per component -- a single
# measure or a difference isolating one size band, e.g.
#   {"gravitational": "PM17", "tangential_wind": "PM10-PM2.5"}
# "sweep" instead fits every assignment of DUST_SPECS to the run's components.
COMPONENT_DUST_TYPES = {"turbulent_wind": "PM2.5", "normal_wind": "PM17-PM2.5", "tangential_wind": "PM2.5"}  #
DUST_SPECS = None  # only used when COMPONENT_DUST_TYPES == "sweep"; None -> auto-detect the site's measures + their differences
WIND_COMPONENT_COMBOS = [
    ["gravitational"],
    ["gravitational", "turbulent_wind"],
    ["gravitational", "normal_wind"],
    ["gravitational", "normal_wind", "tangential_wind"],
    ["gravitational", "tangential_wind"],
    ["gravitational", "impaction_retention"],
    ["gravitational", "tangential_wind", "impaction_retention"],
    ["turbulent_wind", "normal_wind"],
    ["turbulent_wind", "normal_wind", "tangential_wind"],
    ["turbulent_wind", "tangential_wind", "impaction_retention"],
    ["turbulent_wind", "impaction_retention"],
    ["turbulent_wind"],
    ["normal_wind"],
    ["tangential_wind"],
    ["impaction_retention"],
]  # used when MODEL_TYPE == "constant_mean_wind" and WIND_COMPONENTS is None

TILT_ANGLES_TO_PLOT = (0, 30, 60, 90, 180)  # [deg] per-campaign measured-vs-predicted plots
# ==============================================================================


def report_run(cfg: mp.PipelineConfig, data: mp.LoadedData, model_type: str, wind_components: list, component_dust_types, run_name: str) -> None:
    """Fit one (model_type, wind_components, component_dust_types) configuration on the
    training campaign(s) and write its fitted parameters, performance stats, and plots."""
    expression = mp.run_expression(model_type, wind_components, component_dust_types)
    print(f"\n{'=' * 80}\nmodel_type={model_type!r}  {expression}\n{'=' * 80}")

    train_mirrors_run = mp.resolve_training_mirrors(cfg, data.all_mirrors, model_type, wind_components)
    results_dir, _label = mp.results_dir_and_label(cfg, model_type, wind_components, component_dust_types, "simulate", run_name)
    test_experiments = [f for f in range(len(data.files)) if f not in cfg.train_experiments]

    # Primary fit: train on train_experiments, evaluate on all campaigns
    primary = mp.fit_and_evaluate(
        cfg, data, model_type, wind_components, component_dust_types, train_mirrors_run, cfg.train_experiments, test_experiments
    )
    imodel = primary["model"]
    param_names = primary["param_names"]
    param_hat = primary["param_hat"]
    lower_ci = primary["lower_ci"]
    upper_ci = primary["upper_ci"]

    for i, name in enumerate(param_names):
        print(f"{name:16s} = {param_hat[i]:.3e}  [{lower_ci[i]:.3e}, {upper_ci[i]:.3e}]")

    imodel.save(
        f"{results_dir}/fitting_results",
        log_p_hat=primary["log_param_hat"],
        log_p_hat_cov=primary["log_param_cov"],
        training_simulation_data=primary["sim_train"],
        training_reflectance_data=primary["reflect_train"],
    )

    params_df = pd.DataFrame({"parameter": param_names, "value": param_hat, "ci_lower": lower_ci, "ci_upper": upper_ci})
    params_df.to_csv(f"{results_dir}/fitted_parameters.csv", index=False, float_format="%.3g")

    # Performance statistics: R^2 in-sample (training campaign), full stats out-of-sample
    stats_in_sample = primary["stats_in"]
    stats_out_of_sample = primary["stats_out"]

    print(f"\nIn-sample  (experiment(s) {cfg.train_experiments}, N={stats_in_sample['N']}):")
    print(f"  R2   = {stats_in_sample['R2']:.3f}")

    print(f"\nOut-of-sample (experiment(s) {test_experiments}, N={stats_out_of_sample['N']}):")
    print(f"  MBE  = {stats_out_of_sample['MBE']:.4f}")
    print(f"  MAE  = {stats_out_of_sample['MAE']:.4f}")
    print(f"  RMSE = {stats_out_of_sample['RMSE']:.4f}")
    print(f"  R2   = {stats_out_of_sample['R2']:.3f}")

    # model_expression records which mechanisms were fitted and, when they differ, the
    # dust channel driving each ("PM17*gravitational + (PM10-PM2.5)*tangential_wind").
    stats_df = pd.DataFrame(
        [
            {"split": "in_sample", "model_expression": expression, "experiments": str(cfg.train_experiments), **stats_in_sample},
            {"split": "out_of_sample", "model_expression": expression, "experiments": str(test_experiments), **stats_out_of_sample},
        ]
    ).set_index("split")
    stats_df.to_csv(f"{results_dir}/performance_stats.csv", float_format="%.3g")

    # Plot predicted vs. measured soiling factor across all campaigns (grouped by tilt,
    # one colored curve per orientation). reflect_data_total.mirror_names[e] ==
    # all_mirrors for every e, so orientation is built from all_mirrors (train_mirrors_run
    # only labels the training highlight).
    orientation = [[mp.orientation_code(name) for name in data.all_mirrors] for _ in data.files]
    fig, _ax, _ref_output = plot_for_paper(
        imodel, data.reflect_data_total, data.sim_data_total, cfg.train_experiments, train_mirrors_run, orientation, figsize=(16, 15)
    )
    fig.savefig(f"{results_dir}/all_campaigns.pdf", bbox_inches="tight")
    plt.close(fig)

    # Per-campaign, per-tilt measured-vs-predicted reflectance plots (TILT_ANGLES_TO_PLOT),
    # each annotated with that tilt's MAE/RMSE; a CSV with those plus the whole-campaign
    # (all common mirrors) stats is written alongside them.
    for e in range(len(data.files)):
        campaign_dir = f"{results_dir}/campaign_{e + 1}"
        stat_rows = []
        for t in TILT_ANGLES_TO_PLOT:
            fig_t, _, tilt_stats = plot_reflectance_by_tilt(imodel, data.reflect_data_total, data.sim_data_total, e, t, orientation[e])
            if tilt_stats is None:
                continue
            os.makedirs(campaign_dir, exist_ok=True)
            fig_t.savefig(f"{campaign_dir}/reflectance_tilt_{t:02.0f}.pdf", bbox_inches="tight")
            plt.close(fig_t)
            stat_rows.append({"tilt": t, **tilt_stats})

        if stat_rows:
            stat_rows.append({"tilt": "all", **regression_performance_stats(imodel, data.reflect_data_total, [e])})
            pd.DataFrame(stat_rows).to_csv(f"{campaign_dir}/performance_stats.csv", index=False, float_format="%.3g")


def main():
    # Route all HelioSoil output through the "heliosoil" logger: verbose -> INFO
    # (progress messages), else WARNING (only genuine warnings/failures).
    configure_logging(logging.INFO if CONFIG.verbose else logging.WARNING)

    data = mp.load_data(CONFIG)
    dust_specs = DUST_SPECS or (mp.available_dust_specs(CONFIG, files=data.files) if COMPONENT_DUST_TYPES == "sweep" else None)
    if dust_specs is not None:
        print(f"[dust] {CONFIG.location!r} sweeping per-component dust channels: {dust_specs}")

    run_name = mp.resolve_run_dir(CONFIG, "simulate", RUN_NAME)
    runs = mp.build_runs(MODEL_TYPE, WIND_COMPONENTS, WIND_COMPONENT_COMBOS, COMPONENT_DUST_TYPES, dust_specs)
    for model_type, wind_components, component_dust_types in runs:
        report_run(CONFIG, data, model_type, wind_components, component_dust_types, run_name)
    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
