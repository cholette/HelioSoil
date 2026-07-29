"""
Single-campaign soiling-model fit + report workflow.

Fits each selected model on the training campaign(s) and reports, per model:
  - fitted parameters (+ 95% CI) and in-sample/out-of-sample performance stats to CSV,
    plus a plot_for_paper figure (grouped by tilt, one curve per orientation) across
    all campaigns;
  - per-campaign, per-tilt measured-vs-predicted reflectance plots (one per tilt actually
    present in the data), each annotated with that tilt's MAE/RMSE, plus a per-campaign
    stats CSV.

The wind model to fit is given as an expression in the notation the workflows report --
"PM2.5*tangential_wind + (PM10-PM2.5)*turbulent_wind + PMT*gravitational" -- so a model picked
out of a model_selection ranking can be re-fit and reported by pasting its model_expression
back in. A bare mechanism ("gravitational") is driven by the config's own dust_type.

Each (model_type, wind_components, component_dust_types) combination is a separate "run"
written to its own results/simulate/{site}/{run_name}/{label}/ folder, where label is the
model type or the wind model's derived name (e.g. "constant-mean_gravitational_normal-wind",
or a name carrying the per-component dust channels when the mechanisms are driven by different
dust channels).

This module exposes ``run()`` -- it is driven by the unified CLI
(``python -m analysis_scripts.cli fit ...``); see cli.py for the configuration surface,
model_pipeline.py for the shared fitting kernel, and model_selection.py for the cross-
validation model-comparison workflow.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heliosoil.horizontal_impaction import parse_model_expression
from heliosoil.paper_specific_utilities import plot_for_paper, plot_reflectance_by_tilt, regression_performance_stats

from . import model_pipeline as mp


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
    print(f"  R2   = {stats_in_sample['R2']:.3f}  (reflectance)")

    print(f"\nOut-of-sample (experiment(s) {test_experiments}, N={stats_out_of_sample['N']}):")
    print(f"  MBE  = {stats_out_of_sample['MBE']:.4f}  (daily soiling rate)")
    print(f"  MAE  = {stats_out_of_sample['MAE']:.4f}  (daily soiling rate)")
    print(f"  RMSE = {stats_out_of_sample['RMSE']:.4f}  (daily soiling rate)")
    print(f"  R2   = {stats_out_of_sample['R2']:.3f}  (reflectance)")

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

    # Per-campaign, per-tilt measured-vs-predicted reflectance plots, each annotated with that
    # tilt's MAE/RMSE; a CSV with those plus the whole-campaign (all common mirrors) stats is
    # written alongside them. The tilts are the ones actually present in the data
    # (reflect_data_total.tilts, the exact values plot_reflectance_by_tilt matches on), so
    # there is nothing for a user to mis-specify.
    for e in range(len(data.files)):
        campaign_dir = f"{results_dir}/campaign_{e + 1}"
        stat_rows = []
        tilts_present = np.unique(data.reflect_data_total.tilts[e][:, 0])
        for t in tilts_present:
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


def run(
    cfg: mp.PipelineConfig, *, model_type: str | None, model_expression: str | None, run_name: str | None, force: bool, run_metadata: dict
) -> None:
    """Fit + report the model(s) named by `model_type` and `model_expression` for `cfg`'s site.

    `model_expression` describes the constant_mean_wind model to fit, in the notation
    run_expression reports -- "PM2.5*tangential_wind + (PM10-PM2.5)*turbulent_wind" -- with a
    bare mechanism ("gravitational") driven by cfg.dust_type. None fits the library-default
    components on cfg.dust_type. It is ignored by the component-less model types, which
    model_type="all" still fits alongside it.

    Writes each run's outputs under results/simulate/{site}/{run_name}/{label}/, the run
    settings + version to run_config.json, and any suppressed warnings to run.log."""
    components, component_dust_types = parse_model_expression(model_expression) if model_expression else (None, None)
    if components is not None:
        print(f"[model] {mp.run_expression('constant_mean_wind', components, component_dust_types)}")

    data = mp.load_data(cfg)
    run_name = mp.resolve_run_dir(cfg, "simulate", run_name, force=force)
    run_dir = mp.run_dir_path(cfg, "simulate", run_name)
    mp.write_run_metadata(run_dir, run_metadata)
    _log, close_log = mp.open_run_log(run_dir)
    try:
        runs = mp.build_runs(model_type, [components], component_dust_types)
        for model_type_i, wind_components_i, component_dust_types_i in runs:
            report_run(cfg, data, model_type_i, wind_components_i, component_dust_types_i, run_name)
        print("\nAll runs complete.")
    finally:
        close_log()
