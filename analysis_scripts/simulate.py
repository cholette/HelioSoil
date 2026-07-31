"""
Single-campaign soiling-model fit + report workflow.

Fits each selected model on the training campaign(s) and reports, per model:
  - fitted parameters (+ 95% CI) and in-sample/out-of-sample performance stats to CSV,
    plus a plot_for_paper figure (grouped by tilt, one curve per orientation) across
    all campaigns;
  - per-campaign, per-tilt measured-vs-predicted reflectance plots (one per tilt actually
    present in the data), each annotated with that tilt's MAE/RMSE, plus a per-campaign
    stats CSV;
  - a boxplot of the absolute daily soiling-rate error grouped by tilt, paired in-sample vs
    out-of-sample, showing how each reported MAE is distributed rather than only its value.

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
(``python -m analysis_scripts.cli simulate ...``); see cli.py for the configuration surface,
model_pipeline.py for the shared fitting kernel, and model_selection.py for the cross-
validation model-comparison workflow.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heliosoil.horizontal_impaction import parse_model_expression
from heliosoil.paper_specific_utilities import daily_rate_residuals, plot_for_paper, plot_reflectance_by_tilt, regression_performance_stats

from . import model_pipeline as mp
from .campaign_summary import FIGURE_RC, site_display_name

WORKFLOW = "simulate"


def _abs_rate_error_by_tilt(model, rdat, experiments) -> dict:
    """|daily-rate residual| for the mirrors at each tilt, pooled over `experiments`, in
    **percentage points of soiling factor per day**.

    daily_rate_residuals works in soiling-factor units, as performance_stats.csv reports them;
    the x100 here matches the p.p./day the measurement-side figures and tables use
    (experiment_soiling.soiling_rate_pp_day), which is the unit a reader of the report expects.

    Grouped per experiment and concatenated rather than once over a shared tilt vector, so a
    campaign that lacks a tilt simply contributes nothing to it. Mirrors with a NaN tilt are
    dropped: they cannot be placed on the axis, and np.unique would give each its own group."""
    groups: dict = {}
    for e in experiments:
        tilts = rdat.tilts[e][:, 0]
        for tilt in np.unique(tilts[np.isfinite(tilts)]):
            (idx,) = np.where(tilts == tilt)
            residuals = daily_rate_residuals(model, rdat, [e], mirrors=idx)
            if residuals.size:
                groups.setdefault(float(tilt), []).append(np.abs(residuals) * 100.0)
    return {tilt: np.concatenate(parts) for tilt, parts in sorted(groups.items())}


def plot_rate_error_by_tilt(model, rdat, train_experiments, test_experiments, site) -> plt.Figure | None:
    """Distribution of the absolute daily soiling-rate error at each tilt, in-sample vs out-of-sample.

    performance_stats.csv reports one MAE per split and one per tilt, but not how that error is
    spread -- whether a tilt's MAE comes from a uniformly mediocre fit or from a few bad intervals.
    Each box pools every |predicted - measured daily rate| over the mirrors at that tilt and the
    measurement intervals of the split's campaigns; its mean marker is that group's MAE.

    Tilt is a categorical axis, unlike experiment_soiling.plot_soiling_rate_vs_tilt's numeric one:
    here the claim is a per-group comparison, not a trend, and carwarp/yadnarie's 180 deg mirror
    would otherwise strand the 0/30/60 cluster at one end. Fliers are drawn rather than hidden
    because MAE is mean-based and it is the tail that drives it.

    Returns None when neither split yields a residual at any tilt, so the caller writes no file."""
    in_sample = _abs_rate_error_by_tilt(model, rdat, train_experiments)
    out_of_sample = _abs_rate_error_by_tilt(model, rdat, test_experiments)
    tilts = sorted(set(in_sample) | set(out_of_sample))
    if not tilts:
        return None

    # Both boxes are white-filled and separated by edge colour and mean-marker shape: a solid fill
    # swallows the median line, which is the one mark a reader compares across the pair.
    splits = [("In-sample", in_sample, -0.18, "C0", "o"), ("Out-of-sample", out_of_sample, 0.18, "C3", "D")]

    with plt.rc_context(FIGURE_RC):
        fig, ax = plt.subplots(figsize=(max(5.0, 1.1 * len(tilts) + 2.5), 4.2))
        handles = []
        for label, groups, offset, edge, mean_marker in splits:
            present = [(i, tilt) for i, tilt in enumerate(tilts) if tilt in groups]
            if not present:
                continue
            data = [groups[tilt] for _, tilt in present]
            drawn = ax.boxplot(
                data,
                positions=[i + offset for i, _ in present],
                widths=0.3,
                whis=(5, 95),
                showmeans=True,
                patch_artist=True,
                boxprops={"facecolor": "white", "edgecolor": edge, "linewidth": 1.0},
                medianprops={"color": edge, "linewidth": 1.4},
                whiskerprops={"color": edge, "linewidth": 1.0},
                capprops={"color": edge, "linewidth": 1.0},
                flierprops={"marker": ".", "markersize": 2.5, "markerfacecolor": edge, "markeredgecolor": "none", "alpha": 0.35},
                meanprops={"marker": mean_marker, "markersize": 4.5, "markerfacecolor": edge, "markeredgecolor": edge},
            )
            handles.append((drawn["boxes"][0], label))
            # A box over one mirror's handful of intervals is not a distribution; saying n makes that
            # plain. In its own row above the axes, not at the floor, where the low-error boxes sit.
            for (i, _tilt), values in zip(present, data):
                ax.annotate(
                    str(values.size),
                    (i + offset, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=edge,
                    annotation_clip=False,
                )

        ax.set_xticks(range(len(tilts)))
        ax.set_xticklabels([f"{tilt:.0f}" for tilt in tilts])
        ax.set_xlim(-0.5, len(tilts) - 0.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Tilt [°]")
        ax.set_ylabel("|Daily soiling-rate error| [p.p./day]")
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend([h for h, _ in handles], [lbl for _, lbl in handles], ncol=2)
        # No caption: this figure is for a report, where the method note belongs in the prose
        # caption rather than inside the axes. Headroom for the n row instead.
        ax.set_title(" ", pad=10)
        fig.suptitle(f"{site} — Daily soiling-rate error by tilt")
        fig.tight_layout()
    return fig


def rate_error_by_tilt_table(model, rdat, train_experiments, test_experiments) -> pd.DataFrame:
    """The numbers behind plot_rate_error_by_tilt, one row per (split, tilt).

    Named _pp_day because these carry the figure's x100 and so are 100x performance_stats.csv's
    MAE, which is in soiling-factor per day -- the unit has to be on the column or the two tables
    look like they disagree about the same model."""
    rows = []
    for split, experiments in (("in_sample", train_experiments), ("out_of_sample", test_experiments)):
        for tilt, values in _abs_rate_error_by_tilt(model, rdat, experiments).items():
            rows.append(
                {
                    "split": split,
                    "tilt": tilt,
                    "MAE_pp_day": values.mean(),
                    "median_pp_day": np.median(values),
                    "p95_pp_day": np.percentile(values, 95),
                    "N": values.size,
                }
            )
    return pd.DataFrame(rows)


def report_run(cfg: mp.PipelineConfig, data: mp.LoadedData, model_type: str, wind_components: list, component_dust_types, run_name: str) -> None:
    """Fit one (model_type, wind_components, component_dust_types) configuration on the
    training campaign(s) and write its fitted parameters, performance stats, and plots."""
    expression = mp.run_expression(model_type, wind_components, component_dust_types)
    print(f"\n{'=' * 80}\nmodel_type={model_type!r}  {expression}\n{'=' * 80}")

    train_mirrors_run = mp.resolve_training_mirrors(cfg, data.all_mirrors, model_type, wind_components)
    results_dir, _label = mp.results_dir_and_label(cfg, model_type, wind_components, component_dust_types, WORKFLOW, run_name)
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
        imodel,
        data.reflect_data_total,
        data.sim_data_total,
        cfg.train_experiments,
        train_mirrors_run,
        orientation,
        figsize=(16, 15),
        num_legend_cols=7,
        auto_yticks=True,
        plot_rh=False,
    )
    fig.savefig(f"{results_dir}/all_campaigns.pdf", bbox_inches="tight")
    plt.close(fig)

    # How the daily soiling-rate error is distributed within each tilt, in-sample vs out-of-sample.
    # helios.soiling_factor is already populated over every campaign by fit_and_evaluate, so this
    # re-reads the same prediction the stats above were computed from.
    site = site_display_name(cfg.location)
    fig_err = plot_rate_error_by_tilt(imodel, data.reflect_data_total, cfg.train_experiments, test_experiments, site)
    if fig_err is not None:
        fig_err.savefig(f"{results_dir}/rate_error_by_tilt.pdf", bbox_inches="tight")
        plt.close(fig_err)
        rate_error_by_tilt_table(imodel, data.reflect_data_total, cfg.train_experiments, test_experiments).to_csv(
            f"{results_dir}/rate_error_by_tilt.csv", index=False, float_format="%.3g"
        )

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
    run_name = mp.resolve_run_dir(cfg, WORKFLOW, run_name, force=force)
    run_dir = mp.run_dir_path(cfg, WORKFLOW, run_name)
    mp.write_run_metadata(run_dir, run_metadata)
    _log, close_log = mp.open_run_log(run_dir)
    try:
        runs = mp.build_runs(model_type, [components], component_dust_types)
        for model_type_i, wind_components_i, component_dust_types_i in runs:
            report_run(cfg, data, model_type_i, wind_components_i, component_dust_types_i, run_name)
        print("\nAll runs complete.")
    finally:
        close_log()
