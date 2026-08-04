"""
Soiling-model fit + report workflow.

Unless --train-experiments names a training split, each selected model is cross-validated
leave-one-campaign-out: for a site with C campaigns there are C folds, fold k training on every
campaign except campaign k and testing on campaign k. Every campaign therefore gets a genuine
out-of-sample prediction, instead of the out-of-sample number being an accident of which campaign
happened to be listed first. Naming --train-experiments keeps the older single-fit behaviour.

Per model, both modes report:
  - fitted parameters (+ 95% CI) and in-sample/out-of-sample performance stats to CSV -- one row
    per fold, plus across-fold aggregates and the pooled cross-validated statistic, when
    cross-validating;
  - plot_for_paper figures (grouped by tilt, one curve per orientation) across all campaigns:
    all_campaigns.pdf for a single fit; for cross-validation, fold_k/all_campaigns.pdf per fold
    (its training campaigns highlighted) plus all_campaigns_test.pdf, which draws every campaign
    from the fold that held it out;
  - per-campaign, per-tilt measured-vs-predicted reflectance plots (one per tilt actually present
    in the data), named for the campaign's role in the fit that drew them
    ("reflectance_tilt_00_train-f2.pdf", "reflectance_tilt_00_test-f1.pdf"), each annotated with
    that tilt's MAE/RMSE, plus a per-campaign stats CSV;
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
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heliosoil.horizontal_impaction import parse_model_expression
from heliosoil.paper_specific_utilities import daily_rate_residuals, plot_for_paper, plot_reflectance_by_tilt, regression_performance_stats

from . import model_pipeline as mp
from .campaign_summary import FIGURE_RC, site_display_name

WORKFLOW = "simulate"
# Column order of the cross-validated performance-stats table. regression_performance_stats' own
# key order puts N last; a stats file is read left-to-right, and the count qualifies every metric
# after it.
STAT_KEYS = ["N", "MBE", "MAE", "RMSE", "R2"]


def leave_one_campaign_out_folds(n_campaigns: int) -> list[tuple[int, list[int], list[int]]]:
    """The cross-validation schedule: (fold, train_experiments, test_experiments) per fold.

    Fold numbering is global and 1-based, and fold k holds out campaign k -- the same numbering the
    campaign_{e+1} folders use -- so "campaign_1/reflectance_tilt_00_train-f2.pdf" is fold 2's
    model, and ties to fold_2/ and to the fold_2 rows of the CSVs without a lookup.

    Requires at least two campaigns: with one, a fold would have nothing to train on. That is a
    configuration error the caller reports as such (the CLI turns ValueError into a usage error),
    not something to silently degrade into a single fit."""
    if n_campaigns < 2:
        raise ValueError(
            f"leave-one-campaign-out cross-validation needs at least 2 campaigns, this site has {n_campaigns}. "
            "Pass --train-experiments 0 to fit that single campaign instead."
        )
    return [(k + 1, [e for e in range(n_campaigns) if e != k], [k]) for k in range(n_campaigns)]


def tilt_figure_name(tilt: float, role: str, fold: int | None = None) -> str:
    """Filename of one campaign's per-tilt reflectance figure.

    `role` is the campaign's role in the fit that drew it ("train"/"test"); `fold` is the
    cross-validation fold that produced it, omitted for a single fit. A campaign therefore holds
    one figure per fit that touched it -- reflectance_tilt_00_train-f2.pdf,
    reflectance_tilt_00_test-f1.pdf -- rather than one figure each fold silently overwrites."""
    suffix = role if fold is None else f"{role}-f{fold}"
    return f"reflectance_tilt_{tilt:02.0f}_{suffix}.pdf"


class FoldStitchedModel:
    """A model-shaped view of the cross-validation's out-of-sample predictions: every campaign's
    prediction taken from the fold that held that campaign out.

    plot_for_paper, plot_reflectance_by_tilt and regression_performance_stats each take a single
    fitted model, so the only way to show or score the whole site out-of-sample at once is to hand
    them one object whose per-campaign predictions come from different fits. Everything but the
    predictions is fold-invariant (every fold ends by re-running helios_angles on the full
    evaluation data), so tilt/azimuth/nominal_reflectance are taken from `template`, an arbitrary
    fold's fitted model.

    predict_soiling_factor is a no-op on purpose: both plotting helpers call it on entry, which
    would otherwise overwrite the stitched predictions with the template fold's own."""

    def __init__(self, template, soiling_factor: dict, prediction_variance: dict):
        self._template = template
        # Shallow copy, then new dicts for the two stitched fields: the template's own predictions
        # must stay untouched, since it is a real fitted model the caller may still be using.
        self.helios = copy.copy(template.helios)
        self.helios.soiling_factor = dict(soiling_factor)
        self.helios.soiling_factor_prediction_variance = dict(prediction_variance)

    def predict_soiling_factor(self, *args, **kwargs) -> None:
        """No-op: the stitched predictions are already the out-of-sample ones."""

    def __getattr__(self, name):
        # Only reached for attributes this class does not define. The underscore guard keeps a
        # partially-initialised instance (during copy/pickle, say) from recursing on _template.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._template, name)


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


def _merge_rate_error_groups(*group_dicts) -> dict:
    """Pool several _abs_rate_error_by_tilt results into one, tilt by tilt.

    Cross-validation's in-sample side is one such result per fold -- each fold over its own
    training campaigns -- and the question the figure answers ("how big is the in-sample error at
    this tilt?") is about all of them together."""
    merged: dict = {}
    for groups in group_dicts:
        for tilt, values in groups.items():
            merged.setdefault(tilt, []).append(values)
    return {tilt: np.concatenate(parts) for tilt, parts in sorted(merged.items())}


def plot_rate_error_groups(
    in_sample: dict, out_of_sample: dict, site: str, split_labels: tuple = ("In-sample", "Out-of-sample"), title_note: str = ""
) -> plt.Figure | None:
    """Distribution of the absolute daily soiling-rate error at each tilt, in-sample vs out-of-sample.

    performance_stats.csv reports one MAE per split and one per tilt, but not how that error is
    spread -- whether a tilt's MAE comes from a uniformly mediocre fit or from a few bad intervals.
    Each box pools every |predicted - measured daily rate| over the mirrors at that tilt and the
    measurement intervals of the split's campaigns; its mean marker is that group's MAE.

    Takes the two groupings (from _abs_rate_error_by_tilt) rather than a model and two splits, so
    each side can come from a different fit: under cross-validation the in-sample side pools every
    fold's training campaigns while the out-of-sample side is each campaign's held-out prediction.

    Tilt is a categorical axis, unlike experiment_soiling.plot_soiling_rate_vs_tilt's numeric one:
    here the claim is a per-group comparison, not a trend, and carwarp/yadnarie's 180 deg mirror
    would otherwise strand the 0/30/60 cluster at one end. Fliers are drawn rather than hidden
    because MAE is mean-based and it is the tail that drives it.

    Returns None when neither split yields a residual at any tilt, so the caller writes no file."""
    tilts = sorted(set(in_sample) | set(out_of_sample))
    if not tilts:
        return None

    # Both boxes are white-filled and separated by edge colour and mean-marker shape: a solid fill
    # swallows the median line, which is the one mark a reader compares across the pair.
    splits = [(split_labels[0], in_sample, -0.18, "C0", "o"), (split_labels[1], out_of_sample, 0.18, "C3", "D")]

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
            # A box over one mirror's handful of intervals is not a distribution; saying n makes
            # that plain. In its own row above the axes, not at the floor, where the low-error
            # boxes sit. It is also what shows a cross-validated in-sample side pooling (C-1)
            # folds' worth of residuals against one held-out campaign's.
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
        fig.suptitle(f"{site} — Daily soiling-rate error by tilt{title_note}")
        fig.tight_layout()
    return fig


def plot_rate_error_by_tilt(model, rdat, train_experiments, test_experiments, site) -> plt.Figure | None:
    """plot_rate_error_groups for a single fit, whose two splits are two sets of campaigns scored
    by the same model."""
    return plot_rate_error_groups(
        _abs_rate_error_by_tilt(model, rdat, train_experiments), _abs_rate_error_by_tilt(model, rdat, test_experiments), site
    )


def rate_error_groups_table(in_sample: dict, out_of_sample: dict) -> pd.DataFrame:
    """The numbers behind plot_rate_error_groups, one row per (split, tilt).

    Named _pp_day because these carry the figure's x100 and so are 100x performance_stats.csv's
    MAE, which is in soiling-factor per day -- the unit has to be on the column or the two tables
    look like they disagree about the same model."""
    rows = []
    for split, groups in (("in_sample", in_sample), ("out_of_sample", out_of_sample)):
        for tilt, values in groups.items():
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


def rate_error_by_tilt_table(model, rdat, train_experiments, test_experiments) -> pd.DataFrame:
    """rate_error_groups_table for a single fit (see plot_rate_error_by_tilt)."""
    return rate_error_groups_table(_abs_rate_error_by_tilt(model, rdat, train_experiments), _abs_rate_error_by_tilt(model, rdat, test_experiments))


def _all_campaigns_figure(model, data: mp.LoadedData, train_experiments: list, train_mirrors: list, orientation: list, path: str) -> None:
    """Predicted vs. measured soiling factor across all campaigns (grouped by tilt, one coloured
    curve per orientation), with `train_experiments` x `train_mirrors` highlighted as the fit's
    training window. train_experiments=[] highlights nothing, which is what the stitched
    out-of-sample view is: no campaign in it was trained on."""
    fig, _ax, _ref_output = plot_for_paper(
        model,
        data.reflect_data_total,
        data.sim_data_total,
        train_experiments,
        train_mirrors,
        orientation,
        figsize=(16, 15),
        num_legend_cols=7,
        auto_yticks=True,
        plot_rh=False,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _campaign_tilt_figures(model, data: mp.LoadedData, orientation: list, results_dir: str, e: int, role: str, fold: int | None = None) -> list[dict]:
    """Write campaign `e`'s per-tilt measured-vs-predicted reflectance figures for one fit and
    return the per-tilt stat rows (plus a whole-campaign "all" row) rather than writing them:
    under cross-validation a campaign's CSV pools the rows of every fold that touched it.

    The tilts are the ones actually present in the data (reflect_data_total.tilts, the exact
    values plot_reflectance_by_tilt matches on), so there is nothing for a user to mis-specify."""
    campaign_dir = f"{results_dir}/campaign_{e + 1}"
    rows = []
    for t in np.unique(data.reflect_data_total.tilts[e][:, 0]):
        fig_t, _, tilt_stats = plot_reflectance_by_tilt(model, data.reflect_data_total, data.sim_data_total, e, t, orientation[e])
        if tilt_stats is None:
            continue
        os.makedirs(campaign_dir, exist_ok=True)
        fig_t.savefig(f"{campaign_dir}/{tilt_figure_name(t, role, fold)}", bbox_inches="tight")
        plt.close(fig_t)
        rows.append({"tilt": t, **tilt_stats})

    if rows:
        rows.append({"tilt": "all", **regression_performance_stats(model, data.reflect_data_total, [e])})
    return rows


def _report_single_fit(
    cfg: mp.PipelineConfig, data: mp.LoadedData, run: tuple, results_dir: str, expression: str, train_mirrors_run: list, orientation: list, site: str
) -> None:
    """Fit once on cfg.train_experiments, score every other campaign out-of-sample, and write that
    fit's parameters, performance stats and plots."""
    test_experiments = [f for f in range(len(data.files)) if f not in cfg.train_experiments]

    primary = mp.fit_and_evaluate(cfg, data, *run, train_mirrors_run, cfg.train_experiments, test_experiments)
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

    _all_campaigns_figure(imodel, data, cfg.train_experiments, train_mirrors_run, orientation, f"{results_dir}/all_campaigns.pdf")

    # How the daily soiling-rate error is distributed within each tilt, in-sample vs out-of-sample.
    # helios.soiling_factor is already populated over every campaign by fit_and_evaluate, so this
    # re-reads the same prediction the stats above were computed from.
    fig_err = plot_rate_error_by_tilt(imodel, data.reflect_data_total, cfg.train_experiments, test_experiments, site)
    if fig_err is not None:
        fig_err.savefig(f"{results_dir}/rate_error_by_tilt.pdf", bbox_inches="tight")
        plt.close(fig_err)
        rate_error_by_tilt_table(imodel, data.reflect_data_total, cfg.train_experiments, test_experiments).to_csv(
            f"{results_dir}/rate_error_by_tilt.csv", index=False, float_format="%.3g"
        )

    # Per-campaign, per-tilt measured-vs-predicted reflectance plots, each annotated with that
    # tilt's MAE/RMSE and named for whether this campaign was trained on; a CSV with those plus the
    # whole-campaign (all common mirrors) stats is written alongside them.
    for e in range(len(data.files)):
        rows = _campaign_tilt_figures(imodel, data, orientation, results_dir, e, "train" if e in cfg.train_experiments else "test")
        if rows:
            pd.DataFrame(rows).to_csv(f"{results_dir}/campaign_{e + 1}/performance_stats.csv", index=False, float_format="%.3g")


def _report_cross_validation(
    cfg: mp.PipelineConfig, data: mp.LoadedData, run: tuple, results_dir: str, expression: str, train_mirrors_run: list, orientation: list, site: str
) -> None:
    """Leave-one-campaign-out: refit the model once per campaign, each fold trained on every other
    campaign, and report the folds together with the out-of-sample prediction they add up to.

    Each fold's figures are written inside the loop and its model then dropped, so peak memory is
    one fitted model rather than one per campaign; what is kept instead are the two arrays needed
    to stitch the held-out predictions together (see FoldStitchedModel)."""
    all_experiments = list(range(len(data.files)))
    folds = leave_one_campaign_out_folds(len(data.files))
    rdat = data.reflect_data_total

    param_rows, stat_rows = [], []
    campaign_rows: dict = {e: [] for e in all_experiments}
    in_sample_groups: dict = {}
    stitched_soiling_factor, stitched_variance = {}, {}
    template = None

    print(f"\nLeave-one-campaign-out cross-validation: {len(folds)} folds over {len(data.files)} campaigns.")
    for fold, train_experiments, test_experiments in folds:
        (e_test,) = test_experiments
        result = mp.fit_and_evaluate(cfg, data, *run, train_mirrors_run, train_experiments, test_experiments)
        model = result["model"]

        _all_campaigns_figure(model, data, train_experiments, train_mirrors_run, orientation, f"{results_dir}/fold_{fold}/all_campaigns.pdf")

        # This fold's view of every campaign: the held-out one gets its test figure, each training
        # campaign the training figure it contributes to its own campaign folder.
        for e in all_experiments:
            in_sample = e != e_test
            rows = _campaign_tilt_figures(model, data, orientation, results_dir, e, "train" if in_sample else "test", fold)
            campaign_rows[e] += [{"fold": fold, "split": "in_sample" if in_sample else "out_of_sample", **row} for row in rows]

        for name, value, lower, upper in zip(result["param_names"], result["param_hat"], result["lower_ci"], result["upper_ci"]):
            param_rows.append(
                {
                    "fit": f"fold_{fold}",
                    "train_experiments": str(train_experiments),
                    "test_experiments": str(test_experiments),
                    "parameter": name,
                    "value": value,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
        stat_rows += [
            {
                "fit": f"fold_{fold}",
                "split": "in_sample",
                "model_expression": expression,
                "experiments": str(train_experiments),
                **result["stats_in"],
            },
            {
                "fit": f"fold_{fold}",
                "split": "out_of_sample",
                "model_expression": expression,
                "experiments": str(test_experiments),
                **result["stats_out"],
            },
        ]
        print(
            f"fold {fold}: train={train_experiments} test={test_experiments}  "
            f"R2-in={result['stats_in']['R2']:.3f}  R2-out={result['stats_out']['R2']:.3f}  MAE-out={result['stats_out']['MAE']:.4f}"
        )

        in_sample_groups = _merge_rate_error_groups(in_sample_groups, _abs_rate_error_by_tilt(model, rdat, train_experiments))
        # Copied, not referenced: the template fold's own arrays would otherwise alias the stitched
        # dict, and any later prediction on that model would silently rewrite this fold's result.
        stitched_soiling_factor[e_test] = model.helios.soiling_factor[e_test].copy()
        # A least-squares fit carries no noise process, so there is no prediction variance to
        # stitch; the stitched model then simply has no interval to draw, as each fold had none.
        fold_variance = model.helios.soiling_factor_prediction_variance.get(e_test)
        if fold_variance is not None:
            stitched_variance[e_test] = fold_variance.copy()
        template = model

    # Every campaign was held out by exactly one fold, so the stitched model predicts the whole
    # site out-of-sample -- which makes regression_performance_stats over all campaigns the exact
    # pooled cross-validated statistic, with no per-fold weighting to get wrong.
    stitched = FoldStitchedModel(template, stitched_soiling_factor, stitched_variance)
    pooled = regression_performance_stats(stitched, rdat, all_experiments)
    _all_campaigns_figure(stitched, data, [], train_mirrors_run, orientation, f"{results_dir}/all_campaigns_test.pdf")

    print(f"\nCross-validated (every campaign predicted by the fold that held it out, N={pooled['N']}):")
    print(f"  MBE  = {pooled['MBE']:.4f}  (daily soiling rate)")
    print(f"  MAE  = {pooled['MAE']:.4f}  (daily soiling rate)")
    print(f"  RMSE = {pooled['RMSE']:.4f}  (daily soiling rate)")
    print(f"  R2   = {pooled['R2']:.3f}  (reflectance)")

    # Aggregate rows: the metrics are the mean/std over the folds of a split, N the total number of
    # observations behind that mean (blank for the spread row, which has no count of its own).
    # cv_pooled is not a mean of folds -- it is the single statistic over every held-out prediction
    # at once, and it is the one to quote.
    fold_stats = pd.DataFrame(stat_rows)
    for split in ("in_sample", "out_of_sample"):
        split_rows = fold_stats[fold_stats["split"] == split]
        base = {"split": split, "model_expression": expression, "experiments": f"{len(split_rows)} folds"}
        # int, and an empty spread cell rather than NaN: a NaN would make N a float column, and
        # to_csv's 3-significant-digit float format would round an observation count to 1.09e+03.
        stat_rows.append({"fit": "cv_mean", **base, "N": int(split_rows["N"].sum()), **{key: split_rows[key].mean() for key in STAT_KEYS[1:]}})
        stat_rows.append({"fit": "cv_std", **base, "N": "", **{key: split_rows[key].std() for key in STAT_KEYS[1:]}})
    stat_rows.append({"fit": "cv_pooled", "split": "out_of_sample", "model_expression": expression, "experiments": str(all_experiments), **pooled})
    pd.DataFrame(stat_rows)[["fit", "split", "model_expression", "experiments", *STAT_KEYS]].to_csv(
        f"{results_dir}/performance_stats.csv", index=False, float_format="%.3g"
    )

    # Per-fold parameters, plus how far each one moved across the folds: a parameter whose
    # across-fold spread swamps its own confidence interval is not identified by this data.
    params_df = pd.DataFrame(param_rows)
    spread = params_df.groupby("parameter", sort=False)["value"].agg(["mean", "std"]).reset_index()
    aggregate_rows = [
        {"fit": fit, "train_experiments": f"{len(folds)} folds", "test_experiments": "", "parameter": row["parameter"], "value": row[column]}
        for fit, column in (("cv_mean", "mean"), ("cv_std", "std"))
        for _, row in spread.iterrows()
    ]
    pd.concat([params_df, pd.DataFrame(aggregate_rows)], ignore_index=True).to_csv(
        f"{results_dir}/fitted_parameters.csv", index=False, float_format="%.3g"
    )
    print("")
    for _, row in spread.iterrows():
        print(f"{row['parameter']:16s} = {row['mean']:.3e}  (across-fold std {row['std']:.3e})")

    # In-sample pools every fold's own training campaigns, so it carries (C-1)x the observations of
    # the out-of-sample side; the figure's per-box n row says so.
    out_of_sample_groups = _abs_rate_error_by_tilt(stitched, rdat, all_experiments)
    fig_err = plot_rate_error_groups(
        in_sample_groups,
        out_of_sample_groups,
        site,
        split_labels=("In-sample (pooled folds)", "Out-of-sample (held-out fold)"),
        title_note=" (leave-one-campaign-out)",
    )
    if fig_err is not None:
        fig_err.savefig(f"{results_dir}/rate_error_by_tilt.pdf", bbox_inches="tight")
        plt.close(fig_err)
        rate_error_groups_table(in_sample_groups, out_of_sample_groups).to_csv(
            f"{results_dir}/rate_error_by_tilt.csv", index=False, float_format="%.3g"
        )

    for e, rows in campaign_rows.items():
        if rows:
            pd.DataFrame(rows).to_csv(f"{results_dir}/campaign_{e + 1}/performance_stats.csv", index=False, float_format="%.3g")


def report_run(cfg: mp.PipelineConfig, data: mp.LoadedData, model_type: str, wind_components: list, component_dust_types, run_name: str) -> None:
    """Fit one (model_type, wind_components, component_dust_types) configuration -- on
    cfg.train_experiments, or leave-one-campaign-out when it is None -- and write its fitted
    parameters, performance stats, and plots."""
    expression = mp.run_expression(model_type, wind_components, component_dust_types)
    # The method actually used, not the one requested: a model type without a least-squares fit
    # falls back to MLE, and the header is where that has to be visible.
    fit_method = mp.resolve_fit_method(cfg, model_type)
    fallback = "" if fit_method == cfg.fit_method else f" (fit_method={cfg.fit_method!r} unavailable for this model type)"
    print(f"\n{'=' * 80}\nmodel_type={model_type!r}  fit={fit_method}{fallback}  {expression}\n{'=' * 80}")

    train_mirrors_run = mp.resolve_training_mirrors(cfg, data.all_mirrors, model_type, wind_components)
    results_dir, _label = mp.results_dir_and_label(cfg, model_type, wind_components, component_dust_types, WORKFLOW, run_name)
    # reflect_data_total.mirror_names[e] == all_mirrors for every e, so orientation is built from
    # all_mirrors (train_mirrors_run only labels the training highlight).
    orientation = [[mp.orientation_code(name) for name in data.all_mirrors] for _ in data.files]
    site = site_display_name(cfg.location)

    report = _report_cross_validation if cfg.train_experiments is None else _report_single_fit
    report(cfg, data, (model_type, wind_components, component_dust_types), results_dir, expression, train_mirrors_run, orientation, site)


def run(
    cfg: mp.PipelineConfig, *, model_type: str | None, model_expression: str | None, run_name: str | None, force: bool, run_metadata: dict
) -> None:
    """Fit + report the model(s) named by `model_type` and `model_expression` for `cfg`'s site.

    Each model is cross-validated leave-one-campaign-out unless cfg.train_experiments names the
    campaigns to train on, in which case that one split is fitted and reported.

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
    if cfg.train_experiments is None:
        # Validate the schedule before the run folder is created, so a single-campaign site fails
        # with a usage error rather than after the first fold has already written figures.
        leave_one_campaign_out_folds(len(data.files))

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
