"""
Leave-n-campaigns-out cross-validation model- and dust-type-comparison workflow.

Sweeps two axes -- the soiling model and the dust/PM input channel -- via leave-n-
campaigns-out cross-validation. For each usable dust type and each selected model, it sweeps
the number of training campaigns n_train = 1..(C-1) (C = total campaigns), refitting from
scratch for every train/test split, and writes to
results/model_select/{site}/{run_name}/{dust_type}/{label}/:
  - cross_validation_folds.csv   : per-fold out-of-sample stats, in-sample R2, and
    fitted parameters (a NaN row for any fold whose fit failed);
  - cross_validation_summary.csv : per-n_train aggregated stats and a failed-fold
    count (n_failed);
  - cross_validation_summary.pdf : out-of-sample RMSE (daily soiling rate) and R2
    (reflectance) vs number of training campaigns.

Two estimators appear in the summary and they are not interchangeable. The `*_mean`/`*_std`
columns reduce the per-fold statistics, weighting each fold equally. The `*_pooled` columns
score every held-out prediction in one pass, weighting each observation equally; they are
defined only on the leave-one-campaign-out row (n_train = C-1), the one schedule under which
each campaign is held out exactly once, and are identical to simulate.py's cv_pooled row.
They diverge whenever folds carry different observation counts. R2 is reported pooled only:
each fold's R2 is normalised by its own held-out campaign's reflectance variance, so a mean
of fold R2s divides by a different denominator every time and estimates nothing (the per-fold
values are still plotted, as a scatter, and their spread kept as R2_fold_std).

The fold schedule is fixed: every training-set size 1..C-1 and every split of each. What is
compared is set by two arguments (see resolve_selection):

  `model_expression`  one wind model in the notation the workflows report ("gravitational +
                      normal_wind"); None compares every unique non-empty subset of the
                      wind-component vocabulary (mp.ALL_COMPONENT_COMBOS, 31 of them).
  `dust`              one measure ("PM10"), "all" (every measure the site has, no
                      differences), or "sweep".

"sweep" is what the dust axis is really for: it gives each mechanism of the named model its own
channel -- e.g. PM17 settling gravitationally while the 2.5-10 um fraction scours tangentially
-- over every assignment of the site's dust specs (the measures and their differences,
"PM17-PM10") to that model's mechanisms. Such a model does not depend on the outer dust type,
so it is fitted once under {run_name}/per-component-dust/ with its channels in the folder name
and in the reported model_expression. It costs len(dust_specs)**n_mechanisms runs, so it is
aimed at one model and gated behind a printed count and a confirmation (--force to run it
unattended).

After every run, all summaries for the site are compiled into all_models_kfold_summary.csv
(tagged with dust_type + model, sorted globally by pooled out-of-sample MAE with any
failed-fold run pushed to the bottom). The error metrics (MBE/MAE/RMSE) score the daily
soiling rate -- the drop in soiling factor per day -- so ranking rewards predicting the
*change* in soiling loss rather than the reflectance shape; R2 stays a reflectance goodness-of-
fit. All are scored on the common evaluation set (all common mirrors, identical reflectance
targets across dust types) for every run, so it is directly comparable across BOTH model and
dust type.

This module exposes ``run()`` -- it is driven by the unified CLI
(``python -m analysis_scripts.cli select ...``); see cli.py for the configuration surface,
model_pipeline.py for the shared fitting kernel, and simulate.py for the single-campaign
fit/report workflow.
"""

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import heliosoil.utilities as smu
from heliosoil.horizontal_impaction import parse_model_expression
from heliosoil.paper_specific_utilities import regression_performance_stats

from . import model_pipeline as mp


def _fold_task(
    cfg, data, model_type, wind_components, component_dust_types, train_mirrors_run, train_combo, test_combo, stitch=False, keep_model=False
):
    """Fit + evaluate one fold and return only the lightweight fields the summary needs.

    Runs on a worker thread; keeping the return value small lets the fitted model and its
    training inputs be freed as each fold finishes instead of accumulating one big dict per
    fold in flight (see model_pipeline.fit_and_evaluate and the Concurrency-safety notes).

    `stitch` additionally returns this fold's held-out soiling-factor predictions, which the
    leave-one-campaign-out folds need to assemble the pooled statistic. They are copied, since
    the fitted model they came from is dropped as soon as this returns. `keep_model` retains the
    fitted model itself for use as the stitched view's geometry template -- set on exactly one
    fold per run, because a model per fold is precisely the accumulation noted above."""
    result = mp.fit_and_evaluate(cfg, data, model_type, wind_components, component_dust_types, train_mirrors_run, train_combo, test_combo)
    out = {"stats_in": result["stats_in"], "stats_out": result["stats_out"], "param_names": result["param_names"], "param_hat": result["param_hat"]}
    if stitch:
        sf = result["model"].helios.soiling_factor
        out["stitch"] = {e: sf[e].copy() for e in test_combo}
    if keep_model:
        out["model"] = result["model"]
    return out


def cross_validate_run(
    cfg: mp.PipelineConfig,
    data: mp.LoadedData,
    model_type: str,
    wind_components: list,
    component_dust_types,
    run_name: str,
    subdir: str,
    executor: ThreadPoolExecutor,
    log,
) -> pd.DataFrame:
    """Leave-n-campaigns-out cross-validation for one (model_type, wind_components,
    component_dust_types) configuration, sweeping the number of training campaigns. Results
    go under `subdir` (the swept dust type, or mp.PER_COMPONENT_SUBDIR when each mechanism
    names its own dust channel). Folds are fit in parallel on `executor`; progress/PASS-FAILED
    detail goes to `log` (run.log). Returns the per-n_train summary DataFrame.

    The fold schedule is fixed: every training-set size n_train = 1..C-1 and, for each, every
    one of the C-choose-n_train splits (2**C - 2 folds in total). Sub-sampling it was only
    ever a way to trade away comparability for speed, and the campaign counts here (C <= 4)
    make the full schedule cheap."""
    expression = mp.run_expression(model_type, wind_components, component_dust_types)
    fit_method = mp.resolve_fit_method(cfg, model_type)
    log.info("=" * 80)
    log.info(f"dust={subdir!r}  model_type={model_type!r}  fit={fit_method}  {expression}")

    train_mirrors_run = mp.resolve_training_mirrors(cfg, data.all_mirrors, model_type, wind_components)
    results_dir, label = mp.results_dir_and_label(cfg, model_type, wind_components, component_dust_types, "model_select", run_name, subdir=subdir)

    C = len(data.files)

    # Enumerate every fold and dispatch its fit to the pool; results are gathered below in
    # submit order so the folds file is deterministic regardless of completion order.
    # Only the leave-one-campaign-out folds (n_train = C-1) hold out each campaign exactly once,
    # so only they can be stitched into a whole-site out-of-sample prediction and pooled. This is
    # the schedule simulate.py cross-validates on, which makes the pooled row directly comparable
    # to its cv_pooled statistic.
    n_train_pooled = C - 1

    tasks = []  # (row, n_train, fold_idx, train_combo, test_combo, future)
    for n_train in range(1, C):
        for fold_idx, train_combo in enumerate(combinations(range(C), n_train)):
            train_combo = list(train_combo)
            test_combo = [e for e in range(C) if e not in train_combo]
            row = {"n_train_campaigns": n_train, "fold": fold_idx, "train_experiments": str(train_combo), "test_experiments": str(test_combo)}
            stitch = n_train == n_train_pooled
            future = executor.submit(
                _fold_task,
                cfg,
                data,
                model_type,
                wind_components,
                component_dust_types,
                train_mirrors_run,
                train_combo,
                test_combo,
                stitch=stitch,
                keep_model=stitch and fold_idx == 0,
            )
            tasks.append((row, n_train, fold_idx, train_combo, test_combo, future))

    fold_rows = []
    stitched_soiling_factor, stitch_template = {}, None
    for row, n_train, fold_idx, train_combo, test_combo, future in tasks:
        try:
            result = future.result()
        except Exception as err:
            log.info(f"FAILED: n_train={n_train} fold={fold_idx} train={train_combo} test={test_combo} ({err})")
            row.update({"N": np.nan, "MBE": np.nan, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "R2_in_sample": np.nan})
            fold_rows.append(row)
            continue

        row.update(
            {
                "N": result["stats_out"]["N"],
                "MBE": result["stats_out"]["MBE"],
                "MAE": result["stats_out"]["MAE"],
                "RMSE": result["stats_out"]["RMSE"],
                "R2": result["stats_out"]["R2"],
                "R2_in_sample": result["stats_in"]["R2"],
            }
        )
        for name, value in zip(result["param_names"], result["param_hat"]):
            row[name] = value
        fold_rows.append(row)
        stitched_soiling_factor.update(result.get("stitch", {}))
        if result.get("model") is not None:
            stitch_template = result["model"]
        # A fold that completes but yields a non-finite out-of-sample R2 is a failure too
        # (and is counted in n_failed below), so label it FAILED rather than PASS.
        r2_is = result["stats_in"]["R2"]
        r2_oos = result["stats_out"]["R2"]
        status = "PASS" if np.isfinite(r2_oos) else "FAILED"
        log.info(f"{status}: n_train={n_train} fold={fold_idx} train={train_combo} test={test_combo} R2-IS={r2_is:.3f} R2-OOS={r2_oos:.3f}")

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(f"{results_dir}/cross_validation_folds.csv", index=False, float_format="%.3g")

    # The pooled out-of-sample statistic: every campaign predicted by the fold that held it out,
    # scored in ONE pass over all held-out predictions at once. This is a different estimator from
    # the fold means below -- pooling weights each residual equally, whereas an unweighted mean of
    # fold statistics weights each fold equally regardless of how many observations it carried.
    # For R2 the gap is structural rather than a matter of weights: each fold's R2 is normalised by
    # its own held-out campaign's reflectance variance, so averaging them across folds divides by
    # a different denominator every time and does not estimate anything. Only the pooled R2 is
    # reported for that reason. simulate.py computes the identical quantity as its cv_pooled row.
    pooled = {}
    if stitch_template is not None and len(stitched_soiling_factor) == C:
        stitched = mp.FoldStitchedModel(stitch_template, stitched_soiling_factor, {})
        stats = regression_performance_stats(stitched, data.reflect_data_total, list(range(C)))
        pooled = {f"{key}_pooled": stats[key] for key in ("MBE", "MAE", "RMSE", "R2")}
        pooled["N_pooled"] = stats["N"]
    else:
        # A failed fold leaves a campaign unpredicted, so there is no whole-site prediction to
        # pool. Better an absent statistic than one silently computed over a partial site.
        log.info(f"no pooled statistic: {len(stitched_soiling_factor)}/{C} campaigns stitched (a leave-one-out fold failed).")

    # Aggregate per n_train. n_failed counts the NaN (failed) folds so they are not silently
    # dropped by the NaN-skipping mean/std. Ordered by n_train; the meaningful cross-model
    # ranking (by pooled out-of-sample MAE) happens in mp.compile_results.
    summary_df = (
        folds_df.groupby("n_train_campaigns")
        .agg(
            n_folds=("fold", "size"),
            n_failed=("R2", lambda s: int(s.isna().sum())),
            MBE_mean=("MBE", "mean"),
            MBE_std=("MBE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_fold_std=("R2", "std"),
            R2_in_sample_mean=("R2_in_sample", "mean"),
        )
        .reset_index()
        .sort_values("n_train_campaigns")
        .reset_index(drop=True)
    )
    # Pooling needs each campaign held out exactly once, which only the leave-one-campaign-out
    # row satisfies; every smaller training size leaves the pooled columns blank.
    for column, value in pooled.items():
        summary_df[column] = np.where(summary_df["n_train_campaigns"] == n_train_pooled, value, np.nan)
    # Carried into the compiled all-models table: which mechanisms were fitted and, when they
    # differ, the dust channel driving each -- plus how they were fitted, since a run where
    # semi_physical fell back to MLE is not comparing like with like on that column alone.
    summary_df.insert(1, "model_expression", expression)
    summary_df.insert(2, "fit_method", fit_method)
    summary_df.to_csv(f"{results_dir}/cross_validation_summary.csv", index=False, float_format="%.3g")

    fig_cv, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_rmse.errorbar(summary_df["n_train_campaigns"], summary_df["RMSE_mean"], yerr=summary_df["RMSE_std"], marker="o", capsize=4)
    ax_rmse.set_xlabel("Number of training campaigns")
    ax_rmse.set_ylabel("Out-of-sample RMSE (daily soiling rate)")
    ax_rmse.set_xticks(summary_df["n_train_campaigns"])
    ax_rmse.grid(alpha=0.3)

    # Every fold's own R2 as a scatter, rather than a mean +- std across folds: fold R2s are each
    # normalised by a different campaign's reflectance variance, so their mean is not a quantity.
    # The pooled R2 (one pass over all held-out predictions) is the summary value, marked where it
    # is defined -- at the leave-one-campaign-out size only.
    ax_r2.scatter(folds_df["n_train_campaigns"], folds_df["R2"], marker="o", color="darkorange", alpha=0.6, label="per fold")
    if "R2_pooled" in summary_df.columns:
        defined = summary_df.dropna(subset=["R2_pooled"])
        ax_r2.scatter(defined["n_train_campaigns"], defined["R2_pooled"], marker="D", s=70, color="black", zorder=3, label="pooled")
    ax_r2.legend(fontsize=8)
    ax_r2.set_xlabel("Number of training campaigns")
    ax_r2.set_ylabel("Out-of-sample R2 (reflectance)")
    ax_r2.set_xticks(summary_df["n_train_campaigns"])
    ax_r2.grid(alpha=0.3)

    fig_cv.suptitle(f"Leave-n-out cross-validation: {label}")
    fig_cv.tight_layout()
    fig_cv.savefig(f"{results_dir}/cross_validation_summary.pdf", bbox_inches="tight")
    plt.close(fig_cv)

    return summary_df


def _best_mae(summary_df: pd.DataFrame) -> float:
    """Lowest pooled out-of-sample MAE among this run's rows that had no failed fold, or inf if
    none qualify -- so the best-so-far indicator never reflects a run/size with a failed fold.

    Pooled rather than the fold mean, matching the ranking mp.compile_results applies and the
    statistic simulate.py quotes, so the best-so-far readout tracks the same number the final
    table is sorted on."""
    ok = summary_df[summary_df["n_failed"] == 0]
    values = ok["MAE_pooled"].dropna() if "MAE_pooled" in ok.columns else pd.Series(dtype=float)
    return float(values.min()) if not values.empty else np.inf


def _execute_runs(cfg: mp.PipelineConfig, runs: list, dust_types: list, run_name: str, executor: ThreadPoolExecutor, log, bar, best: dict) -> None:
    """Cross-validate every (dust pass, run) work item, updating `bar` and `best` in place.

    A run whose every mechanism names its own dust channel does not depend on cfg.dust_type,
    so it is fitted once (on the first pass) under PER_COMPONENT_SUBDIR; every other run is
    fitted once per swept dust type -- which is what n_work_items counts."""
    for pass_idx, dust_type in enumerate(dust_types):
        run_cfg = dataclasses.replace(cfg, dust_type=dust_type)
        # Data is (re)loaded per dust type because SimulationInputs depend on it (the
        # reflectance targets do not).
        data = mp.load_data(run_cfg)

        for model_type_i, wind_components_i, component_dust_types_i in runs:
            if mp.is_dust_type_independent(model_type_i, wind_components_i, component_dust_types_i):
                if pass_idx > 0:
                    continue
                subdir = mp.PER_COMPONENT_SUBDIR
            else:
                subdir = dust_type

            cur_label = mp.abbreviate_run(model_type_i, wind_components_i, component_dust_types_i, subdir)
            bar.set_postfix_str(f"cur={cur_label}  best={best['label']} ({best['mae']:.4g})")
            summary_df = cross_validate_run(run_cfg, data, model_type_i, wind_components_i, component_dust_types_i, run_name, subdir, executor, log)

            run_mae = _best_mae(summary_df)
            if run_mae < best["mae"]:
                best.update(mae=run_mae, label=cur_label)
            bar.set_postfix_str(f"cur={cur_label}  best={best['label']} ({best['mae']:.4g})")
            bar.update(1)


def n_work_items(runs: list, dust_types: list) -> int:
    """How many cross-validation runs _execute_runs will actually fit: a dust-type-independent
    run once, every other run once per swept dust type."""
    return sum(1 for pass_idx in range(len(dust_types)) for run in runs if not (mp.is_dust_type_independent(*run) and pass_idx > 0))


def resolve_selection(cfg: mp.PipelineConfig, model_type: str | None, model_expression: str | None, dust: str, files: list) -> tuple[list, list, str]:
    """Turn (--model-type, --model, --dust) into the (runs, dust_types, mode) to execute.

    --model names one wind model; without it, every component combination is compared. --dust
    sweep drives each of that model's mechanisms with its own channel, over every assignment of
    the site's dust specs, so it needs a model to assign them to and refuses one that already
    names its channels -- there would be nothing left to sweep."""
    dust_types, dust_mode = mp.resolve_dust_selection(cfg, dust, files=files)
    components, component_dust_types = parse_model_expression(model_expression) if model_expression else (None, None)

    dust_specs = None
    if dust_mode == "sweep":
        if components is None:
            raise ValueError(
                '--dust sweep needs a --model to assign channels to, e.g. --model "gravitational + normal_wind". Use --dust all to compare every model on one channel each.'
            )
        named = [key for key, spec in component_dust_types.items() if spec is not None]
        if named:
            raise ValueError(
                f'--dust sweep assigns a dust channel to every mechanism, but --model already fixes one for {named}. Name the mechanisms only, e.g. --model "{" + ".join(components)}".'
            )
        dust_specs = mp.available_dust_specs(cfg, files=files)
        component_dust_types = "sweep"

    combos = [components] if components is not None else mp.ALL_COMPONENT_COMBOS
    return mp.build_runs(model_type, combos, component_dust_types, dust_specs), dust_types, dust_mode


def _confirm_sweep(components: list, dust_specs: list, n_runs: int, n_folds: int, force: bool) -> bool:
    """Report what a per-component dust sweep will cost and get permission to spend it.

    The sweep is len(dust_specs)**n_mechanisms runs -- 36 for a two-mechanism model at a
    three-measure site, 759,375 for a five-mechanism model at a five-measure one -- so the
    count is worth seeing before the run starts rather than discovering hours in. `force` runs
    it unattended; otherwise the user is asked, and a scripted run with nothing to ask is
    aborted rather than left to churn."""
    print("")
    print("=" * 80)
    print(f"Per-component dust sweep: {' + '.join(components)}")
    print(f"  dust specs ({len(dust_specs)}): {', '.join(dust_specs)}")
    print(f"  runs: {n_runs}  x  folds/run: {n_folds}  =  {n_runs * n_folds} fits")
    print("=" * 80)

    if force:
        return True
    answer = mp.prompt_yes_no("Run it? [y/N]")
    if answer is None:
        raise SystemExit("Aborted: no interactive terminal to confirm the sweep. Re-run with --force to run it unattended.")
    return answer


def run(
    cfg: mp.PipelineConfig,
    *,
    model_type: str | None,
    model_expression: str | None,
    dust: str,
    run_name: str | None,
    force: bool,
    jobs: int,
    run_metadata: dict,
) -> None:
    """Cross-validate every (dust_type, model) configuration for `cfg`'s site, with the folds
    of each run fit in parallel on `jobs` worker threads. A single tqdm bar tracks the runs,
    showing the current run and the best (lowest-MAE, no-failed-fold) run so far. Per-fold
    detail and suppressed warnings go to {run_dir}/run.log; the run's settings + version go to
    run_config.json; results are compiled into all_models_kfold_summary.csv at the end.

    What gets compared is set by `model_expression` (one wind model, or None to compare every
    component combination) and `dust` (one measure, "all", or "sweep"); see resolve_selection.
    A sweep is exponential in the number of mechanisms, so it is reported and confirmed before
    anything is fitted or written."""
    files = smu.get_training_data(cfg.data_dir, cfg.file_prefix)[0]
    runs, dust_types, dust_mode = resolve_selection(cfg, model_type, model_expression, dust, files)
    total = n_work_items(runs, dust_types)
    n_folds = 2 ** len(files) - 2  # every train/test split of every training-set size 1..C-1

    print(f"[dust] {cfg.location!r}: --dust {dust} -> {dust_types}")
    if dust_mode != "single" and len(dust_types) == 1:
        print(f"[dust] {cfg.location!r} has a single usable dust type ({dust_types[0]!r}); no PM-type comparison.")

    # Ask before creating the run folder, so declining leaves nothing behind.
    if dust_mode == "sweep":
        components, _ = parse_model_expression(model_expression)
        if not _confirm_sweep(components, mp.available_dust_specs(cfg, files=files), total, n_folds, force):
            return

    run_name = mp.resolve_run_dir(cfg, "model_select", run_name, force=force)
    run_dir = mp.run_dir_path(cfg, "model_select", run_name)
    mp.write_run_metadata(run_dir, run_metadata)
    log, close_log = mp.open_run_log(run_dir)

    best = {"mae": np.inf, "label": "--"}
    try:
        with ThreadPoolExecutor(max_workers=jobs) as executor, logging_redirect_tqdm(), tqdm(total=total, desc="model_select") as bar:
            _execute_runs(cfg, runs, dust_types, run_name, executor, log, bar, best)

        # Run right after the sweep: stack every (dust_type, model) summary for the site,
        # ranked globally by out-of-sample MAE, flagging any run with failed folds.
        mp.compile_results(cfg, run_name, workflow="model_select")
        print("\nAll runs complete.")
    finally:
        close_log()
