"""
Leave-n-campaigns-out cross-validation model- and dust-type-comparison workflow.

Sweeps two axes -- the soiling model and the dust/PM input channel -- via leave-n-
campaigns-out cross-validation. For each usable dust type (DUST_TYPES, or auto-detected
per site) and each selected model, it sweeps the number of training campaigns
n_train = 1..(C-1) (C = total campaigns), refitting from scratch for every train/test
split, and writes to results/model_select/{site}/{run_name}/{dust_type}/{label}/:
  - cross_validation_folds.csv   : per-fold out-of-sample stats, in-sample R2, and
    fitted parameters (a NaN row for any fold whose fit failed);
  - cross_validation_summary.csv : per-n_train aggregated stats and a failed-fold
    count (n_failed);
  - cross_validation_summary.pdf : out-of-sample RMSE/R2 vs number of training campaigns.

After every run, all summaries for the site are compiled into all_models_kfold_summary.csv
(tagged with dust_type + model, sorted within each n_train by out-of-sample RMSE, best
first), flagging any run that failed one or more folds so a strong-looking mean on the
surviving folds is not taken at face value.

Ranking uses out-of-sample RMSE/R2, scored on the common evaluation set (all common
mirrors, identical reflectance targets across dust types) for every run, so it is
directly comparable across BOTH model and dust type. A training-fit AIC is deliberately
not used: model families train on different mirror sets (one representative mirror vs
all mirrors), so their likelihood magnitudes -- and hence AIC -- are dominated by
observation count rather than merit and are not comparable. Sites with a single usable
dust type (e.g. TSP-only) simply run that one. See model_pipeline.py for the shared
fitting kernel and model-type/dust-type notes, and single_campaign_fit.py for the
single-campaign fit/report workflow. Configure the run from the constants below, then
run directly.
"""

import logging
import dataclasses
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heliosoil.utilities as smu
from heliosoil.utilities import configure_logging

import model_pipeline as mp

# ============================== Configuration ===============================
CONFIG = mp.PipelineConfig(
    location="yadnarie",  # "mountisa" | "carwarp" | "yadnarie" | "qut" | "ablrf" | "wodonga"
    train_experiments=[0],  # unused by the CV sweep, but kept for a consistent config
    train_mirrors=None,  # None -> model-aware default per run; explicit list overrides every run
    dust_type="PM10",  # fallback only; the dust type is swept per DUST_TYPES below
    k_factor="import",  # None sets equal to 1.0, "import" imports from the file
    second_surf=True,  # True: second-surface AOI model, False: first-surface
    verbose=False,
)

RUN_NAME = None  # None -> "run-yy-mm-dd_hh-mm" timestamp; else a label for this run's results folder

DUST_TYPES = None  # None -> auto-detect usable PM/TSP types for the site; else e.g. ["PM2.5", "PM10", "PMT"]

MODEL_TYPE = None  # "constant_mean" | "constant_mean_wind" | "semi_physical" | None (run all three)
WIND_COMPONENTS = None  # only used when MODEL_TYPE == "constant_mean_wind"; None -> sweep WIND_COMPONENT_COMBOS
WIND_COMPONENT_COMBOS = [
    ["gravitational"],
    ["gravitational", "normal_wind"],
    ["gravitational", "normal_wind", "tangential_wind"],
    ["gravitational", "tangential_wind"],
    ["gravitational", "impaction_retention"],
    ["gravitational", "tangential_wind", "impaction_retention"],
    ["normal_wind", "tangential_wind"],
    ["tangential_wind", "impaction_retention"],
    ["normal_wind"],
    ["tangential_wind"],
    ["impaction_retention"],
]  # used when MODEL_TYPE == "constant_mean_wind" and WIND_COMPONENTS is None

CV_TRAIN_CAMPAIGN_COUNTS = None  # None -> sweep every n_train in 1..(C-1)
CV_MAX_FOLDS_PER_SIZE = None  # None -> use every combination; else randomly sample this many (seeded)
# ==============================================================================


def cross_validate_run(cfg, data, model_type, wind_components, run_name, dust_type):
    """Leave-n-campaigns-out cross-validation for one (dust_type, model_type,
    wind_components) configuration, sweeping the number of training campaigns."""
    print(f"\n{'=' * 80}\ndust_type={dust_type!r}  model_type={model_type!r}  wind_components={wind_components}\n{'=' * 80}")

    train_mirrors_run = mp.resolve_training_mirrors(cfg, data.all_mirrors, model_type, wind_components)
    results_dir, label = mp.results_dir_and_label(cfg, model_type, wind_components, "model_select", run_name, subdir=dust_type)

    C = len(data.files)
    train_counts = CV_TRAIN_CAMPAIGN_COUNTS if CV_TRAIN_CAMPAIGN_COUNTS is not None else range(1, C)

    fold_rows = []
    for n_train in train_counts:
        combos = list(combinations(range(C), n_train))
        if CV_MAX_FOLDS_PER_SIZE is not None and len(combos) > CV_MAX_FOLDS_PER_SIZE:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(combos), size=CV_MAX_FOLDS_PER_SIZE, replace=False)
            combos = [combos[i] for i in idx]

        for fold_idx, train_combo in enumerate(combos):
            train_combo = list(train_combo)
            test_combo = [e for e in range(C) if e not in train_combo]

            row = {"n_train_campaigns": n_train, "fold": fold_idx, "train_experiments": str(train_combo), "test_experiments": str(test_combo)}
            try:
                result = mp.fit_and_evaluate(cfg, data, model_type, wind_components, train_mirrors_run, train_combo, test_combo)
            except Exception as err:
                print(f"FAILED: n_train={n_train} fold={fold_idx} train={train_combo} test={test_combo} ({err})")
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
            # A fold that completes but yields a non-finite out-of-sample R2 is a
            # failure too (and is counted in n_failed below), so label it FAILED
            # rather than PASS for consistency with the summary.
            r2_is = result["stats_in"]["R2"]
            r2_oos = result["stats_out"]["R2"]
            status = "PASS" if np.isfinite(r2_oos) else "FAILED"
            print(f"{status}: n_train={n_train} fold={fold_idx} train={train_combo} test={test_combo} R2-IS={r2_is:.3f} R2-OOS={r2_oos:.3f}")

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(f"{results_dir}/cross_validation_folds.csv", index=False, float_format="%.3g")

    # Aggregate per n_train. n_failed counts the NaN (failed) folds so they are not
    # silently dropped by the NaN-skipping mean/std. Ordered by n_train; the meaningful
    # cross-model ranking (by out-of-sample RMSE) happens in compile_results.
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
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            R2_in_sample_mean=("R2_in_sample", "mean"),
        )
        .reset_index()
        .sort_values("n_train_campaigns")
        .reset_index(drop=True)
    )
    summary_df.to_csv(f"{results_dir}/cross_validation_summary.csv", index=False, float_format="%.3g")

    fig_cv, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_rmse.errorbar(summary_df["n_train_campaigns"], summary_df["RMSE_mean"], yerr=summary_df["RMSE_std"], marker="o", capsize=4)
    ax_rmse.set_xlabel("Number of training campaigns")
    ax_rmse.set_ylabel("Out-of-sample RMSE")
    ax_rmse.set_xticks(summary_df["n_train_campaigns"])
    ax_rmse.grid(alpha=0.3)

    ax_r2.errorbar(summary_df["n_train_campaigns"], summary_df["R2_mean"], yerr=summary_df["R2_std"], marker="o", capsize=4, color="darkorange")
    ax_r2.set_xlabel("Number of training campaigns")
    ax_r2.set_ylabel("Out-of-sample R2")
    ax_r2.set_xticks(summary_df["n_train_campaigns"])
    ax_r2.grid(alpha=0.3)

    fig_cv.suptitle(f"Leave-n-out cross-validation: {label}")
    fig_cv.tight_layout()
    fig_cv.savefig(f"{results_dir}/cross_validation_summary.pdf", bbox_inches="tight")
    plt.close(fig_cv)


def main():
    # Route all HelioSoil output through the "heliosoil" logger: verbose -> INFO
    # (progress messages), else WARNING (only genuine warnings/failures).
    configure_logging(logging.INFO if CONFIG.verbose else logging.WARNING)

    files = smu.get_training_data(CONFIG.data_dir, CONFIG.file_prefix)[0]
    dust_types = DUST_TYPES or mp.available_dust_types(CONFIG, files=files)
    print(f"[dust] sweeping dust types: {dust_types}")
    if len(dust_types) == 1:
        print(f"[dust] {CONFIG.location!r} has a single usable dust type ({dust_types[0]!r}); no PM-type comparison.")

    run_name = mp.resolve_run_dir(CONFIG, "model_select", RUN_NAME)
    runs = mp.build_runs(MODEL_TYPE, WIND_COMPONENTS, WIND_COMPONENT_COMBOS)

    # Outer loop over dust/PM types; each gets a per-type config copy so load_data and
    # fit_and_evaluate (which read cfg.dust_type) use it. Data is (re)loaded per dust
    # type because the SimulationInputs depend on it (the reflectance targets do not).
    for dust_type in dust_types:
        run_cfg = dataclasses.replace(CONFIG, dust_type=dust_type)
        data = mp.load_data(run_cfg)
        for model_type, wind_components in runs:
            cross_validate_run(run_cfg, data, model_type, wind_components, run_name, dust_type)

    # Run right after the sweep: stack every (dust_type, model) summary for the site,
    # ranked by out-of-sample RMSE, flagging any run with failed folds.
    mp.compile_results(CONFIG, run_name, workflow="model_select")
    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
