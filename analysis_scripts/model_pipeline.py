"""
Shared kernel for the HelioSoil analysis workflows.

Holds the configuration, data loading, model construction, and fit/evaluate logic
shared by the two workflow scripts:

  - single_campaign_fit.py -- fit each model on the training campaign(s) and report
    fitted parameters, performance statistics, and per-campaign plots.
  - model_selection.py     -- leave-n-campaigns-out cross-validation sweeping the
    number of training campaigns, plus out-of-sample-RMSE-ranked model comparison.

Nothing here runs at import time; the workflow scripts import these helpers and
drive them from their own main().

Model types
-----------
  - "constant_mean"      : ConstantMeanDeposition (mu_tilde, sigma_dep).
  - "constant_mean_wind" : ConstantMeanWindDeposition; its parameter names/count
    depend on the active wind `components` (any subset of "gravitational" /
    "turbulent_wind" / "normal_wind" / "tangential_wind" / "impaction_retention"; see
    heliosoil.horizontal_impaction), so they are read from the fitted model's
    `parameter_names` property rather than hardcoded.
  - "semi_physical"      : SemiPhysical (hrz0, sigma_dep).

Dust channels
-------------
A run is defined by (model_type, wind_components, component_dust_types). The last of
these drives each wind mechanism with its own dust spec -- a single measure ("PM17",
"PM10") or the difference of two ("PM17-PM10", the mass coarser than 10 µm) -- so a
model can settle whole-distribution mass gravitationally while scouring only a narrow
size band tangentially. It defaults to None, meaning every mechanism uses cfg.dust_type
and the dust axis is the workflow's outer dust-type sweep. See build_runs,
available_dust_specs and component_dust_assignments.

Training-mirror selection is model-aware (see smu.default_training_mirrors): a
purely gravitational/constant-mean model shares one deposition-noise process across
mirrors and trains on a single representative (lowest-tilt) mirror; a model with an
active normal_wind/tangential_wind/impaction_retention component ties its noise to
wind direction x each mirror's tilt/azimuth, so every common mirror contributes
independent information and all are used. Evaluation always scores every common
mirror, regardless of what was used for training, so results are comparable across
model types.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd
import heliosoil.base_models as smb
import heliosoil.fitting as smf
import heliosoil.utilities as smu
from heliosoil.horizontal_impaction import ConstantMeanWindDeposition, describe_components, resolve_component_keys
from heliosoil.paper_specific_utilities import regression_performance_stats

MODEL_CLASSES = {"constant_mean": smf.ConstantMeanDeposition, "constant_mean_wind": ConstantMeanWindDeposition, "semi_physical": smf.SemiPhysical}
# constant_mean_wind's parameter names/count depend on wind_components and are read
# from the fitted model's `parameter_names` property instead of hardcoded here.
PARAM_NAMES = {"constant_mean": ["mu_tilde", "sigma_dep"], "semi_physical": ["hrz0", "sigma_dep"]}
MODEL_TYPES = list(MODEL_CLASSES)
# Results subdir for runs whose every mechanism names its own dust channel: they do not
# depend on cfg.dust_type, so they sit outside the per-dust-type subtrees (see
# is_dust_type_independent).
PER_COMPONENT_SUBDIR = "per-component-dust"


@dataclass
class PipelineConfig:
    """Shared, non-cross-validation configuration for both analysis workflows.

    Derived paths (data_dir/file_prefix/parameter_file/site_name) follow the
    heliosoil analysis-script convention for a dataset named by `location`.
    """

    location: str = "mountisa"  # "mountisa" | "carwarp" | "yadnarie" | "qut" | "ablrf" | "wodonga"
    train_experiments: list = field(default_factory=lambda: [0])  # indices (into the sorted file list) used for training
    train_mirrors: Any = None  # None -> model-aware default per run; explicit list overrides every run
    dust_type: str = "PM10"  # "PM10" or "PM2.5", selects the dust distribution in the parameter file
    k_factor: Any = "import"  # None sets equal to 1.0, "import" imports from the file
    number_of_measurements: float = 9.0
    reflectometer_incidence_angle: float = 15  # [deg]
    reflectometer_acceptance_angle: float = 12.5e-3  # [rad]
    second_surf: bool = True  # True: second-surface AOI model, False: first-surface
    daily_average: bool = False  # True: daily-averaged reflectance, False: all measurements
    verbose: bool = False

    @property
    def data_dir(self):
        return f"{smu.get_project_root()}/data/{self.location}/"

    @property
    def file_prefix(self):
        # Soiling data files are named like "{file_prefix}_YYYYMMDD.xlsx"
        return f"soiling_{self.location}"

    @property
    def parameter_file(self):
        return os.path.join(self.data_dir, f"parameters_{self.location}_experiments.xlsx")

    @property
    def site_name(self):
        return os.path.basename(os.path.normpath(self.data_dir))


@dataclass
class LoadedData:
    """The file list, per-campaign training intervals, common mirrors, and the full
    evaluation dataset (all campaigns, every common mirror). Shared by the primary
    fit and every cross-validation fold; scored on all common mirrors regardless of
    what a given fit trained on."""

    files: list
    training_intervals: Any
    all_mirrors: list
    site_name: str
    parameter_file: str
    sim_data_total: Any
    reflect_data_total: Any


def orientation_code(mirror_name):
    """e.g. "ON_M1_T00" -> "N", "OSE_M2_T00" -> "SE" (letters between the leading
    "O" and the first underscore)."""
    return mirror_name.split("_")[0][1:]


def extract(x, ind):
    return [x[ii] for ii in ind]


def uses_wind_variance(model_type, wind_components):
    """Only "constant_mean_wind" with an active wind-driven noise term ties sigma to
    wind direction x mirror orientation; every other configuration (constant_mean,
    semi_physical, or constant_mean_wind with only "gravitational") shares a single
    sigma_dep process across mirrors and must train on one representative mirror."""
    return model_type == "constant_mean_wind" and any(c in wind_components for c in ("normal_wind", "tangential_wind", "impaction_retention"))


def build_model(
    parameter_file: str, model_type: str, wind_components: list, component_dust_types: Any = None, verbose: bool = False
) -> tuple[ConstantMeanWindDeposition | smf.ConstantMeanDeposition | smf.SemiPhysical, str]:
    """Construct a fresh (unfitted) model and its results-folder label."""
    if model_type == "constant_mean_wind":
        model = ConstantMeanWindDeposition(parameter_file, components=wind_components, component_dust_types=component_dust_types, verbose=verbose)
        model_label = model.model_name
    else:
        model = MODEL_CLASSES[model_type](parameter_file, verbose=verbose)
        model_label = model_type
    return model, model_label


def build_runs(model_type, wind_components, wind_component_combos, component_dust_types=None, dust_specs=None):
    """List the (model_type, wind_components, component_dust_types) runs to execute.

    model_type=None sweeps all three model types; for "constant_mean_wind",
    wind_components=None sweeps wind_component_combos, otherwise a single wind
    configuration is used.

    component_dust_types selects the dust channel driving each wind mechanism:
      - None: every mechanism uses the run's cfg.dust_type, so the dust axis is the
        workflow's outer dust-type sweep (the behaviour before per-component channels);
      - a dict {component: spec} (or one spec for all): that fixed assignment;
      - "sweep": every assignment of `dust_specs` to that run's components
        (see component_dust_assignments), which is the combinatorial exploration.
    Only "constant_mean_wind" has components; the other model types always run with
    component_dust_types=None.
    """
    model_types_to_run = MODEL_TYPES if model_type is None else [model_type]
    if component_dust_types == "sweep" and not dust_specs:
        raise ValueError('component_dust_types="sweep" needs dust_specs to sweep over (e.g. available_dust_specs(cfg)).')

    runs = []
    for mt in model_types_to_run:
        if mt != "constant_mean_wind":
            runs.append((mt, None, None))
            continue
        combos = wind_component_combos if wind_components is None else [wind_components]
        for combo in combos:
            if component_dust_types == "sweep":
                runs.extend((mt, combo, assignment) for assignment in component_dust_assignments(combo, dust_specs))
            else:
                runs.append((mt, combo, component_dust_types))
    return runs


def is_dust_type_independent(model_type: str, wind_components: list, component_dust_types: Any) -> bool:
    """True if a run's result does not depend on cfg.dust_type, i.e. every mechanism it
    fits names its own dust channel. Such a run is executed once instead of once per
    swept dust type, and its results live under PER_COMPONENT_SUBDIR."""
    if model_type != "constant_mean_wind" or component_dust_types is None:
        return False
    if isinstance(component_dust_types, str):
        return True
    return all(component_dust_types.get(key) is not None for key in resolve_component_keys(wind_components))


def run_expression(model_type: str, wind_components: list, component_dust_types: Any = None) -> str:
    """One-line description of what a run fits: for constant_mean_wind, the mechanisms
    it sums and the dust channel driving each ("PM17*gravitational +
    (PM10-PM2.5)*tangential_wind"); otherwise just the model type."""
    if model_type != "constant_mean_wind":
        return model_type
    return describe_components(wind_components, component_dust_types)


def resolve_training_mirrors(cfg: PipelineConfig, all_mirrors: list, model_type: str, wind_components: list) -> list[str]:
    """train_mirrors from the config overrides every run; otherwise pick the
    model-aware default (see smu.default_training_mirrors)."""
    if cfg.train_mirrors is not None:
        return cfg.train_mirrors
    return smu.default_training_mirrors(all_mirrors, uses_wind_variance(model_type, wind_components))


def resolve_run_dir(cfg: PipelineConfig, workflow: str, run_name: str | None = None) -> str:
    """Resolve (and create) the results/{workflow}/{site}/{run_name} folder. run_name
    =None -> a "run-yy-mm-dd_hh-mm" timestamp. If the folder already exists, prompt
    before overwriting and abort (SystemExit) on anything but "y". Returns the
    run_name."""
    run_name = run_name or datetime.now().strftime("run-%y-%m-%d_%H-%M")
    run_dir = f"{smu.get_project_root()}/results/{workflow}/{cfg.site_name}/{run_name}"
    if os.path.isdir(run_dir):
        answer = input(f"{run_dir} already exists. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            raise SystemExit("Aborted: run folder already exists.")
    os.makedirs(run_dir, exist_ok=True)
    return run_name


def results_dir_and_label(
    cfg: PipelineConfig, model_type: str, wind_components: list, component_dust_types: Any, workflow: str, run_name: str, subdir: str | None = None
):
    """Build (and create) the model results folder, returning (results_dir, label).

    Path is results/{workflow}/{site}/{run_name}/{label}, with an optional `subdir`
    level inserted before {label} (model_select passes subdir=dust_type to give each
    PM type its own subtree). The label is the model's own name, so a run with
    per-component dust channels is already distinguished by them
    ("constant-mean_PM17*gravitational_PM10-PM2.5*tangential-wind")."""
    _, label = build_model(cfg.parameter_file, model_type, wind_components, component_dust_types, verbose=cfg.verbose)
    parts = [smu.get_project_root(), "results", workflow, cfg.site_name, run_name]
    if subdir is not None:
        parts.append(subdir)
    results_dir = "/".join(str(p) for p in [*parts, label])
    os.makedirs(results_dir, exist_ok=True)
    return results_dir, label


# ------------------------------ dust / PM channels ------------------------------
# A "dust spec" names the airborne-mass channel a model -- or one mechanism of a wind
# model -- is driven by: a single measure ("TSP", "PM17", "PM10", "PM2.5") or the
# difference of two ("PM17-PM10"), which isolates the mass carried by particles between
# the two cutoffs. Weather-column spellings differ across sites (PM2.5 / PM2p5 / PM2_5;
# PM17 / PM_TOT), but SimulationInputs canonicalizes both the columns it loads (into
# dust_concentration_channels) and the dust_type it is given, so the helpers here only
# decide *which* specs a location can support.


def _weather_columns(files: list) -> list[str]:
    """Columns common to every campaign file's Weather sheet, in first-file order."""
    per_file = [list(pd.read_excel(f, sheet_name="Weather", nrows=0).columns) for f in files]
    common = set.intersection(*(set(cols) for cols in per_file)) if per_file else set()
    return [c for c in per_file[0] if c in common] if per_file else []


def available_dust_types(cfg: PipelineConfig, files: list | None = None) -> list[str]:
    """Normalized dust/PM types usable for this location: the PM/TSP columns common to
    every campaign file (first-file order, de-duplicated). Single-element (e.g.
    ["TSP"]) for sites that cannot benefit from a PM-type comparison."""
    if files is None:
        files, *_ = smu.get_training_data(cfg.data_dir, cfg.file_prefix)
    seen, out = set(), []
    for col in _weather_columns(files):
        dt = smu.normalize_dust_name(col)
        if dt is not None and dt not in seen:
            seen.add(dt)
            out.append(dt)
    return out


def available_dust_specs(cfg: PipelineConfig, files: list | None = None, include_differences: bool = True) -> list[str]:
    """Every dust channel a model at this location can be driven by: the measures from
    available_dust_types, followed (by default) by the difference of every pair whose
    cutoffs differ -- "PM17-PM10" is the coarse mass PM10 does not see, "PM10-PM2.5" the
    2.5-10 µm fraction. Differences are what let the wind mechanisms be driven by
    disjoint size bands rather than by nested, strongly correlated ones."""
    dust_types = available_dust_types(cfg, files)
    specs = list(dust_types)
    if include_differences:
        ascending = sorted(dust_types, key=smu.dust_cutoff)
        specs += [f"{major}-{minor}" for minor, major in combinations(ascending, 2) if smu.dust_cutoff(major) > smu.dust_cutoff(minor)]
    return specs


def component_dust_assignments(wind_components: list, dust_specs: list) -> list[dict]:
    """Every way of driving each of a wind model's components with one of `dust_specs`
    -- len(dust_specs) ** n_components assignments, e.g. {"gravitational": "PM17",
    "tangential_wind": "PM10-PM2.5"}."""
    keys = resolve_component_keys(wind_components)
    return [dict(zip(keys, combo)) for combo in product(dust_specs, repeat=len(keys))]


def build_training_inputs(
    cfg: PipelineConfig, data: LoadedData, train_mirrors_run: list, train_exps: list
) -> tuple[smb.SimulationInputs, smb.ReflectanceMeasurements]:
    """Build the SimulationInputs and ReflectanceMeasurements for a training fold.

    The training fold is defined by `train_exps` (indices into the full file list) and
    `train_mirrors_run` (the common mirrors used for training this fold).

    Returns the SimulationInputs and ReflectanceMeasurements trimmed to the training reflectance_data interval (the latter is a subset of the former).
    """
    files_train = extract(data.files, train_exps)
    intervals_train = extract(data.training_intervals, train_exps)

    sim_train_raw = build_sim_inputs(cfg, files_train)
    reflect_train_raw = smb.ReflectanceMeasurements(
        files_train,
        sim_train_raw.time,
        number_of_measurements=cfg.number_of_measurements,
        reflectometer_incidence_angle=cfg.reflectometer_incidence_angle,
        reflectometer_acceptance_angle=cfg.reflectometer_acceptance_angle,
        import_tilts=True,
        imported_column_names=train_mirrors_run,
        verbose=cfg.verbose,
    )
    if cfg.daily_average:
        reflect_train_raw = smu.daily_average(reflect_train_raw, sim_train_raw.time, sim_train_raw.dt)

    sim_train, reflect_train = smu.trim_experiment_data(sim_train_raw, reflect_train_raw, intervals_train)
    sim_train, reflect_train = smu.trim_experiment_data(sim_train, reflect_train, "reflectance_data")

    return sim_train, reflect_train


def build_sim_inputs(cfg: PipelineConfig, files: list) -> smb.SimulationInputs:
    """Construct SimulationInputs for `cfg.dust_type` -- the single place a dust spec
    becomes simulation inputs. SimulationInputs resolves the spec against the weather
    files' particulate-matter columns, so any spelling ("PM2p5") and any difference
    ("PM17-PM10") works provided the underlying columns are present; it also loads every
    other PM column as a channel, which is what per-component dust types draw on."""
    return smb.SimulationInputs(files, k_factors=cfg.k_factor, dust_type=cfg.dust_type, verbose=cfg.verbose)


def load_data(cfg: PipelineConfig):
    """Load the file list, per-campaign training intervals, common mirrors, and the
    full evaluation dataset (all campaigns, every common mirror) shared by the primary
    fit and every cross-validation fold."""
    files, training_intervals, _mirror_name_list, all_mirrors = smu.get_training_data(cfg.data_dir, cfg.file_prefix)

    sim_data_total = build_sim_inputs(cfg, files)
    reflect_data_total = smb.ReflectanceMeasurements(
        files,
        sim_data_total.time,
        number_of_measurements=cfg.number_of_measurements,
        reflectometer_incidence_angle=cfg.reflectometer_incidence_angle,
        reflectometer_acceptance_angle=cfg.reflectometer_acceptance_angle,
        import_tilts=True,
        imported_column_names=all_mirrors,
        verbose=cfg.verbose,
    )
    if cfg.daily_average:
        reflect_data_total = smu.daily_average(reflect_data_total, sim_data_total.time, sim_data_total.dt)
    sim_data_total, reflect_data_total = smu.trim_experiment_data(sim_data_total, reflect_data_total, "reflectance_data")

    return LoadedData(
        files=files,
        training_intervals=training_intervals,
        all_mirrors=all_mirrors,
        site_name=cfg.site_name,
        parameter_file=cfg.parameter_file,
        sim_data_total=sim_data_total,
        reflect_data_total=reflect_data_total,
    )


def fit_and_evaluate(
    cfg: PipelineConfig,
    data: LoadedData,
    model_type: str,
    wind_components: list,
    component_dust_types: Any,
    train_mirrors_run: list,
    train_exps: list,
    test_exps: list,
) -> dict:
    """
    Build a fresh model, fit it on `train_exps` (using `train_mirrors_run`), and score
    it against the full evaluation dataset (all campaigns, all common mirrors) for both
    `train_exps` (in-sample) and `test_exps` (out-of-sample).

    Returns a dict with the fitted model, parameter names/estimates/CIs (original
    scale), the training data used, and both stats dicts (see
    heliosoil.paper_specific_utilities.regression_performance_stats).
    """
    verbose = cfg.verbose
    model, _ = build_model(data.parameter_file, model_type, wind_components, component_dust_types, verbose=verbose)
    sim_train, reflect_train = build_training_inputs(cfg, data, train_mirrors_run, train_exps)

    model.helios_angles(sim_train, reflect_train, second_surface=cfg.second_surf, verbose=verbose)

    ext_weights = None
    if model_type == "semi_physical":
        # The physical model additionally needs Mie extinction weights, computed from
        # this fold's training dust distribution and re-applied (not recomputed) on the
        # full evaluation set below.
        model.helios.compute_extinction_weights(
            sim_train,
            model.loss_model,
            lookup_table_file_folder="extinction_lookup_tables",
            acceptance_bin_width=1.0,  # [mrad] bin width for the extinction lookup table
            verbose=verbose,
        )
        ext_weights = model.helios.extinction_weighting[0].copy()

    log_param_hat, log_param_cov = model.fit_mle(sim_train, reflect_train, verbose=verbose, transform_to_original_scale=False)
    # The parameter transform is model-specific (e.g. constant_mean_wind logs mu_tilde/
    # sigma_dep/sigma_dep_gamma but leaves omega_windward/omega_leeward linear); only
    # model.transform_scale is needed to go from the fitted (possibly log-transformed)
    # scale back to each parameter's natural scale.
    s = smu._std_errors_from_cov(log_param_cov)
    param_ci = log_param_hat + 1.96 * s * np.array([[-1], [1]])
    lower_ci = model.transform_scale(param_ci[0, :])
    upper_ci = model.transform_scale(param_ci[1, :])
    param_hat = model.transform_scale(log_param_hat)
    model.update_model_parameters(param_hat)

    param_names = getattr(model, "parameter_names", PARAM_NAMES.get(model_type))

    model.helios_angles(data.sim_data_total, data.reflect_data_total, second_surface=cfg.second_surf, verbose=verbose)
    if model_type == "semi_physical":
        model = smu.set_extinction_coefficients(model, ext_weights, np.arange(len(data.files)))
    model.predict_soiling_factor(data.sim_data_total, reflectance_data=data.reflect_data_total, verbose=verbose)

    stats_in = regression_performance_stats(model, data.reflect_data_total, train_exps)
    stats_out = regression_performance_stats(model, data.reflect_data_total, test_exps)

    return {
        "model": model,
        "param_names": param_names,
        "param_hat": param_hat,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "log_param_hat": log_param_hat,
        "log_param_cov": log_param_cov,
        "sim_train": sim_train,
        "reflect_train": reflect_train,
        "stats_in": stats_in,
        "stats_out": stats_out,
    }


def compile_results(cfg: PipelineConfig, run_name: str, workflow: str = "model_select") -> None:
    """After a model_selection run, stack every (dust_type, model)
    cross_validation_summary.csv for this site into all_models_kfold_summary.csv,
    sorted within each training-set size by out-of-sample RMSE (best first).

    The run tree is {run_dir}/{dust_type}/{model}/, so every row is tagged with both
    `dust_type` and `model`, letting the compiled table answer "which PM type + model
    generalizes best?". Runs whose mechanisms each name their own dust channel sit under
    dust_type=PER_COMPONENT_SUBDIR instead, with the channels shown in `model` and
    `model_expression`. RMSE (and the other out-of-sample stats) is scored on the
    common evaluation set (all common mirrors, same reflectance targets across dust
    types) for every model, so it is directly comparable across both axes -- unlike a
    training-fit AIC, whose magnitude would track training-mirror/observation count.

    Surfaces per-run fold failures: because a failed fold is a NaN row that the
    NaN-skipping summary means silently drop, a (dust_type, model) that only *looks*
    competitive on its surviving folds is flagged via the stacked `n_failed` column
    and a printed warning. Summaries missing a file are skipped (with a note) rather
    than aborting the whole run.
    """
    print(f"\nCompiling {workflow} results for {cfg.site_name}/{run_name}...")
    run_dir = f"{smu.get_project_root()}/results/{workflow}/{cfg.site_name}/{run_name}"
    if not os.path.isdir(run_dir):
        print(f"no results at {run_dir}; nothing to compile.")
        return

    kfold_frames = []
    for dust_type in sorted(d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))):
        dust_dir = os.path.join(run_dir, dust_type)
        for model_name in sorted(m for m in os.listdir(dust_dir) if os.path.isdir(os.path.join(dust_dir, m))):
            summary_path = os.path.join(dust_dir, model_name, "cross_validation_summary.csv")
            if os.path.exists(summary_path):
                df = pd.read_csv(summary_path)
                df.insert(0, "model", model_name)
                df.insert(0, "dust_type", dust_type)
                kfold_frames.append(df)
            else:
                print(f"{cfg.site_name}/{dust_type}/{model_name}: no cross_validation_summary.csv; skipped.")

    if not kfold_frames:
        print(f"{cfg.site_name}: no cross_validation_summary.csv files found.")
        return

    kfold_df = pd.concat(kfold_frames, ignore_index=True)
    if "RMSE_mean" in kfold_df.columns:
        # Best (lowest out-of-sample RMSE) (dust_type, model) first within each size.
        kfold_df = kfold_df.sort_values(["n_train_campaigns", "RMSE_mean"]).reset_index(drop=True)

    if "n_failed" in kfold_df.columns:
        failed = kfold_df.groupby(["dust_type", "model"])["n_failed"].sum()
        for (dust_type, model_name), n in failed[failed > 0].items():
            print(f"WARNING: {cfg.site_name}/{dust_type}/{model_name} had {int(n)} failed fold(s).")

    out_path = os.path.join(run_dir, "all_models_kfold_summary.csv")
    kfold_df.to_csv(out_path, index=False, float_format="%.3g")
    print(f"wrote {out_path} ({len(kfold_frames)} rows).")
