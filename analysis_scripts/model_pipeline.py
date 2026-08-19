"""
Shared kernel for the HelioSoil analysis workflows.

Holds the configuration, data loading, model construction, and fit/evaluate logic
shared by the two workflow scripts:

  - simulate.py        -- fit each model on the training campaign(s) (or, by default,
    leave-one-campaign-out over every campaign) and report fitted parameters,
    performance statistics, and per-campaign plots.
  - model_selection.py -- leave-n-campaigns-out cross-validation sweeping the
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

Fit method
----------
cfg.fit_method picks how a model's parameters are estimated. "mle" (the default) maximizes
the likelihood over every parameter at once. "ls" fits the mean parameters by least squares,
and every model type supports it: constant_mean and constant_mean_wind are affine in their
means, so it is one bounded linear solve with no starting point and no local optima
(heliosoil.fitting.AffineMeanLeastSquares); semi_physical is not affine in hrz0 -- it enters
through the deposition-velocity physics -- so it uses a bounded scalar search over the same
sum of squares instead (SemiPhysical.fit_ls). resolve_fit_method still falls back to MLE for
any model type whose class lacks fit_ls, so a future one cannot break --model-type all.

"ls" fits no noise parameters, since the sum of squares does not depend on them, and that
propagates in two places. Training-mirror selection stops restricting to one representative
mirror -- that rule is about not over-counting a shared deposition-noise process, which an
"ls" run does not have -- so every common mirror is trained on (see
resolve_training_mirrors). And the fitted model carries no prediction variance, so the
reflectance figures draw the mean prediction without a shaded prediction interval
(heliosoil.paper_specific_utilities._prediction_variance). The reported parameters are
correspondingly the mean ones only.

Dust channels
-------------
A run is defined by (model_type, wind_components, component_dust_types). The last of
these drives each wind mechanism with its own dust spec -- a single measure ("PM17",
"PM10") or the difference of two ("PM17-PM10", the mass coarser than 10 µm) -- so a
model can settle whole-distribution mass gravitationally while scouring only a narrow
size band tangentially. It defaults to None, meaning every mechanism uses cfg.dust_type
and the dust axis is the workflow's outer dust-type sweep. See build_runs,
available_dust_specs and component_dust_assignments.

Training-mirror selection follows the VARIANCE MODEL (see resolve_training_mirrors).
"independent" treats the mirrors as carrying independent information, which they do not, so
it trains on a single lowest-tilt representative as crude protection. "shared_kappa" and
"per_mechanism" model the between-mirror correlation directly and train on every common
mirror. An "ls" run fits no noise parameters at all and likewise uses every mirror.
Evaluation always scores every common mirror, so runs stay comparable whatever was trained
on.
"""

import os
import sys
import copy
import json
import logging
import platform
import threading
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd
import heliosoil
import heliosoil.base_models as smb
import heliosoil.fitting as smf
import heliosoil.utilities as smu
from heliosoil.horizontal_impaction import COMPONENT_KEYS, ConstantMeanWindDeposition, describe_components, poorly_identified_components, resolve_component_keys
from heliosoil.paper_specific_utilities import regression_performance_stats

# Serializes the semi_physical extinction-lookup-table step across worker threads:
# compute_extinction_weights writes into one shared "extinction_lookup_tables" folder on a
# cache miss (see heliosoil.base_models), so concurrent folds must not generate it at once.
_EXTINCTION_LOCK = threading.Lock()

MODEL_CLASSES = {"constant_mean": smf.ConstantMeanDeposition, "constant_mean_wind": ConstantMeanWindDeposition, "semi_physical": smf.SemiPhysical}
# constant_mean_wind's parameter names/count depend on wind_components and are read
# from the fitted model's `parameter_names` property instead of hardcoded here.
PARAM_NAMES = {"constant_mean": ["mu_tilde", "sigma_dep"], "semi_physical": ["hrz0", "sigma_dep"]}
MODEL_TYPES = list(MODEL_CLASSES)
# How a model's parameters are estimated (see resolve_fit_method / fit_and_evaluate).
FIT_METHODS = ("mle", "ls")
# Results subdir for runs whose every mechanism names its own dust channel: they do not
# depend on cfg.dust_type, so they sit outside the per-dust-type subtrees (see
# is_dust_type_independent).
PER_COMPONENT_SUBDIR = "per-component-dust"


class FoldStitchedModel:
    """A model-shaped view of the cross-validation's out-of-sample predictions: every campaign's
    prediction taken from the fold that held that campaign out.

    plot_for_paper, plot_reflectance_by_tilt and regression_performance_stats each take a single
    fitted model, so the only way to show or score the whole site out-of-sample at once is to hand
    them one object whose per-campaign predictions come from different fits. Everything but the
    predictions is fold-invariant (every fold ends by re-running helios_angles on the full
    evaluation data), so tilt/azimuth/nominal_reflectance are taken from `template`, an arbitrary
    fold's fitted model.

    Only meaningful when each campaign was held out exactly once, i.e. the leave-one-campaign-out
    schedule (n_train = C-1). At smaller training sizes a campaign is held out by several folds and
    has several competing predictions, which this mapping cannot represent.

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


@dataclass
class PipelineConfig:
    """Shared site/physics configuration for both analysis workflows.

    Derived paths (data_dir/file_prefix/parameter_file/site_name) follow the
    heliosoil analysis-script convention for a dataset named by `location`.

    `train_experiments` is the only fold-related field: it names `simulate`'s training
    campaigns, or None for its leave-one-campaign-out default. `model_selection` ignores
    it and enumerates its own fold schedule.
    """

    location: str = "mountisa"  # "mountisa" | "carwarp" | "yadnarie" | "qut" | "ablrf" | "wodonga"
    data_subdir: str = ""
    # Indices (into the sorted file list) used for training. None -> leave-one-campaign-out
    # cross-validation: `simulate` fits one model per campaign, each trained on every other one.
    train_experiments: list | None = None
    train_mirrors: Any = None  # None -> model-aware default per run; explicit list overrides every run
    # "mle" maximizes the likelihood over every parameter; "ls" fits the mean parameters by linear
    # least squares and profiles the likelihood over the noise parameters at that solution. Only
    # model types whose class provides fit_ls honour it -- see resolve_fit_method.
    fit_method: str = "mle"
    # How the deposition noise is correlated across mirrors (heliosoil.fitting's
    # set_variance_model). "independent" is the historical model and stays the default, so
    # an unchanged command reproduces an unchanged number; "shared_kappa" estimates one
    # common fraction across mechanisms and "per_mechanism" one each. Only the MLE path
    # sees this -- least squares fits no noise parameters at all.
    variance_model: str = "independent"
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
        # os.path.join drops an empty component, so the default reproduces the old
        # data/<location>/ path exactly rather than emitting a doubled separator.
        return os.path.join(smu.get_project_root(), "data", self.data_subdir, self.location) + os.sep

    @property
    def file_prefix(self):
        # Soiling data files are named like "{file_prefix}_YYYYMMDD.xlsx"
        return f"soiling_{self.location}"

    @property
    def parameter_file(self):
        return os.path.join(self.data_dir, f"parameters_{self.location}_experiments.xlsx")

    @property
    def site_name(self):
        # The results folder is keyed on the SITE, not on which copy of its data was read
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


def build_runs(model_type, combos, component_dust_types=None, dust_specs=None):
    """List the (model_type, wind_components, component_dust_types) runs to execute.

    model_type=None sweeps all three model types. `combos` is the list of component
    combinations to fit as "constant_mean_wind" models -- ALL_COMPONENT_COMBOS for the
    exploratory sweep, or a single-element list for one named model.

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
    """train_mirrors from the config overrides every run; otherwise pick the default for this
    run's fit method and model.

    The single-representative-mirror default exists because a shared sigma_dep is one
    stochastic process across mirrors, and fitting it on several correlated mirrors overstates
    the independent information available. A least-squares fit estimates no sigma at all, so
    that argument does not apply to it: every mirror contributes a genuine row to the design
    matrix, and all of them are used."""
    if cfg.train_mirrors is not None:
        return cfg.train_mirrors
    if resolve_fit_method(cfg, model_type) == "ls":
        return list(all_mirrors)
    if cfg.variance_model != "independent":
        # The likelihood models the between-mirror correlation directly, so there is nothing
        # left for a single-representative-mirror rule to protect against: every mirror
        # contributes, and the common component stops the shared part being counted once per
        # mirror. This is the plan's D6, delivered on the path that earns it.
        return list(all_mirrors)
    # "independent" believes the mirrors carry independent information. They do not -- for a
    # wind model either, which is what the deleted uses_wind_variance used to assert. One
    # representative mirror is the only protection this likelihood has.
    #
    # One mirror cannot identify a multi-mechanism model, and that is not worked around here:
    # it is the cost of the independent-mirror likelihood, and the fix is to model the
    # correlation with --variance-model shared_kappa rather than to widen the training set
    # under a likelihood that would then over-count it. What IS done is to say so.
    mirrors = smu.default_training_mirrors(all_mirrors)
    if model_type == "constant_mean_wind":
        tilt = smu.parse_mirror_tilt(mirrors[0])
        starved = poorly_identified_components(wind_components, tilt)
        if starved:
            named = ", ".join(f"{key} (geometry factor {factor:.3g})" for key, factor in starved)
            smu.logger.warning(
                f"Training on the single mirror {mirrors[0]} at tilt {tilt} deg, which carries essentially no information about: {named}. "
                "Those coefficients are not identified by this fit and will report whatever the optimiser started from. "
                "This is what --variance-model independent costs on a multi-mechanism model; use shared_kappa to train on every mirror instead."
            )
    return mirrors


def resolve_fit_method(cfg: PipelineConfig, model_type: str) -> str:
    """The fitting routine `model_type` will actually be fitted with under `cfg`.

    The capability is read from the model class rather than hardcoded: a class providing fit_ls
    supports least squares (all three do today, by two different routes -- see the module
    docstring), and any other type falls back to fit_mle so that --model-type all still runs
    end to end. Callers that report the method (simulate's header, model_selection's summary
    column) go through here too, so what is reported is what was run."""
    if cfg.fit_method not in FIT_METHODS:
        raise ValueError(f"Unknown fit_method {cfg.fit_method!r}; choose from {list(FIT_METHODS)}.")
    if cfg.fit_method == "ls" and hasattr(MODEL_CLASSES[model_type], "fit_ls"):
        return "ls"
    return "mle"


def prompt_yes_no(question: str) -> bool | None:
    """Ask `question` on the terminal: True/False for an answer, None when there is nobody to
    ask -- no TTY (scripted/CI), or a stdin that claims to be one but is already at EOF (a
    detached/redirected stdin on Windows does exactly this). Callers decide what "nobody to
    ask" means for them; what neither may do is block a scripted run on stdin or, worse, die
    on EOFError partway through a long run."""
    if sys.stdin is None or not sys.stdin.isatty():
        return None
    try:
        return input(f"{question} ").strip().lower() == "y"
    except EOFError:
        return None


def resolve_run_dir(cfg: PipelineConfig, workflow: str, run_name: str | None = None, force: bool = False) -> str:
    """Resolve (and create) the results/{workflow}/{site}/{run_name} folder. run_name
    =None -> a "run-yy-mm-dd_hh-mm" timestamp. Returns the run_name.

    If the folder already exists: `force` overwrites it silently; otherwise, on an
    interactive terminal, prompt before overwriting and abort (SystemExit) on anything
    but "y"; with no TTY (scripted/CI), abort with a message pointing at --force so the
    run never blocks on stdin."""
    run_name = run_name or datetime.now().strftime("run-%y-%m-%d_%H-%M")
    run_dir = f"{smu.get_project_root()}/results/{workflow}/{cfg.site_name}/{run_name}"
    if os.path.isdir(run_dir) and not force:
        answer = prompt_yes_no(f"{run_dir} already exists. Overwrite? [y/N]")
        if answer is None:
            raise SystemExit(f"Aborted: {run_dir} already exists. Re-run with --force to overwrite or choose a different --run-name.")
        if not answer:
            raise SystemExit("Aborted: run folder already exists.")
    os.makedirs(run_dir, exist_ok=True)
    return run_name


def run_dir_path(cfg: PipelineConfig, workflow: str, run_name: str) -> str:
    """Absolute path of the top-level run folder resolve_run_dir created/returned."""
    return f"{smu.get_project_root()}/results/{workflow}/{cfg.site_name}/{run_name}"


def wind_component_combos(pool: list, min_size: int = 1, max_size: int | None = None) -> list[list]:
    """Every unique, canonically-ordered, non-empty subset of the component `pool`.

    resolve_component_keys validates the pool and returns its keys in canonical order with
    duplicates removed, so itertools.combinations yields exactly the unique canonically-ordered
    subsets -- e.g. a 5-component pool gives 2**5 - 1 = 31 combos. min_size/max_size bound the
    subset size (defaults: every non-empty subset). This replaces the hand-maintained
    WIND_COMPONENT_COMBOS list; feed the result to build_runs as wind_component_combos."""
    keys = resolve_component_keys(pool)
    max_size = max_size or len(keys)
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}.")
    return [list(combo) for r in range(min_size, max_size + 1) for combo in combinations(keys, r)]


# The component exploration both workflows sweep: every unique non-empty subset of the full
# component vocabulary (2**5 - 1 = 31). Fixed rather than configurable -- narrowing the pool
# only ever removed models from the comparison, and the CV cost is dominated by the dust axis.
ALL_COMPONENT_COMBOS = wind_component_combos(COMPONENT_KEYS)

# Short component labels for the progress bar (see abbreviate_run).
COMPONENT_ABBREV = {"gravitational": "Grav", "turbulent_wind": "Turb", "normal_wind": "Norm", "tangential_wind": "Tan", "impaction_retention": "Imp"}
assert set(COMPONENT_ABBREV) == set(COMPONENT_KEYS), "COMPONENT_ABBREV must cover exactly heliosoil.horizontal_impaction.COMPONENT_KEYS."


def abbreviate_run(model_type: str, wind_components: Any, component_dust_types: Any = None, subdir: str | None = None) -> str:
    """Compact one-line label for tqdm progress, e.g. "PM10|Grav+Norm",
    "Grav(PM17)+Tan(PM10-PM2.5)", "ConstMean", "SemiPhys"."""
    if model_type == "constant_mean":
        label = "ConstMean"
    elif model_type == "semi_physical":
        label = "SemiPhys"
    elif model_type == "constant_mean_wind":
        keys = resolve_component_keys(wind_components)
        if isinstance(component_dust_types, dict):
            # A key mapped to None names no channel of its own (it uses cfg.dust_type), so it
            # is left unqualified -- same as one absent from the dict.
            parts = [
                f"{COMPONENT_ABBREV.get(k, k)}({component_dust_types[k]})" if component_dust_types.get(k) is not None else COMPONENT_ABBREV.get(k, k)
                for k in keys
            ]
        elif isinstance(component_dust_types, str) and component_dust_types != "sweep":
            parts = [f"{COMPONENT_ABBREV.get(k, k)}({component_dust_types})" for k in keys]
        else:
            parts = [COMPONENT_ABBREV.get(k, k) for k in keys]
        label = "+".join(parts)
    else:
        label = str(model_type)
    if subdir and subdir not in (None, PER_COMPONENT_SUBDIR):
        return f"{subdir}|{label}"
    return label


def write_run_metadata(run_dir: str, metadata: dict) -> None:
    """Write run_config.json into the run folder: the invocation plus the environment it ran
    in (heliosoil version, python/platform, timestamp), so results can be traced back to the
    exact settings and code version. `default=str` keeps any non-JSON value (a k_factor
    sentinel, stray callables) from crashing the dump."""
    payload = dict(metadata)
    payload.setdefault("heliosoil_version", heliosoil.__version__)
    payload.setdefault("python_version", platform.python_version())
    payload.setdefault("platform", platform.platform())
    payload.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def open_run_log(run_dir: str) -> tuple[logging.Logger, Any]:
    """Route workflow progress (INFO) and otherwise-suppressed warnings (WARNING) to
    {run_dir}/run.log. Returns (analysis_logger, close):
      - analysis_logger: a dedicated, non-propagating "heliosoil.analysis" logger the
        workflow writes its header/per-fold lines to -- they land in run.log only, never
        the console (so they can't corrupt a tqdm bar);
      - close(): detaches and closes the file handler; call it in a finally.
    The same handler is also attached to the "heliosoil" and captured-warnings ("py.warnings")
    loggers so their WARNING+ records are persisted alongside the fold detail."""
    handler = logging.FileHandler(os.path.join(run_dir, "run.log"), mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    analysis_logger = logging.getLogger("heliosoil.analysis")
    analysis_logger.setLevel(logging.INFO)
    analysis_logger.propagate = False  # keep INFO fold detail off the console handler

    logging.captureWarnings(True)
    attached = [logging.getLogger(name) for name in ("heliosoil.analysis", "heliosoil", "py.warnings")]
    for lg in attached:
        lg.addHandler(handler)

    def close():
        for lg in attached:
            lg.removeHandler(handler)
        handler.close()

    return analysis_logger, close


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


DUST_MODES = ("all", "sweep")


def resolve_dust_selection(cfg: PipelineConfig, dust: str, files: list | None = None) -> tuple[list[str], str]:
    """Turn the workflow's single --dust argument into (dust_types, mode).

    "all" and "sweep" both expand to every measure the site has (mode carries which one was
    asked for -- "sweep" additionally explores per-component channels downstream); anything
    else names one measure, which is canonicalized ("PMT" -> "PM17", "pm2p5" -> "PM2.5") so
    the results-folder name always matches the spelling the model reports, and checked
    against the site's measures so a typo fails before any fitting rather than producing an
    empty subtree."""
    dust_types = available_dust_types(cfg, files)
    mode = str(dust).strip().lower()
    if mode in DUST_MODES:
        return dust_types, mode

    name = smu.normalize_dust_name(dust)
    if name is None:
        raise ValueError(f"--dust {dust!r} is neither a mode {DUST_MODES} nor a particulate-matter measure (expected TSP or PM<number>).")
    if name not in dust_types:
        raise ValueError(f"{cfg.location!r} has no {name} data; its measures are {dust_types}. Use one of those, 'all', or 'sweep'.")
    return [name], "single"


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

    The fit is cfg.fit_method, resolved per model type (see resolve_fit_method) and returned
    as "fit_method" so callers report what was actually run rather than what was asked for.

    Returns a dict with the fitted model, parameter names/estimates/CIs (original
    scale), the training data used, and both stats dicts (see
    heliosoil.paper_specific_utilities.regression_performance_stats).
    """
    verbose = cfg.verbose
    model, _ = build_model(data.parameter_file, model_type, wind_components, component_dust_types, verbose=verbose)
    sim_train, reflect_train = build_training_inputs(cfg, data, train_mirrors_run, train_exps)

    model.helios_angles(sim_train, reflect_train, second_surface=cfg.second_surf, verbose=verbose)
    # Before the fit, so that parameter_names (and therefore every downstream report) already
    # carries whatever common fractions this run estimates.
    model.set_variance_model(cfg.variance_model)

    ext_weights = None
    if model_type == "semi_physical":
        # The physical model additionally needs Mie extinction weights, computed from
        # this fold's training dust distribution and re-applied (not recomputed) on the
        # full evaluation set below. The lock serializes the shared-folder lookup-table
        # generation so parallel folds never race on writing it (see _EXTINCTION_LOCK).
        with _EXTINCTION_LOCK:
            model.helios.compute_extinction_weights(
                sim_train,
                model.loss_model,
                lookup_table_file_folder="extinction_lookup_tables",
                acceptance_bin_width=1.0,  # [mrad] bin width for the extinction lookup table
                verbose=verbose,
            )
        ext_weights = model.helios.extinction_weighting[0].copy()

    # Both fits return (estimate, covariance) on the model's own fitted scale, so everything
    # downstream -- the transform back, the CIs, the CSVs -- is identical either way. They
    # differ in *which* parameters they estimate: least squares fits the means only and leaves
    # the model with no noise process, so its vector is the shorter mean_parameter_names one.
    fit_method = resolve_fit_method(cfg, model_type)
    if fit_method == "ls":
        log_param_hat, log_param_cov = model.fit_ls(sim_train, reflect_train, verbose=verbose, transform_to_original_scale=False)
        param_names = model.mean_parameter_names
    else:
        log_param_hat, log_param_cov = model.fit_mle(sim_train, reflect_train, verbose=verbose, transform_to_original_scale=False)
        param_names = getattr(model, "parameter_names", PARAM_NAMES.get(model_type))
    # The parameter transform is model-specific (e.g. constant_mean_wind logs mu_tilde/
    # sigma_dep/sigma_dep_gamma but leaves omega_windward/omega_leeward linear); only
    # model.transform_scale is needed to go from the fitted (possibly log-transformed)
    # scale back to each parameter's natural scale.
    s = smu._std_errors_from_cov(log_param_cov)
    param_ci = log_param_hat + 1.96 * s * np.array([[-1], [1]])
    lower_ci = np.asarray(model.transform_scale(param_ci[0, :]), dtype=float)
    upper_ci = np.asarray(model.transform_scale(param_ci[1, :]), dtype=float)
    # A Wald interval is not defined where an estimate sits on its bound, and the symptom
    # here is an interval that has diverged: a non-finite standard error, or one so wide that
    # exp() of the endpoints overflows to inf and underflows to 0. Reporting "[0, inf]" in
    # fitted_parameters.csv reads as a measurement; blank says what is true, that no interval
    # is available (plan D9).
    estimate = np.asarray(model.transform_scale(log_param_hat), dtype=float)
    diverged = ~np.isfinite(s) | ~np.isfinite(upper_ci) | ((lower_ci == 0.0) & (estimate != 0.0))
    lower_ci = np.where(diverged, np.nan, lower_ci)
    upper_ci = np.where(diverged, np.nan, upper_ci)
    param_hat = estimate
    # update_model_parameters zips against the full parameter list, so a mean-only vector sets
    # exactly the means and leaves fit_ls's sigmas at None -- no prediction variance is
    # resurrected by writing the estimate back.
    model.update_model_parameters(param_hat)

    model.helios_angles(data.sim_data_total, data.reflect_data_total, second_surface=cfg.second_surf, verbose=verbose)
    if model_type == "semi_physical":
        model = smu.set_extinction_coefficients(model, ext_weights, np.arange(len(data.files)))
    model.predict_soiling_factor(data.sim_data_total, reflectance_data=data.reflect_data_total, verbose=verbose)

    stats_in = regression_performance_stats(model, data.reflect_data_total, train_exps)
    stats_out = regression_performance_stats(model, data.reflect_data_total, test_exps)

    return {
        "model": model,
        "fit_method": fit_method,
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
    sorted globally by out-of-sample MAE (lowest first).

    The run tree is {run_dir}/{dust_type}/{model}/, so every row is tagged with both
    `dust_type` and `model`, letting the compiled table answer "which PM type + model
    generalizes best?". Runs whose mechanisms each name their own dust channel sit under
    dust_type=PER_COMPONENT_SUBDIR instead, with the channels shown in `model` and
    `model_expression`. MAE (a daily-soiling-rate error; and the other out-of-sample stats)
    is scored on the common evaluation set (all common mirrors, same reflectance targets
    across dust types) for every model, so it is directly comparable across both axes --
    unlike a training-fit AIC, whose magnitude would track training-mirror/observation count.

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
    if "MAE_pooled" in kfold_df.columns:
        # Global ranking: best (lowest pooled out-of-sample MAE) first across every (dust_type,
        # model, n_train) row, with any row that had a failed fold pushed to the very bottom
        # regardless of its MAE (a strong-looking mean on surviving folds is not trustworthy).
        #
        # Ranked on the POOLED MAE, which weights every held-out residual equally and is the
        # number simulate.py quotes, rather than the unweighted mean of per-fold MAEs -- the two
        # disagree whenever folds carry different observation counts, and ranking on one while
        # reporting the other is how a model selected here could fail to be the best one there.
        # Only the leave-one-campaign-out rows carry it (see cross_validate_run), so the smaller
        # training sizes sort below them as diagnostics rather than competing for the top spot.
        has_failed = kfold_df["n_failed"] > 0 if "n_failed" in kfold_df.columns else False
        kfold_df = (
            kfold_df.assign(_has_failed=has_failed)
            .sort_values(["_has_failed", "MAE_pooled"], na_position="last")
            .drop(columns="_has_failed")
            .reset_index(drop=True)
        )

    if "n_failed" in kfold_df.columns:
        failed = kfold_df.groupby(["dust_type", "model"])["n_failed"].sum()
        for (dust_type, model_name), n in failed[failed > 0].items():
            print(f"WARNING: {cfg.site_name}/{dust_type}/{model_name} had {int(n)} failed fold(s).")

    out_path = os.path.join(run_dir, "all_models_kfold_summary.csv")
    kfold_df.to_csv(out_path, index=False, float_format="%.3g")
    print(f"wrote {out_path} ({len(kfold_frames)} rows).")
