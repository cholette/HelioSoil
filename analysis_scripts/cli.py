"""
Unified command-line interface for the HelioSoil analysis workflows.

Three subcommands:

  simulate    single-campaign fit + report            -> simulate.run
  select      leave-n-out cross-validation sweep      -> model_selection.run
  experiment  per-campaign weather + soiling summary  -> experiment.run

Run from the repository root:

    python -m analysis_scripts.cli simulate   --help
    python -m analysis_scripts.cli select     --help
    python -m analysis_scripts.cli experiment --help

`simulate` and `select` share the site/physics and run-control options and differ only in how the
model and its dust channels are chosen; `experiment` characterises the site rather than a model, so
it takes only --location and the run-control options.

Both model workflows take a --model expression in the notation the workflows report, e.g.

    --model "PM2.5*tangential_wind + (PM10-PM2.5)*turbulent_wind + PMT*gravitational"

`simulate` fits that one model (defaulting to the library's gravitational + normal_wind). `select`
compares models, so --model is optional there: without it, every unique non-empty subset of the
wind-component vocabulary is compared. The fold schedule is fixed either way, leaving --dust as
the other choice -- one measure, "all", or "sweep", which drives each of --model's mechanisms
with its own channel and therefore requires one (see model_selection).

Shared-option defaults come from model_pipeline.PipelineConfig, so the dataclass is the single
source of truth.
"""

import os
import sys
import argparse
import logging
import dataclasses

import matplotlib

# Headless, non-interactive backend: the workflows only savefig (never show), and `select`
# fits folds on worker threads. A GUI backend (e.g. TkAgg) creates Tk objects whose __del__
# may run in a worker thread, which raises "main thread is not in main loop" and can abort the
# process ("Tcl_AsyncDelete ... wrong thread"). Agg has no such objects. Must precede any
# pyplot import -- heliosoil's package __init__ imports pyplot transitively.
matplotlib.use("Agg")

from heliosoil.utilities import configure_logging  # noqa: E402

from . import experiment  # noqa: E402
from . import model_pipeline as mp  # noqa: E402
from . import model_selection  # noqa: E402
from . import simulate  # noqa: E402

MODEL_TYPE_CHOICES = ["constant_mean", "constant_mean_wind", "semi_physical", "all"]


def _k_factor(value: str):
    """--k-factor: 'import' (from the parameter file), 'none' (-> None, treated as 1.0
    downstream), or a float."""
    v = value.strip().lower()
    if v == "import":
        return "import"
    if v == "none":
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--k-factor must be 'import', 'none', or a float, got {value!r}")


def _add_location(group, defaults) -> None:
    """--location is the one site option every subcommand takes (experiment takes only it), so it is
    registered from one place rather than duplicated across the parent parsers."""
    group.add_argument("--location", default=defaults.location, help='dataset name, e.g. "mountisa", "carwarp", "yadnarie" (default: %(default)s)')


def build_parser() -> argparse.ArgumentParser:
    defaults = mp.PipelineConfig()
    parser = argparse.ArgumentParser(prog="analysis_scripts.cli", description="HelioSoil analysis workflows (simulate / select / experiment).")

    # --- shared options, in three parents so each subcommand takes only what applies to it:
    # simulate/select get all three, experiment only the site name and run control. ---
    common_location = argparse.ArgumentParser(add_help=False)
    _add_location(common_location.add_argument_group("data / site"), defaults)

    common_site = argparse.ArgumentParser(add_help=False)
    site = common_site.add_argument_group("data / site")
    _add_location(site, defaults)
    site.add_argument(
        "--k-factor", type=_k_factor, default=defaults.k_factor, metavar="{import,none,FLOAT}", help="deposition scaling (default: %(default)s)"
    )
    site.add_argument("--number-of-measurements", type=float, default=defaults.number_of_measurements, help="(default: %(default)s)")
    site.add_argument(
        "--reflectometer-incidence-angle", type=float, default=defaults.reflectometer_incidence_angle, help="[deg] (default: %(default)s)"
    )
    site.add_argument(
        "--reflectometer-acceptance-angle", type=float, default=defaults.reflectometer_acceptance_angle, help="[rad] (default: %(default)s)"
    )
    surface_default = "second" if defaults.second_surf else "first"
    site.add_argument("--surface", choices=["first", "second"], default=surface_default, help="AOI model surface (default: %(default)s)")
    site.add_argument(
        "--daily-average",
        action=argparse.BooleanOptionalAction,
        default=defaults.daily_average,
        help="daily-average the reflectance targets (default: %(default)s)",
    )
    site.add_argument(
        "--train-mirrors", nargs="*", default=defaults.train_mirrors, metavar="MIRROR", help="explicit training mirrors; default: model-aware per run"
    )

    common_model = argparse.ArgumentParser(add_help=False)
    model = common_model.add_argument_group("model")
    model.add_argument(
        "--model-type",
        choices=MODEL_TYPE_CHOICES,
        default="all",
        help='"all" runs constant_mean, constant_mean_wind, semi_physical (default: %(default)s)',
    )
    model.add_argument(
        "--model",
        default=None,
        metavar="EXPRESSION",
        help='the constant_mean_wind model, e.g. "PM2.5*tangential_wind + (PM10-PM2.5)*turbulent_wind"; a bare mechanism uses the run\'s dust type. '
        "Default: fit -> gravitational + normal_wind; select -> compare every component combination. Required by select's --dust sweep",
    )

    common_run = argparse.ArgumentParser(add_help=False)
    run = common_run.add_argument_group("run control")
    run.add_argument("--run-name", default=None, help="results subfolder name (default: run-yy-mm-dd_hh-mm timestamp)")
    run.add_argument("--force", action="store_true", help="overwrite an existing run folder, and confirm select's dust sweep, without prompting")
    run.add_argument("--verbose", action="store_true", default=defaults.verbose, help="INFO-level HelioSoil logging")

    # --- subcommands ---
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser(
        "simulate",
        parents=[common_site, common_model, common_run],
        help="fit each model on the training campaign(s) and report",
        description="Single-campaign fit + report workflow.",
    )
    p_sim.add_argument("--dust-type", default=defaults.dust_type, help='dust channel driving the model, e.g. "PM10", "PM2.5" (default: %(default)s)')
    p_sim.add_argument(
        "--train-experiments",
        nargs="+",
        type=int,
        default=defaults.train_experiments,
        metavar="INDEX",
        help="campaign indices to train on (default: %(default)s)",
    )
    p_sim.set_defaults(func=_run_simulate)

    p_sel = sub.add_parser(
        "select",
        parents=[common_site, common_model, common_run],
        help="leave-n-campaigns-out cross-validation sweep",
        description="Model- and dust-type-comparison cross-validation workflow.",
    )
    p_sel.add_argument(
        "--dust",
        default="all",
        metavar="{TYPE,all,sweep}",
        help='one measure ("PM10"), "all" (every measure the site has), or "sweep" (every per-component dust channel assignment over --model\'s mechanisms) (default: %(default)s)',
    )
    p_sel.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="worker threads for parallel fold fits; 1 disables parallelism (default: %(default)s)",
    )
    p_sel.set_defaults(func=_run_select)

    # No model or physics options: the weather summary depends only on the site and on each
    # campaign's reflectance-measurement window, which none of them influence.
    p_exp = sub.add_parser(
        "experiment",
        parents=[common_location, common_run],
        help="summarise the weather and the measured soiling of each campaign",
        description="Experimental location assessment workflow: per-campaign weather and measured-soiling figures + summary tables.",
    )
    p_exp.set_defaults(func=_run_experiment)

    return parser


def _build_config(args) -> mp.PipelineConfig:
    # `select` overwrites dust_type per pass from its own --dust selection, so it does not
    # take one; PipelineConfig's default stands in until then.
    return mp.PipelineConfig(
        location=args.location,
        train_experiments=list(getattr(args, "train_experiments", None) or [0]),
        train_mirrors=args.train_mirrors,
        dust_type=getattr(args, "dust_type", None) or mp.PipelineConfig.dust_type,
        k_factor=args.k_factor,
        number_of_measurements=args.number_of_measurements,
        reflectometer_incidence_angle=args.reflectometer_incidence_angle,
        reflectometer_acceptance_angle=args.reflectometer_acceptance_angle,
        second_surf=(args.surface == "second"),
        daily_average=args.daily_average,
        verbose=args.verbose,
    )


def _run_metadata(args, cfg: mp.PipelineConfig, **extra) -> dict:
    """Reproducibility snapshot written to run_config.json (versions/timestamp added by
    write_run_metadata). Drop the subparser dispatch func so the dict stays serializable."""
    args_dict = {k: v for k, v in vars(args).items() if k != "func"}
    return {"argv": sys.argv, "command": args.command, "args": args_dict, "config": dataclasses.asdict(cfg), **extra}


def _run_simulate(args) -> None:
    cfg = _build_config(args)
    simulate.run(
        cfg,
        model_type=None if args.model_type == "all" else args.model_type,
        model_expression=args.model,
        run_name=args.run_name,
        force=args.force,
        run_metadata=_run_metadata(args, cfg),
    )


def _run_select(args) -> None:
    cfg = _build_config(args)
    model_selection.run(
        cfg,
        model_type=None if args.model_type == "all" else args.model_type,
        model_expression=args.model,
        dust=args.dust,
        run_name=args.run_name,
        force=args.force,
        jobs=args.jobs,
        run_metadata=_run_metadata(args, cfg, n_component_combos=1 if args.model else len(mp.ALL_COMPONENT_COMBOS)),
    )


def _run_experiment(args) -> None:
    # Builds its config directly rather than through _build_config: this workflow takes none of the
    # model/physics options, and it overrides dust_type/k_factor itself (see experiment._prepare_config).
    cfg = mp.PipelineConfig(location=args.location, verbose=args.verbose)
    experiment.run(cfg, run_name=args.run_name, force=args.force, run_metadata=_run_metadata(args, cfg))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Route all HelioSoil output through the "heliosoil" logger: verbose -> INFO (progress),
    # else WARNING (only genuine warnings/failures).
    configure_logging(logging.INFO if args.verbose else logging.WARNING)
    try:
        args.func(args)
    except ValueError as err:
        # Configuration errors raised by the workflows (an unusable --dust or --model) are
        # the user's mistake, not a crash: report them like any other CLI usage error.
        raise SystemExit(f"error: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
