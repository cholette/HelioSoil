"""
Unit tests for the analysis-workflow configuration surface: the model-expression grammar,
the --dust selection, run enumeration, and the argument parser itself.

Everything here is pure -- no Excel I/O and no fitting -- so it covers the parts of the CLI
contract that are cheap to get wrong and expensive to discover in the middle of a sweep.
"""

import os
import pytest

from heliosoil.horizontal_impaction import COMPONENT_KEYS, describe_components, parse_model_expression

from analysis_scripts import model_pipeline as mp
from analysis_scripts import model_selection as ms
from analysis_scripts.cli import _build_config, build_parser


# ---------------------------------------------------------------------------
# 1. parse_model_expression: the inverse of describe_components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "components, component_dust_types",
    [
        (["gravitational"], None),
        (["gravitational", "normal_wind"], None),
        (["gravitational", "tangential_wind"], {"gravitational": "PM17", "tangential_wind": "PM10-PM2.5"}),
        (list(COMPONENT_KEYS), "PM10"),
        (["turbulent_wind", "impaction_retention"], {"turbulent_wind": "PM10-PM2.5"}),  # partial: the other is None
    ],
)
def test_parse_model_expression_round_trips_describe_components(components, component_dust_types):
    expression = describe_components(components, component_dust_types)
    keys, dust_types = parse_model_expression(expression)

    assert keys == [k for k in COMPONENT_KEYS if k in components]
    assert describe_components(keys, dust_types) == expression


def test_parse_model_expression_accepts_the_users_spelling():
    """Parens around a difference are optional, key separators are interchangeable, "PMT" is
    canonicalized to PM17, and whitespace is free-form."""
    keys, dust_types = parse_model_expression("PM2.5*tangential-wind + PM10-PM2.5*turbulent_wind+PMT * gravitational")

    assert keys == ["gravitational", "turbulent_wind", "tangential_wind"]  # canonical order, not written order
    assert dust_types == {"gravitational": "PM17", "turbulent_wind": "PM10-PM2.5", "tangential_wind": "PM2.5"}


def test_parse_model_expression_bare_components_defer_to_the_configs_dust_type():
    keys, dust_types = parse_model_expression("gravitational + PM10*normal_wind")

    assert keys == ["gravitational", "normal_wind"]
    assert dust_types == {"gravitational": None, "normal_wind": "PM10"}


@pytest.mark.parametrize(
    "expression, message",
    [
        ("gravitational + not_a_mechanism", "Unknown wind component"),
        ("PM10*gravitational + PM2.5*gravitational", "more than once"),
        ("PM2.5-PM10*gravitational", "encloses no size band"),  # difference must subtract the smaller cutoff
        ("XYZ*gravitational", "Unrecognized dust"),
        ("gravitational + ", "Empty term"),
        ("", "Empty model expression"),
    ],
)
def test_parse_model_expression_rejects_bad_input(expression, message):
    with pytest.raises(ValueError, match=message):
        parse_model_expression(expression)


# ---------------------------------------------------------------------------
# 2. resolve_dust_selection
# ---------------------------------------------------------------------------


@pytest.fixture
def site(monkeypatch):
    """A PipelineConfig whose site reports a fixed set of measures, so the dust helpers can be
    exercised without opening any campaign workbook."""
    monkeypatch.setattr(mp, "available_dust_types", lambda cfg, files=None: ["PM2.5", "PM10", "PM17"])
    return mp.PipelineConfig(location="yadnarie")


@pytest.mark.parametrize("mode", ["all", "sweep"])
def test_resolve_dust_selection_modes_expand_to_every_measure(site, mode):
    dust_types, resolved = mp.resolve_dust_selection(site, mode)

    assert dust_types == ["PM2.5", "PM10", "PM17"]
    assert resolved == mode


@pytest.mark.parametrize("given, expected", [("PM10", "PM10"), ("PMT", "PM17"), ("pm2p5", "PM2.5"), ("PM2_5", "PM2.5")])
def test_resolve_dust_selection_canonicalizes_a_named_measure(site, given, expected):
    """The resolved name becomes a results-folder name, so it must match the spelling the
    model itself reports -- "PMT" is PM17."""
    assert mp.resolve_dust_selection(site, given) == ([expected], "single")


def test_resolve_dust_selection_rejects_a_measure_the_site_lacks(site):
    with pytest.raises(ValueError, match="has no PM1 data"):
        mp.resolve_dust_selection(site, "PM1")


def test_resolve_dust_selection_rejects_a_non_measure(site):
    with pytest.raises(ValueError, match="neither a mode"):
        mp.resolve_dust_selection(site, "everything")


# ---------------------------------------------------------------------------
# 3. build_runs / ALL_COMPONENT_COMBOS
# ---------------------------------------------------------------------------


def test_all_component_combos_is_every_non_empty_subset():
    combos = mp.ALL_COMPONENT_COMBOS

    assert len(combos) == 2 ** len(COMPONENT_KEYS) - 1 == 31
    assert [c for c in combos if len(c) == 1] == [[k] for k in COMPONENT_KEYS]
    assert len({tuple(c) for c in combos}) == len(combos)  # unique
    assert all(c == [k for k in COMPONENT_KEYS if k in c] for c in combos)  # canonically ordered


def test_build_runs_sweeps_every_model_type_over_the_combos():
    runs = mp.build_runs(None, mp.ALL_COMPONENT_COMBOS)

    assert len(runs) == 2 + len(mp.ALL_COMPONENT_COMBOS)  # the two component-less types, once each
    assert [r for r in runs if r[0] != "constant_mean_wind"] == [("constant_mean", None, None), ("semi_physical", None, None)]
    assert all(cdt is None for _mt, _wc, cdt in runs)


def test_build_runs_with_one_model_type_and_one_combo():
    runs = mp.build_runs("constant_mean_wind", [["gravitational", "normal_wind"]], {"gravitational": "PM17"})

    assert runs == [("constant_mean_wind", ["gravitational", "normal_wind"], {"gravitational": "PM17"})]


def test_build_runs_sweep_enumerates_every_assignment():
    specs = ["PM10", "PM2.5", "PM10-PM2.5"]
    combos = [["gravitational"], ["gravitational", "normal_wind"]]
    runs = mp.build_runs("constant_mean_wind", combos, "sweep", specs)

    assert len(runs) == len(specs) + len(specs) ** 2
    assert all(mp.is_dust_type_independent(*r) for r in runs)  # every mechanism names its own channel


def test_build_runs_sweep_needs_specs():
    with pytest.raises(ValueError, match="needs dust_specs"):
        mp.build_runs("constant_mean_wind", [["gravitational"]], "sweep")


# ---------------------------------------------------------------------------
# 4. resolve_selection: what --model-type / --model / --dust add up to
# ---------------------------------------------------------------------------


def test_selection_without_a_model_compares_every_combination(site):
    runs, dust_types, mode = ms.resolve_selection(site, None, None, "all", files=[])

    assert (dust_types, mode) == (["PM2.5", "PM10", "PM17"], "all")
    assert len(runs) == 2 + len(mp.ALL_COMPONENT_COMBOS)
    assert ms.n_work_items(runs, dust_types) == 3 * len(runs)  # nothing is dust-type independent


def test_selection_with_a_model_compares_only_that_model(site):
    runs, dust_types, _mode = ms.resolve_selection(site, "constant_mean_wind", "gravitational + normal_wind", "PM10", files=[])

    assert dust_types == ["PM10"]
    assert runs == [("constant_mean_wind", ["gravitational", "normal_wind"], {"gravitational": None, "normal_wind": None})]


def test_a_model_with_fixed_channels_is_dust_type_independent(site):
    """Naming every channel makes the model independent of the swept dust type, so it is fit
    once rather than three times even though --dust asked for all three."""
    runs, dust_types, _mode = ms.resolve_selection(site, "constant_mean_wind", "PM17*gravitational + PM2.5*normal_wind", "all", files=[])

    assert len(dust_types) == 3
    assert ms.n_work_items(runs, dust_types) == 1


def test_sweep_enumerates_every_assignment_over_the_models_mechanisms(site, monkeypatch):
    monkeypatch.setattr(mp, "available_dust_specs", lambda cfg, files=None: ["PM2.5", "PM10", "PM17", "PM10-PM2.5", "PM17-PM2.5", "PM17-PM10"])
    runs, dust_types, mode = ms.resolve_selection(site, "constant_mean_wind", "gravitational + normal_wind", "sweep", files=[])

    assert mode == "sweep"
    assert len(runs) == 6**2
    assert all(mp.is_dust_type_independent(*r) for r in runs)
    assert ms.n_work_items(runs, dust_types) == 6**2  # fit once, not once per dust type


def test_sweep_needs_a_model_to_assign_channels_to(site):
    with pytest.raises(ValueError, match="--dust sweep needs a --model"):
        ms.resolve_selection(site, None, None, "sweep", files=[])


def test_sweep_refuses_a_model_that_already_fixes_its_channels(site):
    """Otherwise the sweep would silently overwrite the channels the user asked for."""
    with pytest.raises(ValueError, match=r"already fixes one for \['gravitational'\]"):
        ms.resolve_selection(site, None, "PM17*gravitational + normal_wind", "sweep", files=[])


@pytest.mark.parametrize(
    "component_dust_types, expected",
    [
        (None, "Grav+Norm"),
        ({"gravitational": None, "normal_wind": None}, "Grav+Norm"),  # bare mechanisms name no channel
        ({"gravitational": "PM17", "normal_wind": "PM10-PM2.5"}, "Grav(PM17)+Norm(PM10-PM2.5)"),
        ({"gravitational": "PM17", "normal_wind": None}, "Grav(PM17)+Norm"),
        ("PM10", "Grav(PM10)+Norm(PM10)"),
    ],
)
def test_progress_labels_only_name_a_channel_a_mechanism_actually_has(component_dust_types, expected):
    assert mp.abbreviate_run("constant_mean_wind", ["gravitational", "normal_wind"], component_dust_types) == expected


def test_a_sweep_is_exponential_in_the_mechanism_count():
    """Why the sweep is gated: it is len(dust_specs)**n_mechanisms runs, so the difference
    between a two- and a five-mechanism model at a 5-measure site is four orders of magnitude."""
    specs = ["x"] * (5 + 10)  # the measures plus every pair difference

    def n_runs(components):
        return len(mp.build_runs("constant_mean_wind", [components], "sweep", specs))

    assert n_runs(["gravitational", "normal_wind"]) == 225
    assert n_runs(list(COMPONENT_KEYS)) == 759_375


# ---------------------------------------------------------------------------
# 5. prompt_yes_no: a long sweep must never die on an unanswerable prompt
# ---------------------------------------------------------------------------


class _Stdin:
    def __init__(self, tty):
        self._tty = tty

    def isatty(self):
        return self._tty


@pytest.mark.parametrize("typed, expected", [("y", True), ("Y\n", True), ("n", False), ("", False)])
def test_prompt_yes_no_reads_an_answer(monkeypatch, typed, expected):
    monkeypatch.setattr(mp.sys, "stdin", _Stdin(tty=True))
    monkeypatch.setattr("builtins.input", lambda _prompt: typed)

    assert mp.prompt_yes_no("Go?") is expected


def test_prompt_yes_no_returns_none_without_a_terminal(monkeypatch):
    monkeypatch.setattr(mp.sys, "stdin", _Stdin(tty=False))

    assert mp.prompt_yes_no("Go?") is None


def test_prompt_yes_no_returns_none_on_eof(monkeypatch):
    """A stdin that claims to be a TTY but is detached raises EOFError on read. That must be
    an unanswered question, not an exception thrown away hours into a sweep."""

    def _eof(_prompt):
        raise EOFError

    monkeypatch.setattr(mp.sys, "stdin", _Stdin(tty=True))
    monkeypatch.setattr("builtins.input", _eof)

    assert mp.prompt_yes_no("Go?") is None


# ---------------------------------------------------------------------------
# 6. The parser surface
# ---------------------------------------------------------------------------


def test_select_needs_only_a_location_and_a_dust_policy():
    args = build_parser().parse_args(["select", "--location", "yadnarie", "--dust", "sweep"])

    assert (args.location, args.dust, args.model_type) == ("yadnarie", "sweep", "all")
    assert not hasattr(args, "dust_type")  # select resolves its own dust types per pass


def test_select_defaults_to_all_dust_types_and_no_named_model():
    args = build_parser().parse_args(["select"])

    assert args.dust == "all"
    assert args.model is None  # compare every component combination


def test_select_takes_a_model_expression_for_a_dust_sweep():
    args = build_parser().parse_args(["select", "--model", "gravitational + tangential_wind", "--dust", "sweep"])

    assert (args.model, args.dust) == ("gravitational + tangential_wind", "sweep")


def test_simulate_takes_a_model_expression():
    args = build_parser().parse_args(
        ["simulate", "--location", "yadnarie", "--model", "PM2.5*tangential_wind + PMT*gravitational", "--dust-type", "PM10"]
    )

    assert args.model == "PM2.5*tangential_wind + PMT*gravitational"
    assert args.dust_type == "PM10"
    assert parse_model_expression(args.model)[0] == ["gravitational", "tangential_wind"]


def test_simulate_defaults_to_the_library_model():
    assert build_parser().parse_args(["simulate"]).model is None


def test_simulate_cross_validates_unless_a_training_split_is_named():
    """No --train-experiments means no named training split, which is what tells `simulate` to
    run leave-one-campaign-out instead of fitting one."""
    cross_validated = _build_config(build_parser().parse_args(["simulate"]))
    named = _build_config(build_parser().parse_args(["simulate", "--train-experiments", "0", "2"]))

    assert cross_validated.train_experiments is None
    assert named.train_experiments == [0, 2]


def test_select_carries_no_training_split():
    """`select` enumerates its own fold schedule and never reads the field; it must not inherit a
    stale default from the shared config either."""
    assert _build_config(build_parser().parse_args(["select"])).train_experiments is None


def test_the_old_fit_subcommand_is_gone():
    """The stage is named `simulate` everywhere now (it always wrote to results/simulate/); a
    script still saying `fit` must fail loudly rather than silently do something else."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["fit", "--location", "yadnarie"])


@pytest.mark.parametrize(
    "flag",
    [
        "--wind-components",
        "--explore-components",
        "--min-components",
        "--max-components",
        "--component-dust",
        "--component-dust-sweep",
        "--dust-specs",
        "--dust-types",
        "--cv-train-counts",
        "--cv-max-folds",
    ],
)
def test_removed_flags_are_gone(flag):
    """These were folded into fixed defaults or into --dust/--model; a stale script using one
    must fail loudly rather than silently ignore it."""
    for command in ("select", "simulate"):
        with pytest.raises(SystemExit):
            build_parser().parse_args([command, flag, "gravitational"])


def test_select_still_takes_the_site_and_run_control_options():
    args = build_parser().parse_args(
        [
            "select",
            "--dust",
            "PM10",
            "--k-factor",
            "none",
            "--surface",
            "first",
            "--no-daily-average",
            "--train-mirrors",
            "ON_M1_T00",
            "--jobs",
            "2",
            "--force",
        ]
    )

    assert args.k_factor is None
    assert args.surface == "first"
    assert args.daily_average is False
    assert args.train_mirrors == ["ON_M1_T00"]
    assert (args.jobs, args.force) == (2, True)


def test_experiment_takes_only_a_location_and_run_control():
    args = build_parser().parse_args(["experiment", "--location", "yadnarie", "--run-name", "wx", "--force", "--verbose"])

    assert (args.location, args.run_name) == ("yadnarie", "wx")
    assert (args.force, args.verbose) == (True, True)
    # The weather summary depends on none of the model/physics options, so it does not carry them.
    for option in ("model", "model_type", "dust_type", "dust", "k_factor", "surface", "daily_average", "train_mirrors"):
        assert not hasattr(args, option)


def test_experiment_defaults_to_the_configs_location():
    assert build_parser().parse_args(["experiment"]).location == mp.PipelineConfig().location


@pytest.mark.parametrize("flag", ["--model", "--model-type", "--dust", "--dust-type", "--k-factor", "--surface", "--train-mirrors"])
def test_experiment_rejects_the_model_and_physics_options(flag):
    """The parent-parser split is what keeps `experiment --help` honest; a flag leaking back in must
    fail loudly rather than be silently accepted and ignored."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["experiment", flag, "gravitational"])


@pytest.mark.parametrize("command", ["simulate", "select"])
def test_the_model_workflows_keep_their_options_after_the_parent_split(command):
    """The other half of the split guard: simulate/select must be unchanged by it."""
    args = build_parser().parse_args(
        [command, "--model-type", "semi_physical", "--model", "gravitational", "--k-factor", "2.0", "--surface", "first"]
    )

    assert (args.model_type, args.model) == ("semi_physical", "gravitational")
    assert (args.k_factor, args.surface) == (2.0, "first")


# ---------------------------------------------------------------------------
# 7. --fit-method / resolve_fit_method
# ---------------------------------------------------------------------------


def test_fit_method_defaults_to_mle_and_reaches_the_config():
    assert _build_config(build_parser().parse_args(["simulate"])).fit_method == "mle"
    assert _build_config(build_parser().parse_args(["select", "--fit-method", "ls"])).fit_method == "ls"


def test_fit_method_rejects_an_unknown_name():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["simulate", "--fit-method", "bayes"])


def test_every_model_type_honours_least_squares():
    """All three fit their mean by least squares -- the constant-mean pair by a linear solve,
    semi_physical by a bounded scalar search -- so --fit-method ls compares like with like."""
    cfg = mp.PipelineConfig(location="yadnarie", fit_method="ls")

    assert [mp.resolve_fit_method(cfg, mt) for mt in mp.MODEL_TYPES] == ["ls"] * len(mp.MODEL_TYPES)


def test_resolve_fit_method_falls_back_for_a_model_type_without_a_least_squares_fit():
    """The capability is read off the model class, not a hardcoded list, so a future model type
    that cannot do least squares degrades to MLE instead of breaking --model-type all."""

    class NoLeastSquares:
        pass

    cfg = mp.PipelineConfig(location="yadnarie", fit_method="ls")
    monkeyed = dict(mp.MODEL_CLASSES, exotic=NoLeastSquares)

    assert not hasattr(NoLeastSquares, "fit_ls")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mp, "MODEL_CLASSES", monkeyed)
        assert mp.resolve_fit_method(cfg, "exotic") == "mle"
        assert mp.resolve_fit_method(cfg, "semi_physical") == "ls"


def test_resolve_fit_method_leaves_every_model_type_on_mle_by_default():
    cfg = mp.PipelineConfig(location="yadnarie")

    assert [mp.resolve_fit_method(cfg, mt) for mt in mp.MODEL_TYPES] == ["mle"] * len(mp.MODEL_TYPES)


def test_resolve_fit_method_rejects_an_unknown_method():
    cfg = mp.PipelineConfig(location="yadnarie", fit_method="ols")

    with pytest.raises(ValueError, match="Unknown fit_method"):
        mp.resolve_fit_method(cfg, "constant_mean_wind")


MIRRORS = ["ON_M1_T00", "OE_M2_T30", "OS_M3_T60", "OW_M4_T90"]


def test_least_squares_trains_on_every_mirror():
    """The single-representative-mirror default guards against over-counting one shared
    deposition-noise process across correlated mirrors. Least squares fits no noise process at
    all, so that argument does not apply and every mirror is a genuine design-matrix row --
    including for a purely gravitational model, which under MLE would be cut to one."""
    ls = mp.PipelineConfig(location="yadnarie", fit_method="ls")
    mle = mp.PipelineConfig(location="yadnarie")

    assert mp.resolve_training_mirrors(ls, MIRRORS, "constant_mean_wind", ["gravitational"]) == MIRRORS
    assert mp.resolve_training_mirrors(mle, MIRRORS, "constant_mean_wind", ["gravitational"]) == ["ON_M1_T00"]


def test_least_squares_mirror_default_applies_to_every_model_type():
    """Every type honours "ls" and so fits no sigma_dep, which is the only reason the default
    ever cut training down to one mirror. Under MLE semi_physical still does."""
    ls = mp.PipelineConfig(location="yadnarie", fit_method="ls")
    mle = mp.PipelineConfig(location="yadnarie")

    assert mp.resolve_training_mirrors(ls, MIRRORS, "semi_physical", None) == MIRRORS
    assert mp.resolve_training_mirrors(mle, MIRRORS, "semi_physical", None) == ["ON_M1_T00"]


def test_explicit_train_mirrors_still_override_every_fit_method():
    cfg = mp.PipelineConfig(location="yadnarie", fit_method="ls", train_mirrors=["OE_M2_T30"])

    assert mp.resolve_training_mirrors(cfg, MIRRORS, "constant_mean_wind", ["gravitational"]) == ["OE_M2_T30"]


# --------------------------------------------------------------------------- #
# --data-subdir: an optional folder level between data/ and the location
# --------------------------------------------------------------------------- #


def test_data_subdir_defaults_to_the_historical_layout():
    """The default must reproduce data/<location>/ exactly -- os.path.join drops the empty
    component rather than emitting a doubled separator."""
    cfg = mp.PipelineConfig(location="yadnarie")
    assert cfg.data_subdir == ""
    assert os.path.normpath(cfg.data_dir).endswith(os.path.join("data", "yadnarie"))
    assert "data" + os.sep + os.sep not in cfg.data_dir


def test_data_subdir_inserts_one_folder_level_and_leaves_the_names_alone():
    cfg = mp.PipelineConfig(location="yadnarie", data_subdir="heliosoil-input")
    assert os.path.normpath(cfg.data_dir).endswith(os.path.join("data", "heliosoil-input", "yadnarie"))
    # The file-naming convention is keyed on the location, not on where the folder sits.
    assert cfg.file_prefix == "soiling_yadnarie"
    assert os.path.basename(cfg.parameter_file) == "parameters_yadnarie_experiments.xlsx"


def test_data_subdir_accepts_a_nested_path():
    cfg = mp.PipelineConfig(location="qut", data_subdir="archive/2025")
    assert os.path.normpath(cfg.data_dir).endswith(os.path.join("data", "archive", "2025", "qut"))


def test_site_name_ignores_the_subdir_so_results_stay_keyed_on_the_site():
    """Two copies of one site's data report into the same results folder; --run-name is what
    separates them. Asserted rather than assumed, since it is a collision risk."""
    plain = mp.PipelineConfig(location="yadnarie")
    nested = mp.PipelineConfig(location="yadnarie", data_subdir="heliosoil-input")
    assert plain.site_name == nested.site_name == "yadnarie"


def test_cli_passes_data_subdir_through_to_the_config():
    parser = build_parser()
    args = parser.parse_args(["simulate", "--location", "yadnarie", "--data-subdir", "heliosoil-input"])
    cfg = _build_config(args)
    assert cfg.data_subdir == "heliosoil-input"
    assert os.path.normpath(cfg.data_dir).endswith(os.path.join("data", "heliosoil-input", "yadnarie"))

    default = _build_config(parser.parse_args(["simulate", "--location", "yadnarie"]))
    assert default.data_subdir == ""
