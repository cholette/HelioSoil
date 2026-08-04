"""
Unit tests for the `simulate` workflow: the tilt-grouped daily soiling-rate error figure, the
leave-one-campaign-out schedule, the figure-naming contract, and the stitched out-of-sample view.

Pure -- no Excel I/O, no fitting -- driven by stubs with the same int-keyed-dict shape as
ReflectanceMeasurements and a fitted model's Heliostats. Two campaigns share a tilt set so the
in-sample/out-of-sample pairing is exercised, and one mirror carries a NaN tilt to check it is
dropped rather than given a group of its own.
"""

import os
import types

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis_scripts import simulate as sim  # noqa: E402

R0 = 0.9  # scalar nominal (clean) reflectance fallback


def _campaign(soiling_factor, average, tilts, days=(0, 2, 4)):
    """One campaign's (soiling_factor, average, tilts, times) in the layouts the real objects use:
    soiling_factor (n_mirror, n_time), average (n_time, n_mirror), tilts (n_mirror, n_time)."""
    return (
        np.asarray(soiling_factor, float),
        np.asarray(average, float),
        np.asarray(tilts, float),
        np.array(days, dtype="timedelta64[D]") + np.datetime64("2024-01-01"),
    )


@pytest.fixture
def model_and_data():
    """Two campaigns x 4 mirrors x 3 measurements. Mirrors sit at 0/0/30 deg, and the fourth has no
    tilt at all. Campaign 1's predictions are deliberately worse than campaign 0's."""
    tilts = np.vstack([np.full(3, 0.0), np.full(3, 0.0), np.full(3, 30.0), np.full(3, np.nan)])
    c0 = _campaign(
        soiling_factor=[[1.0, 0.98, 0.96]] * 4,
        average=[[R0, R0, R0, R0], [R0 * 0.98, R0 * 0.98, R0 * 0.97, R0 * 0.98], [R0 * 0.96, R0 * 0.96, R0 * 0.94, R0 * 0.96]],
        tilts=tilts,
    )
    c1 = _campaign(
        soiling_factor=[[1.0, 0.99, 0.98]] * 4,
        average=[[R0, R0, R0, R0], [R0 * 0.95, R0 * 0.95, R0 * 0.93, R0 * 0.95], [R0 * 0.90, R0 * 0.90, R0 * 0.87, R0 * 0.90]],
        tilts=tilts,
    )

    helios = types.SimpleNamespace(soiling_factor={0: c0[0], 1: c1[0]}, nominal_reflectance=R0)
    model = types.SimpleNamespace(helios=helios)
    rdat = types.SimpleNamespace(
        prediction_indices={0: [0, 1, 2], 1: [0, 1, 2]},
        average={0: c0[1], 1: c1[1]},
        tilts={0: c0[2], 1: c1[2]},
        times={0: c0[3], 1: c1[3]},
        nominal_reflectance={},  # empty -> _nominal_reflectance_series returns the scalar fallback
    )
    return model, rdat


def test_groups_mirrors_by_tilt_and_drops_the_nan_tilt(model_and_data):
    model, rdat = model_and_data
    groups = sim._abs_rate_error_by_tilt(model, rdat, [0])

    assert sorted(groups) == [0.0, 30.0]  # the NaN-tilt mirror gets no group of its own
    assert groups[0.0].size == 4  # 2 mirrors x 2 intervals
    assert groups[30.0].size == 2  # 1 mirror x 2 intervals
    assert all((values >= 0).all() for values in groups.values())  # absolute error


def test_pools_the_campaigns_of_a_split(model_and_data):
    model, rdat = model_and_data
    one = sim._abs_rate_error_by_tilt(model, rdat, [0])
    both = sim._abs_rate_error_by_tilt(model, rdat, [0, 1])

    assert both[0.0].size == 2 * one[0.0].size
    assert both[30.0].size == 2 * one[30.0].size


def test_the_box_mean_is_the_reported_mae(model_and_data):
    """The figure's mean marker must be the MAE performance_stats.csv reports for that tilt, up to
    the p.p./day x100 -- otherwise the two disagree about the same model."""
    from heliosoil.paper_specific_utilities import regression_performance_stats

    model, rdat = model_and_data
    groups = sim._abs_rate_error_by_tilt(model, rdat, [0])
    at_30 = regression_performance_stats(model, rdat, [0], mirrors=np.array([2]))

    assert groups[30.0].mean() == pytest.approx(at_30["MAE"] * 100.0)


def test_table_has_a_row_per_split_and_tilt(model_and_data):
    model, rdat = model_and_data
    table = sim.rate_error_by_tilt_table(model, rdat, [0], [1])

    assert list(table.columns) == ["split", "tilt", "MAE_pp_day", "median_pp_day", "p95_pp_day", "N"]
    assert sorted(table["split"].unique()) == ["in_sample", "out_of_sample"]
    assert len(table) == 4  # 2 splits x 2 tilts
    # Campaign 1's predictions are the worse ones, so its MAE must be the larger at every tilt.
    by_split = table.set_index(["split", "tilt"])["MAE_pp_day"]
    assert (by_split["out_of_sample"] > by_split["in_sample"]).all()


def test_figure_draws_paired_boxes(model_and_data):
    model, rdat = model_and_data
    fig = sim.plot_rate_error_by_tilt(model, rdat, [0], [1], "Test Site")

    assert fig is not None
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "30"]
    assert len(ax.get_legend().get_texts()) == 2  # in-sample and out-of-sample
    plt.close(fig)


def test_figure_survives_an_empty_test_split(model_and_data):
    """Training on every campaign leaves nothing out of sample; the in-sample boxes still draw."""
    model, rdat = model_and_data
    fig = sim.plot_rate_error_by_tilt(model, rdat, [0, 1], [], "Test Site")

    assert fig is not None
    assert len(fig.axes[0].get_legend().get_texts()) == 1
    plt.close(fig)


def test_figure_is_none_when_no_tilt_has_data(model_and_data):
    """No campaigns at all -> no file written, rather than an empty pair of axes."""
    model, rdat = model_and_data

    assert sim.plot_rate_error_by_tilt(model, rdat, [], [], "Test Site") is None


def test_the_two_splits_can_come_from_different_fits(model_and_data):
    """What cross-validation needs: the in-sample side pooled over the folds' training campaigns
    and the out-of-sample side taken from the stitched held-out predictions, in one figure."""
    model, rdat = model_and_data
    in_sample = sim._merge_rate_error_groups(sim._abs_rate_error_by_tilt(model, rdat, [0]), sim._abs_rate_error_by_tilt(model, rdat, [1]))
    out_of_sample = sim._abs_rate_error_by_tilt(model, rdat, [1])

    table = sim.rate_error_groups_table(in_sample, out_of_sample)
    by_split = table.set_index(["split", "tilt"])["N"]
    assert by_split[("in_sample", 0.0)] == 2 * by_split[("out_of_sample", 0.0)]  # both campaigns vs one

    fig = sim.plot_rate_error_groups(in_sample, out_of_sample, "Test Site", split_labels=("Pooled folds", "Held-out fold"))
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["Pooled folds", "Held-out fold"]
    plt.close(fig)


def test_merging_groups_pools_tilt_by_tilt(model_and_data):
    model, rdat = model_and_data
    one = sim._abs_rate_error_by_tilt(model, rdat, [0])
    merged = sim._merge_rate_error_groups(one, sim._abs_rate_error_by_tilt(model, rdat, [1]))

    assert sorted(merged) == sorted(one)  # same tilts, more observations in each
    assert all(merged[tilt].size == 2 * one[tilt].size for tilt in one)


# ---------------------------------------------------------------------------
# The leave-one-campaign-out schedule and the naming it drives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_campaigns", [2, 3, 4])
def test_every_campaign_is_held_out_exactly_once(n_campaigns):
    """The property the stitched out-of-sample view depends on: one fold per campaign, each
    training on all the others, so every campaign has exactly one held-out prediction."""
    folds = sim.leave_one_campaign_out_folds(n_campaigns)

    assert [fold for fold, _train, _test in folds] == list(range(1, n_campaigns + 1))
    assert sorted(test[0] for _f, _train, test in folds) == list(range(n_campaigns))
    for fold, train, test in folds:
        assert test == [fold - 1]  # fold k holds out campaign k, 1-based like the campaign folders
        assert sorted(train + test) == list(range(n_campaigns))
        assert len(train) == n_campaigns - 1


def test_a_single_campaign_cannot_be_cross_validated():
    """A fold would have nothing to train on; the CLI turns this into a usage error naming the
    flag that fixes it."""
    with pytest.raises(ValueError, match="at least 2 campaigns"):
        sim.leave_one_campaign_out_folds(1)


@pytest.mark.parametrize(
    "tilt, role, fold, expected",
    [
        (0.0, "train", 2, "reflectance_tilt_00_train-f2.pdf"),
        (30.0, "test", 1, "reflectance_tilt_30_test-f1.pdf"),
        (180.0, "train", None, "reflectance_tilt_180_train.pdf"),  # single fit: no fold to name
        (60.0, "test", None, "reflectance_tilt_60_test.pdf"),
    ],
)
def test_tilt_figure_names_carry_the_role_and_the_fold(tilt, role, fold, expected):
    """Each fold writes its own view of every campaign into that campaign's folder, so the name has
    to say which fit drew it -- otherwise the next fold silently overwrites the last."""
    assert sim.tilt_figure_name(tilt, role, fold) == expected


def test_a_campaign_gets_one_figure_per_fit_that_touched_it():
    """Three campaigns -> each campaign folder holds two training views and one test view of a
    given tilt, all distinctly named."""
    names = {
        sim.tilt_figure_name(0.0, "train" if e != fold - 1 else "test", fold)
        for fold, _train, _test in sim.leave_one_campaign_out_folds(3)
        for e in [0]  # campaign 1's figures
    }

    assert names == {"reflectance_tilt_00_test-f1.pdf", "reflectance_tilt_00_train-f2.pdf", "reflectance_tilt_00_train-f3.pdf"}


# ---------------------------------------------------------------------------
# FoldStitchedModel: the whole site, out-of-sample, in one model-shaped object
# ---------------------------------------------------------------------------


def test_stitched_model_serves_each_campaign_from_its_own_fold(model_and_data):
    template, _rdat = model_and_data
    held_out = {0: np.full((4, 3), 0.5), 1: np.full((4, 3), 0.25)}
    stitched = sim.FoldStitchedModel(template, held_out, {0: np.zeros((4, 3)), 1: np.zeros((4, 3))})

    assert stitched.helios.soiling_factor[0] is held_out[0]
    assert stitched.helios.soiling_factor[1] is held_out[1]
    assert stitched.helios.nominal_reflectance == template.helios.nominal_reflectance  # fold-invariant


def test_stitched_model_leaves_the_template_untouched(model_and_data):
    """The template is a real fitted model the caller may still be scoring; stitching must not
    reach into its predictions."""
    template, _rdat = model_and_data
    before = {e: array.copy() for e, array in template.helios.soiling_factor.items()}
    sim.FoldStitchedModel(template, {0: np.full((4, 3), 0.5), 1: np.full((4, 3), 0.25)}, {0: None, 1: None})

    assert all(np.array_equal(template.helios.soiling_factor[e], array) for e, array in before.items())


def test_stitched_model_refuses_to_re_predict(model_and_data):
    """Both plotting helpers call predict_soiling_factor on entry. Re-predicting would replace
    every campaign's held-out prediction with the template fold's own."""
    template, _rdat = model_and_data
    held_out = {0: np.full((4, 3), 0.5), 1: np.full((4, 3), 0.25)}
    stitched = sim.FoldStitchedModel(template, held_out, {0: None, 1: None})

    stitched.predict_soiling_factor(object(), reflectance_data=object())

    assert stitched.helios.soiling_factor[0] is held_out[0]


def test_stitched_model_delegates_everything_else(model_and_data):
    template, _rdat = model_and_data
    template.model_name = "constant-mean_gravitational"
    stitched = sim.FoldStitchedModel(template, {0: np.zeros((4, 3))}, {0: None})

    assert stitched.model_name == "constant-mean_gravitational"
    with pytest.raises(AttributeError):
        stitched.not_a_model_attribute


def test_stitched_model_is_scored_like_any_other(model_and_data):
    """The point of the shape: regression_performance_stats over every campaign of the stitched
    model is the pooled cross-validated statistic."""
    from heliosoil.paper_specific_utilities import regression_performance_stats

    template, rdat = model_and_data
    stitched = sim.FoldStitchedModel(template, dict(template.helios.soiling_factor), {})

    assert regression_performance_stats(stitched, rdat, [0, 1]) == pytest.approx(regression_performance_stats(template, rdat, [0, 1]))


# ---------------------------------------------------------------------------
# Fit-quality panels and the loss-vs-tilt applicability guard
# ---------------------------------------------------------------------------


@pytest.fixture
def fit_quality_data(model_and_data):
    """model_and_data plus the rho0 that fit_quality_plots indexes measurements against, and
    array-valued prediction_indices: unlike the stats helpers, fit_quality_plots does arithmetic
    on them (pi[f] - pi[f][0]), which the real ReflectanceMeasurements supports and a list does
    not."""
    model, rdat = model_and_data
    rdat.rho0 = {0: np.full(4, R0), 1: np.full(4, R0)}
    rdat.prediction_indices = {f: np.asarray(v) for f, v in rdat.prediction_indices.items()}
    return model, rdat


def test_fit_quality_drops_the_panel_when_no_mirror_was_held_out(fit_quality_data):
    """Any least-squares fit, and any orientation-dependent noise model, trains on every common
    mirror. The held-out-mirror panel then has nothing to show and must be dropped, not drawn
    blank -- and certainly not reached with an empty index list."""
    from heliosoil.paper_specific_utilities import summarize_fit_quality

    model, rdat = fit_quality_data
    fig, ax = summarize_fit_quality(model, rdat, [0], [0, 1, 2, 3], [], [1], save_file=None)
    assert len(ax) == 2
    plt.close(fig)

    fig3, ax3 = summarize_fit_quality(model, rdat, [0], [0, 1], [2, 3], [1], save_file=None)
    assert len(ax3) == 3
    plt.close(fig3)


def test_fit_quality_plots_annotates_an_empty_mirror_selection(fit_quality_data):
    """The guard is on the plotting primitive too, so a caller passing no mirrors gets a labelled
    axes back rather than a min-of-empty-array error."""
    from heliosoil.paper_specific_utilities import fit_quality_plots

    model, rdat = fit_quality_data
    fig, ax = plt.subplots()
    returned_fig, returned_ax = fit_quality_plots(model, rdat, [0], [], ax=ax)
    assert returned_ax is ax and returned_fig is fig
    assert any("no mirrors held out" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_fit_quality_plots_returns_the_axes_it_drew_on(fit_quality_data):
    """It used to fall off the end returning None whenever an axes was supplied, so a caller could
    not chain from it."""
    from heliosoil.paper_specific_utilities import fit_quality_plots

    model, rdat = fit_quality_data
    fig, ax = plt.subplots()
    assert fit_quality_plots(model, rdat, [0], [0, 1], ax=ax) == (fig, ax)
    plt.close(fig)


def test_standalone_fit_quality_figures_are_one_file_per_split(fit_quality_data, tmp_path):
    """Standalone rather than panels: three square scatters side by side leave each too small to
    read against its own 1:1 line. The held-out-mirror split still disappears when there is none,
    so the file set follows the split that exists."""
    from heliosoil.paper_specific_utilities import save_fit_quality_figures

    model, rdat = fit_quality_data
    paths = save_fit_quality_figures(model, rdat, [0], [0, 1], [2, 3], [1], str(tmp_path))
    assert [os.path.basename(p) for p in paths] == [
        "fit_quality_train_mirrors.pdf",
        "fit_quality_test_mirrors.pdf",
        "fit_quality_test_experiments.pdf",
    ]
    assert all(os.path.exists(p) for p in paths)

    held_none = save_fit_quality_figures(model, rdat, [0], [0, 1, 2, 3], [], [1], str(tmp_path / "none"))
    assert [os.path.basename(p) for p in held_none] == ["fit_quality_train_mirrors.pdf", "fit_quality_test_experiments.pdf"]

    # Fitted on every campaign: no test experiments either, so that file is not written rather
    # than written containing only the "nothing here" annotation.
    trained_on_all = save_fit_quality_figures(model, rdat, [0, 1], [0, 1, 2, 3], [], [], str(tmp_path / "all"))
    assert [os.path.basename(p) for p in trained_on_all] == ["fit_quality_train_mirrors.pdf"]


def test_standalone_fit_quality_figures_share_axis_limits(fit_quality_data, tmp_path):
    """Splitting the panels apart loses the shared axes a subplot grid gave for free, so the
    pooled loss range is computed over every split first and passed to all of them. Without it two
    figures on different scales would look like different fits."""
    from heliosoil.paper_specific_utilities import daily_loss_pairs, fit_quality_panels, fit_quality_plots

    model, rdat = fit_quality_data
    panels = fit_quality_panels([0], [0, 1], [2, 3], [1])

    limits = []
    for _title, _slug, experiments, mirrors in panels:
        fig, ax = plt.subplots()
        pooled = [
            v
            for _t, _s, exps, mirs in panels
            for pair in [daily_loss_pairs(model, rdat, exps, mirs)]
            for v in (pair[0].min(), pair[0].max(), pair[1].min(), pair[1].max())
        ]
        fit_quality_plots(model, rdat, experiments, mirrors, ax=ax, min_loss=min(0.0, *pooled), max_loss=max(0.0, *pooled))
        limits.append((ax.get_xlim(), ax.get_ylim()))
        plt.close(fig)

    assert len(set(limits)) == 1


def test_loss_curve_covers_every_constant_mean_model_with_a_noise_process():
    """The curve evaluates the model forward on virtual mirrors, so any constant-mean model --
    plain or wind-driven -- qualifies. What it still needs is a noise process to sample, which a
    least-squares fit does not leave behind."""
    import heliosoil.fitting as smf
    from heliosoil.horizontal_impaction import ConstantMeanWindDeposition
    from heliosoil.paper_specific_utilities import supports_loss_curve

    plain = smf.ConstantMeanDeposition.__new__(smf.ConstantMeanDeposition)
    plain.sigma_dep = 1e-4
    assert supports_loss_curve(plain)

    # The point of the generalisation: a wind model is no longer refused.
    wind = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    wind._sigma_param_names = ["sigma_dep", "sigma_dep_gamma"]
    wind.sigma_dep, wind.sigma_dep_gamma = 1e-4, 1e-6
    assert supports_loss_curve(wind)

    # A least-squares fit leaves every sigma None: no noise process to sample.
    for model, names in ((smf.ConstantMeanDeposition, None), (ConstantMeanWindDeposition, ["sigma_dep", "sigma_dep_gamma"])):
        ls = model.__new__(model)
        if names is not None:
            ls._sigma_param_names = names
            ls.sigma_dep_gamma = None
        ls.sigma_dep = None
        assert not supports_loss_curve(ls)

    # A physical model has no virtual-mirror forward evaluation at all.
    physical = smf.SemiPhysical.__new__(smf.SemiPhysical)
    physical.sigma_dep = 1e-4
    assert not supports_loss_curve(physical)


def test_sampling_covariance_repairs_an_unidentified_parameter():
    """An MLE covariance from an inverted numerical Hessian can hand back a negative variance, or
    one so large a log-scale draw overflows exp() to inf. Both were observed on a fitted wind
    model and both turned the whole curve into NaN, so they are repaired before sampling."""
    from heliosoil.paper_specific_utilities import sampling_covariance

    cov = np.diag([2.4e-2, -1.69e10, 2.96e6, 3.7e-3])
    repaired, pinned = sampling_covariance(cov)

    # The two broken entries are neutralised by different steps. Clipping the negative eigenvalue
    # already drives index 1's variance to zero, which is itself "hold at the estimate", so only
    # the large-but-positive index 2 needs pinning by name.
    assert pinned.tolist() == [2]
    assert repaired[1, 1] == pytest.approx(0.0)
    assert np.allclose(repaired[2, :], 0.0) and np.allclose(repaired[:, 2], 0.0)
    # Still usable for sampling: symmetric, positive semi-definite, and finite throughout.
    assert np.isfinite(repaired).all()
    assert np.allclose(repaired, repaired.T)
    assert (np.linalg.eigvalsh(repaired) >= -1e-12).all()
    # The identified parameters keep their variance untouched.
    assert repaired[0, 0] == pytest.approx(2.4e-2)
    assert repaired[3, 3] == pytest.approx(3.7e-3)


def test_sampling_covariance_leaves_a_healthy_covariance_alone():
    from heliosoil.paper_specific_utilities import sampling_covariance

    cov = np.array([[4.0e-2, 1.0e-3], [1.0e-3, 9.0e-3]])
    repaired, pinned = sampling_covariance(cov)
    assert pinned.size == 0
    np.testing.assert_allclose(repaired, cov)


# ---------------------------------------------------------------------------
# Loss-distribution scenarios
# ---------------------------------------------------------------------------


def test_point_estimate_interval_matches_the_sampled_percentiles():
    """Both interval columns in loss_distributions.csv are 5-95%, so they can be read side by
    side. The point-estimate one therefore uses 1.645 sigma, not the more familiar 1.96, which
    would be a 2.5-97.5% range and would make the point estimate look wider by construction."""
    import inspect

    from heliosoil.paper_specific_utilities import plot_loss_distributions

    source = inspect.getsource(plot_loss_distributions)
    assert "1.645 * sigma" in source
    assert "1.96 * sigma" not in source


def test_loss_distributions_rejects_mismatched_scenario_labels():
    """percents and labels are zipped, so a mismatch would silently drop scenarios off the end."""
    from heliosoil.paper_specific_utilities import plot_loss_distributions

    with pytest.raises(ValueError, match="same length"):
        plot_loss_distributions(None, None, "unused", percents=(5, 50, 95), labels=("Low", "High"))


def test_scenario_ramp_is_ordinal_not_categorical():
    """Low -> Maximum is ordered magnitude, not four identities, so the scenarios take one hue
    stepped light to dark rather than four categorical hues. Monotone lightness is the property
    that makes the ordering readable without consulting the legend."""
    from heliosoil.paper_specific_utilities import _SCENARIO_COLORS

    def relative_luminance(hex_color):
        channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    luminances = [relative_luminance(c) for c in _SCENARIO_COLORS]
    assert luminances == sorted(luminances, reverse=True), "scenario ramp must run light -> dark"
    # The palest step still has to separate from a white-ish page rather than vanish into it.
    assert (relative_luminance(_SCENARIO_COLORS[0]) + 0.05) / 0.05 < 21
    assert relative_luminance(_SCENARIO_COLORS[0]) < 0.6
