"""
Unit tests for the `simulate` workflow: the tilt-grouped daily soiling-rate error figure, the
leave-one-campaign-out schedule, the figure-naming contract, and the stitched out-of-sample view.

Pure -- no Excel I/O, no fitting -- driven by stubs with the same int-keyed-dict shape as
ReflectanceMeasurements and a fitted model's Heliostats. Two campaigns share a tilt set so the
in-sample/out-of-sample pairing is exercised, and one mirror carries a NaN tilt to check it is
dropped rather than given a group of its own.
"""

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
