"""
Unit tests for the measured-soiling section of the experimental location assessment workflow.

Pure -- no Excel I/O, no fitting -- driven by a stub with the same int-keyed-dict shape as
ReflectanceMeasurements. The stub deliberately packs every awkward mirror the real sites produce into
one campaign: a clean one, one with leading NaNs, one with an interior NaN and no tilt data, one
whose name encodes no orientation (qut), and one with no usable data at all.
"""

import types

import pytest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis_scripts import model_pipeline as mp  # noqa: E402
from analysis_scripts import experiment_soiling as es  # noqa: E402

FALLBACK = 0.95
NOMINAL_0 = 0.98  # campaign 0 has a reference mirror; campaign 1 does not


def _times(start: str, n: int, step_days: int = 2) -> np.ndarray:
    return pd.date_range(start, periods=n, freq=f"{step_days}D").values


@pytest.fixture
def data():
    """Two campaigns. Campaign 0: 5 mirrors x 5 times, with a reference mirror. Campaign 1: 2
    mirrors x 3 times, no reference."""
    # mirror 0 clean; 1 leading NaNs; 2 interior NaN; 3 qut-like name; 4 unusable
    average_0 = np.array(
        [
            [0.95, np.nan, 0.95, 0.96, np.nan],
            [0.94, np.nan, 0.94, 0.95, np.nan],
            [0.93, 0.94, np.nan, 0.94, np.nan],
            [0.92, 0.93, 0.92, 0.93, np.nan],
            [0.91, 0.92, 0.90, 0.92, np.nan],
        ]
    )
    tilts_0 = np.vstack(
        [
            np.full(24, 0.0),
            np.concatenate([np.full(10, np.nan), np.full(14, 30.0)]),  # installed mid-campaign
            np.full(24, np.nan),  # no sheet data at all -> name regex
            np.full(24, 45.0),
            np.full(24, 45.0),
        ]
    )
    sim = types.SimpleNamespace(
        time={0: pd.Series(_times("2024-11-11", 5)), 1: pd.Series(_times("2025-02-06", 3))},
        start_datetime={0: pd.Timestamp("2024-01-01"), 1: pd.Timestamp("2025-01-01")},  # stale on purpose
    )
    rdat = types.SimpleNamespace(
        mirror_names={0: ["ONE_M1_T00", "OSE_M2_T30", "OSW_M3_T60", "Mirror_4", "Mirror_5"], 1: ["ONE_M1_T00", "OSE_M2_T30"]},
        average={0: average_0, 1: np.array([[0.96, 0.95], [0.94, 0.93], [0.92, 0.90]])},
        sigma_of_the_mean={0: np.full((5, 5), 0.005), 1: np.full((3, 2), 0.005)},
        times={0: _times("2024-11-11", 5), 1: _times("2025-02-06", 3)},
        tilts={0: tilts_0, 1: np.vstack([np.full(12, 0.0), np.full(12, 30.0)])},
        nominal_reflectance={0: np.full((5, 5), NOMINAL_0)},  # campaign 1 absent -> scalar fallback
        # An obviously wrong sentinel: a test asserts the rate is computed fresh, not read from here.
        soiling_rate={0: np.full(5, 999.0), 1: np.full(2, 999.0)},
        # delta_ref deliberately absent: it is stale after trimming, so a read must raise, not lie.
    )
    return mp.LoadedData(
        files=["a.xlsx", "b.xlsx"],
        training_intervals=None,
        all_mirrors=[],
        site_name="yadnarie",
        parameter_file="",
        sim_data_total=sim,
        reflect_data_total=rdat,
    )


@pytest.fixture
def cfg():
    return mp.PipelineConfig(location="yadnarie")


@pytest.fixture
def mirrors(cfg, data):
    return es.mirror_table(cfg, data, fallback=FALLBACK)


def _row(mirrors, f, name):
    return mirrors[(mirrors["campaign_index"] == f) & (mirrors["mirror"] == name)].iloc[0]


# ---------------------------------------------------------------------------
# 1. Geometry
# ---------------------------------------------------------------------------


def test_mirror_orientation_reads_one_and_two_letter_codes():
    assert es.mirror_orientation("ONE_M1_T00") == "NE"
    assert es.mirror_orientation("OW_M4_T30") == "W"


def test_mirror_orientation_returns_none_for_an_unparseable_qut_name():
    """qut names its mirrors Mirror_1..Mirror_5; the usual split yields "irror"."""
    assert es.mirror_orientation("Mirror_1") is None


@pytest.mark.parametrize("name, expected", [("M_T0", 0.0), ("M_T00", 0.0), ("M_T05", 5.0), ("M_T15", 15.0), ("M_T180", 180.0)])
def test_mirror_tilt_regex_handles_every_naming_scheme(name, expected):
    rdat = types.SimpleNamespace(tilts={})
    assert es.mirror_tilts(rdat, 0, [name])[0] == expected


def test_mirror_tilt_is_nan_safe_when_a_mirror_was_installed_mid_campaign(data):
    """tilts[f][:, 0] -- the usual idiom -- is NaN for this mirror; the median is not."""
    tilts = es.mirror_tilts(data.reflect_data_total, 0, data.reflect_data_total.mirror_names[0])

    assert np.isnan(np.asarray(data.reflect_data_total.tilts[0], dtype=float)[1, 0])
    assert tilts[1] == 30.0


def test_mirror_tilt_falls_back_to_the_name_when_the_sheet_is_all_nan(data):
    tilts = es.mirror_tilts(data.reflect_data_total, 0, data.reflect_data_total.mirror_names[0])

    assert tilts[2] == 60.0  # from "OSW_M3_T60"


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------


def test_soiling_factor_divides_by_the_reference_series_when_the_campaign_has_one(mirrors):
    row = _row(mirrors, 0, "ONE_M1_T00")

    assert row.sf_initial == pytest.approx(0.95 / NOMINAL_0)
    assert row.sf_final == pytest.approx(0.91 / NOMINAL_0)


def test_soiling_factor_falls_back_to_the_scalar_without_a_reference_mirror(mirrors):
    row = _row(mirrors, 1, "ONE_M1_T00")

    assert row.sf_initial == pytest.approx(0.96 / FALLBACK)
    assert row.nominal_source == "parameter file"


def test_the_nominal_source_records_which_branch_was_taken(mirrors):
    assert _row(mirrors, 0, "ONE_M1_T00").nominal_source == "reference mirror"


def test_endpoints_skip_leading_nans_so_a_mirror_installed_mid_campaign_is_not_lost(mirrors):
    row = _row(mirrors, 0, "OSE_M2_T30")

    assert row.n_points == 3
    assert row.rho_initial == pytest.approx(0.94)  # the third row, not the NaN first one
    assert row.rho_final == pytest.approx(0.92)


def test_endpoints_span_an_interior_nan_rather_than_stopping_at_it(mirrors):
    row = _row(mirrors, 0, "OSW_M3_T60")

    assert row.n_points == 4  # one interior sample missing
    assert row.rho_initial == pytest.approx(0.95)
    assert row.rho_final == pytest.approx(0.90)


def test_elapsed_days_uses_each_mirrors_own_endpoints_not_the_campaign_duration(mirrors):
    """The late mirror spans 2 of the campaign's 4 intervals, so its window is shorter."""
    assert _row(mirrors, 0, "ONE_M1_T00").elapsed_days == pytest.approx(8.0)
    assert _row(mirrors, 0, "OSE_M2_T30").elapsed_days == pytest.approx(4.0)


def test_a_mirror_with_fewer_than_two_finite_points_is_excluded_not_nan_propagated(mirrors):
    row = _row(mirrors, 0, "Mirror_5")

    assert row.n_points == 0
    assert np.isnan(row.soiling_rate_pp_day)
    assert len(mirrors) == 7  # the row is kept, so nothing silently disappears


def test_soiling_rate_is_loss_over_elapsed_days_in_pp_per_day(mirrors):
    row = _row(mirrors, 0, "ONE_M1_T00")
    expected = (0.95 / NOMINAL_0 - 0.91 / NOMINAL_0) * 100 / 8.0

    assert row.loss_pp == pytest.approx((0.95 / NOMINAL_0 - 0.91 / NOMINAL_0) * 100)
    assert row.soiling_rate_pp_day == pytest.approx(expected)


def test_soiling_rate_is_computed_fresh_not_read_from_the_precomputed_field(mirrors):
    """reflect_data.soiling_rate is positional and on raw rho, so it is not NaN-safe and carries no
    nominal correction; the stub sets it to 999 to prove it is not consulted."""
    assert _row(mirrors, 0, "ONE_M1_T00").soiling_rate_pp_day != pytest.approx(999.0)


def test_raw_rho_endpoints_are_reported_alongside_the_soiling_factor(mirrors):
    row = _row(mirrors, 0, "ONE_M1_T00")

    assert (row.rho_initial, row.rho_final) == (pytest.approx(0.95), pytest.approx(0.91))


def test_nominal_reflectance_fallback_degrades_to_one_when_the_value_is_blank(monkeypatch, cfg):
    """float(nan) does not raise, so a blank cell would otherwise pass for a value -- yadnarie's
    workbook has exactly that."""
    monkeypatch.setattr(es.pd, "read_excel", lambda *a, **k: pd.DataFrame({"Value": [np.nan]}, index=["nominal_reflectance"]))

    assert es.nominal_reflectance_fallback(cfg) == (1.0, "none")


def test_nominal_reflectance_fallback_reads_a_usable_value(monkeypatch, cfg):
    monkeypatch.setattr(es.pd, "read_excel", lambda *a, **k: pd.DataFrame({"Value": [0.965]}, index=["nominal_reflectance"]))

    assert es.nominal_reflectance_fallback(cfg) == (0.965, "parameter file")


# ---------------------------------------------------------------------------
# 3. The two tables
# ---------------------------------------------------------------------------


def test_soiling_summary_has_one_row_per_campaign_with_mean_std_and_display_columns(data, mirrors):
    summary = es.summary_table(data, mirrors)

    assert list(summary["campaign"]) == ["Yadnarie 11-11-24", "Yadnarie 06-02-25"]
    assert {"loss_pp_mean", "loss_pp_std", "loss_pp"} <= set(summary.columns)
    assert summary.loc[0, "loss_pp"] == es.format_mean_std(summary.loc[0, "loss_pp_mean"], summary.loc[0, "loss_pp_std"])


def test_soiling_summary_counts_only_usable_mirrors(data, mirrors):
    summary = es.summary_table(data, mirrors)

    assert list(summary["n_mirrors"]) == [4, 2]  # Mirror_5 has no data
    assert list(summary["n_measurements"]) == [5, 3]


def test_soiling_summary_duration_days_can_exceed_a_mirrors_own_window(data, mirrors):
    """The gap is the late mirror; both numbers are reported so it is visible."""
    summary = es.summary_table(data, mirrors)

    assert summary.loc[0, "duration_days"] == pytest.approx(8.0)
    assert _row(mirrors, 0, "OSE_M2_T30").elapsed_days < summary.loc[0, "duration_days"]


def test_soiling_by_tilt_has_one_row_per_campaign_and_tilt(mirrors):
    by_tilt = es.by_tilt_table(mirrors)

    assert list(by_tilt["campaign"]) == ["Yadnarie 11-11-24"] * 4 + ["Yadnarie 06-02-25"] * 2
    assert list(by_tilt["tilt_deg"]) == [0.0, 30.0, 45.0, 60.0, 0.0, 30.0]


def test_soiling_by_tilt_orientation_is_na_when_names_carry_none(mirrors):
    by_tilt = es.by_tilt_table(mirrors)
    qut_like = by_tilt[(by_tilt["tilt_deg"] == 45.0)].iloc[0]

    assert qut_like["orientation"] == es.NO_ORIENTATION
    assert by_tilt[by_tilt["tilt_deg"] == 30.0].iloc[0]["orientation"] == "SE"


def test_soiling_by_tilt_single_mirror_group_prints_the_mean_alone(mirrors):
    by_tilt = es.by_tilt_table(mirrors)
    single = by_tilt[by_tilt["tilt_deg"] == 45.0].iloc[0]

    assert single["n_mirrors"] == 1
    assert np.isnan(single["loss_pp_std"])
    assert "±" not in single["loss_pp"]


def test_soiling_by_tilt_mirror_counts_sum_to_the_campaign_total(data, mirrors):
    summary = es.summary_table(data, mirrors)
    by_tilt = es.by_tilt_table(mirrors)

    per_campaign = by_tilt.groupby("campaign")["n_mirrors"].sum()
    for _, row in summary.iterrows():
        assert per_campaign.get(row["campaign"], 0) == row["n_mirrors"]


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------


@pytest.fixture
def ticks():
    return ["11-11-24", "06-02-25"]


def test_plot_soiling_rate_vs_tilt_draws_one_series_per_campaign(mirrors, ticks):
    fig = es.plot_soiling_rate_vs_tilt(mirrors, ticks, "Yadnarie")
    ax = fig.axes[0]

    assert [text.get_text() for text in ax.get_legend().get_texts()] == ticks
    assert ax.get_ylabel() == "Soiling rate [p.p./day]"
    plt.close(fig)


def test_plot_soiling_rate_vs_tilt_offsets_campaigns_so_a_shared_tilt_stays_visible(mirrors, ticks):
    """Both campaigns measure tilt 30; without an offset they would overplot exactly. Tilt 30 rather
    than 0 because the zero reference line's own xdata is [0, 1] in axes coordinates."""
    fig = es.plot_soiling_rate_vs_tilt(mirrors, ticks, "Yadnarie")
    ax = fig.axes[0]
    near_30 = {round(float(x), 6) for line in ax.lines for x in line.get_xdata() if abs(float(x) - 30.0) < 1.0}

    assert len(near_30) == 2  # one nudged position per campaign
    assert 30.0 not in near_30
    plt.close(fig)


def test_plot_soiling_rate_vs_tilt_omits_the_legend_for_a_single_campaign(mirrors, ticks):
    one = mirrors[mirrors["campaign_index"] == 0]
    fig = es.plot_soiling_rate_vs_tilt(one, ticks, "Yadnarie")

    assert fig.axes[0].get_legend() is None
    plt.close(fig)


def test_plot_soiling_rate_by_orientation_groups_bars_by_campaign(mirrors, ticks):
    fig = es.plot_soiling_rate_by_orientation(mirrors, ["yadnarie/a.xlsx"], ticks, "Yadnarie")
    ax = fig.axes[0]

    assert [text.get_text() for text in ax.get_xticklabels()] == ticks
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_soiling_rate_by_orientation_returns_none_when_no_mirror_has_one(mirrors, ticks):
    """The qut case: no orientation anywhere means no figure, not an empty one."""
    anonymous = mirrors.copy()
    anonymous["orientation"] = es.NO_ORIENTATION

    assert es.plot_soiling_rate_by_orientation(anonymous, ["qut/a.xlsx"], ticks, "QUT") is None


def test_plot_reflectance_series_draws_one_panel_per_campaign(data, mirrors, ticks):
    fig = es.plot_reflectance_series(data, mirrors, ticks, "Yadnarie", FALLBACK)

    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ticks
    plt.close(fig)


def test_plot_reflectance_series_handles_a_single_campaign(data, mirrors, ticks):
    """squeeze=False is what keeps a one-campaign site from returning a bare Axes."""
    data.files = ["a.xlsx"]
    fig = es.plot_reflectance_series(data, mirrors, ticks, "Yadnarie", FALLBACK)

    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_reflectance_series_does_not_normalise_the_first_point_to_one(data, mirrors, ticks):
    """plot_for_paper shifts each trace to start at 1; doing that here would contradict the raw rho
    the table reports and would blank any mirror whose first sample is NaN."""
    fig = es.plot_reflectance_series(data, mirrors, ticks, "Yadnarie", FALLBACK)
    drawn = {round(float(y), 6) for line in fig.axes[0].lines for y in np.asarray(line.get_ydata()) if np.isfinite(y)}

    assert 0.95 in drawn and 0.91 in drawn  # raw reflectance, unshifted
    assert 1.0 not in drawn
    plt.close(fig)


def test_plot_reflectance_series_marks_the_nominal_reflectance(data, mirrors, ticks):
    fig = es.plot_reflectance_series(data, mirrors, ticks, "Yadnarie", FALLBACK)
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]

    assert "nominal (reference)" in labels  # campaign 0 has one
    assert "nominal (parameter file)" in [text.get_text() for text in fig.axes[1].get_legend().get_texts()]
    plt.close(fig)


@pytest.mark.parametrize("key", ["xtick.labelsize", "legend.fontsize", "axes.labelsize"])
def test_figures_do_not_leak_the_paper_rcparams(data, mirrors, ticks, key):
    before = plt.rcParams[key]
    for fig in [
        es.plot_soiling_rate_vs_tilt(mirrors, ticks, "Yadnarie"),
        es.plot_soiling_rate_by_orientation(mirrors, ["yadnarie/a.xlsx"], ticks, "Yadnarie"),
        es.plot_reflectance_series(data, mirrors, ticks, "Yadnarie", FALLBACK),
    ]:
        plt.close(fig)

    assert plt.rcParams[key] == before


# ---------------------------------------------------------------------------
# 5. Wiring
# ---------------------------------------------------------------------------


def test_write_outputs_skips_the_soiling_section_without_reflectance_data(cfg, data, tmp_path):
    import logging

    data.reflect_data_total = None

    assert es.write_outputs(cfg, data, str(tmp_path), logging.getLogger("test"), ["a", "b"], "Yadnarie") is None
