"""
Unit tests for the experimental-location assessment workflow's summarising layer.

Everything here is pure -- no Excel I/O and no fitting -- driven by a stub with the same
int-keyed-dict shape as SimulationInputs. It covers the parts that are quietly wrong rather than
loud: the stale start_datetime trap, the circular-statistics wrap, NaN and missing-channel handling,
and the discovery order that figure filenames and CSV columns both follow.
"""

import types

import pytest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from analysis_scripts import experiment  # noqa: E402
from analysis_scripts import model_pipeline as mp  # noqa: E402

N_SAMPLES = 24


def _times(start: str) -> pd.Series:
    return pd.Series(pd.date_range(start, periods=N_SAMPLES, freq="h"))


N_CALM = 6


@pytest.fixture
def data():
    """Two campaigns of a site that records temperature, wind speed and wind direction but no
    humidity and no DNI, and that measured PM2.5 only in the first campaign.

    Campaign 0 opens with N_CALM still samples whose direction the logger wrote as 0 deg -- the
    yadnarie pattern, where those fake northerlies were an eighth of the campaign."""
    wind_speed = np.full(N_SAMPLES, 3.0)
    wind_speed[:N_CALM] = 0.0
    wind_direction = np.resize([80.0, 85.0, 95.0, 100.0], N_SAMPLES)
    wind_direction[:N_CALM] = 0.0

    sim = types.SimpleNamespace(
        time={0: _times("2024-11-11 20:00"), 1: _times("2025-02-06 14:00")},
        # Deliberately wrong: import-time values that trimming never updates.
        start_datetime={0: pd.Timestamp("2024-11-11 00:00"), 1: pd.Timestamp("2025-02-06 00:00")},
        air_temp={0: np.full(N_SAMPLES, 25.0), 1: np.full(N_SAMPLES, 30.0)},
        wind_speed={0: wind_speed, 1: np.full(N_SAMPLES, 4.0)},
        wind_direction={0: wind_direction, 1: np.full(N_SAMPLES, 180.0)},
        relative_humidity={},  # the attribute exists but the site has no such column
        dust_concentration_channels={0: {"PM10": np.full(N_SAMPLES, 20.0), "PM2.5": np.full(N_SAMPLES, 8.0)}, 1: {"PM10": np.full(N_SAMPLES, 12.0)}},
    )
    return mp.LoadedData(
        files=["campaign_a.xlsx", "campaign_b.xlsx"],
        training_intervals=None,
        all_mirrors=[],
        site_name="yadnarie",
        parameter_file="",
        sim_data_total=sim,
        reflect_data_total=None,
    )


@pytest.fixture
def cfg():
    return mp.PipelineConfig(location="yadnarie")


# ---------------------------------------------------------------------------
# 1. Labels and durations
# ---------------------------------------------------------------------------


def test_site_display_name_uses_the_curated_map():
    assert experiment.site_display_name("mountisa") == "Mount Isa"
    assert experiment.site_display_name("qut") == "QUT"


def test_site_display_name_falls_back_to_title_case_for_an_unknown_site():
    assert experiment.site_display_name("new_site") == "New Site"


def test_campaign_label_uses_the_trimmed_window_start_not_the_stale_start_datetime(data):
    """start_datetime is set at import and never updated by trim_experiment_data, so reading it here
    would silently date every campaign to before its measurements began."""
    assert experiment.campaign_label("yadnarie", data.sim_data_total, 0) == "Yadnarie 11-11-24"
    assert experiment.campaign_label("yadnarie", data.sim_data_total, 1) == "Yadnarie 06-02-25"


def test_campaign_duration_days_spans_the_trimmed_window(data):
    assert experiment.campaign_duration_days(data.sim_data_total, 0) == pytest.approx((N_SAMPLES - 1) / 24)


# ---------------------------------------------------------------------------
# 2. Statistics
# ---------------------------------------------------------------------------


def test_summarize_ignores_nans():
    mean, std, n = experiment.summarize([1.0, np.nan, 3.0, np.nan])

    assert (mean, n) == (2.0, 2)
    assert std == pytest.approx(np.sqrt(2))


def test_summarize_returns_nan_for_an_all_nan_campaign():
    mean, std, n = experiment.summarize([np.nan, np.nan])

    assert n == 0
    assert np.isnan(mean) and np.isnan(std)


def test_summarize_returns_nan_std_for_a_single_sample():
    mean, std, n = experiment.summarize([7.0, np.nan])

    assert (mean, n) == (7.0, 1)
    assert np.isnan(std)


def test_summarize_on_an_empty_campaign_is_blank_not_an_error():
    assert experiment.summarize(np.empty(0))[2] == 0


def test_summarize_wraps_wind_direction_around_north():
    """The arithmetic mean of these is 180 deg -- exactly backwards."""
    mean, std, _n = experiment.summarize([350.0, 355.0, 5.0, 10.0], circular=True)

    assert min(mean, 360.0 - mean) < 1.0
    assert std < 15.0


def test_format_mean_std_renders_the_plus_minus_string():
    assert experiment.format_mean_std(24.34, 5.12) == "24.3 ± 5.12"


def test_format_mean_std_drops_an_undefined_spread():
    assert experiment.format_mean_std(24.34, np.nan) == "24.3"


def test_format_mean_std_is_blank_without_data():
    assert experiment.format_mean_std(np.nan, np.nan) == ""


def test_dust_slug_is_filesystem_safe():
    assert experiment.dust_slug("PM2.5") == "pm2p5"
    assert experiment.dust_slug("TSP") == "tsp"
    assert experiment.dust_slug("PM17-PM10") == "pm17_minus_pm10"


# ---------------------------------------------------------------------------
# 3. Variable discovery
# ---------------------------------------------------------------------------


def test_collect_variables_orders_base_variables_before_pm_channels(data):
    assert [var.key for var in experiment.collect_variables(data)] == ["air_temp", "wind_speed", "wind_direction", "pm10", "pm2p5"]


def test_collect_variables_skips_variables_the_site_lacks(data):
    """An empty dict (relative_humidity) and a missing attribute (dni) both mean "no such column",
    and must not produce an empty figure or empty CSV columns."""
    keys = [var.key for var in experiment.collect_variables(data)]

    assert "relative_humidity" not in keys
    assert "dni" not in keys


def test_collect_variables_unions_pm_channels_across_campaigns(data):
    """PM2.5 was measured in only one of the two campaigns; it still earns a figure, with one box."""
    pm2p5 = next(var for var in experiment.collect_variables(data) if var.key == "pm2p5")

    assert pm2p5.values[0].size == N_SAMPLES
    assert pm2p5.values[1].size == 0


def test_collect_variables_drops_a_variable_no_campaign_has_data_for(data):
    data.sim_data_total.air_temp = {0: np.full(N_SAMPLES, np.nan), 1: np.full(N_SAMPLES, np.nan)}

    assert "air_temp" not in [var.key for var in experiment.collect_variables(data)]


# ---------------------------------------------------------------------------
# 3b. Calm masking: a vane reports nothing meaningful in still air
# ---------------------------------------------------------------------------


def test_calm_samples_are_blanked_from_wind_direction(data):
    wd = next(var for var in experiment.collect_variables(data) if var.key == "wind_direction")

    assert np.isnan(wd.values[0][:N_CALM]).all()  # the 0 deg sentinels are gone
    assert np.isfinite(wd.values[0][N_CALM:]).all()  # the real measurements are untouched
    assert wd.n_calm == [N_CALM, 0]


def test_calm_masking_leaves_linear_variables_alone(data):
    """A wind speed of zero is a real measurement; only the direction is undefined there."""
    ws = next(var for var in experiment.collect_variables(data) if var.key == "wind_speed")

    assert np.isfinite(ws.values[0]).all()
    assert (ws.values[0][:N_CALM] == 0.0).all()


def test_calm_masking_moves_the_reported_wind_direction(data):
    """The whole point: left in, the fake northerlies drag the circular mean away from where the
    wind actually blew (here 90 deg, dragged to ~68 deg by a quarter of the samples reading 0)."""
    wd = next(var for var in experiment.collect_variables(data) if var.key == "wind_direction")
    masked, _std, _n = experiment.summarize(wd.values[0], circular=True)
    unmasked, _std, _n = experiment.summarize(data.sim_data_total.wind_direction[0], circular=True)

    assert masked == pytest.approx(90.0, abs=1.0)
    assert unmasked < 75.0


def test_calm_mask_is_empty_when_the_site_records_no_wind_speed(data):
    data.sim_data_total.wind_speed = {}

    assert not experiment.calm_mask(data.sim_data_total, 0, N_SAMPLES).any()


def test_calm_mask_is_empty_when_wind_speed_is_on_a_different_grid(data):
    """Better an unmasked direction than one masked by a misaligned speed record."""
    data.sim_data_total.wind_speed = {0: np.zeros(N_SAMPLES + 5)}

    assert not experiment.calm_mask(data.sim_data_total, 0, N_SAMPLES).any()


# ---------------------------------------------------------------------------
# 4. The summary table
# ---------------------------------------------------------------------------


def test_summary_table_has_numeric_and_display_columns_per_variable(cfg, data):
    summary = experiment.summary_table(cfg, data, experiment.collect_variables(data))

    assert list(summary["campaign"]) == ["Yadnarie 11-11-24", "Yadnarie 06-02-25"]
    assert summary.loc[0, "air_temp_mean"] == 25.0
    assert summary.loc[0, "air_temp_std"] == 0.0
    assert summary.loc[0, "air_temp"] == "25 ± 0"


def test_summary_table_column_order_follows_the_variable_order(cfg, data):
    variables = experiment.collect_variables(data)
    summary = experiment.summary_table(cfg, data, variables)

    assert list(summary.columns) == ["campaign", "duration_days", *[c for var in variables for c in (*experiment.stat_columns(var), var.key)]]


def test_summary_table_names_circular_statistics_circmean_and_circstd(cfg, data):
    summary = experiment.summary_table(cfg, data, experiment.collect_variables(data))

    assert {"wind_direction_circmean", "wind_direction_circstd"} <= set(summary.columns)
    assert "wind_direction_mean" not in summary.columns
    assert summary.loc[1, "wind_direction_circmean"] == pytest.approx(180.0)


def test_summary_table_leaves_a_campaign_without_data_blank(cfg, data):
    summary = experiment.summary_table(cfg, data, experiment.collect_variables(data))

    assert np.isnan(summary.loc[1, "pm2p5_mean"])
    assert summary.loc[1, "pm2p5"] == ""


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------


def _violin_bodies(ax):
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def test_plot_variable_draws_a_violin_per_campaign_with_spread(data):
    wind_direction = next(var for var in experiment.collect_variables(data) if var.key == "wind_direction")
    fig = experiment.plot_variable(wind_direction, ["11-11-24", "06-02-25"], "Yadnarie")
    ax = fig.axes[0]

    # Campaign 0 varies so it gets a body; campaign 1 is a constant 180 deg and cannot have a
    # kernel density, but both campaigns keep their tick so neither silently disappears.
    assert len(_violin_bodies(ax)) == 1
    assert [text.get_text() for text in ax.get_xticklabels()] == ["11-11-24", "06-02-25"]
    assert ax.get_xlim() == (0.5, 2.5)
    plt.close(fig)


def test_plot_variable_survives_a_campaign_with_no_spread(data):
    """Rain that never fell is a column of exact zeros, whose kernel density is singular."""
    flat = experiment.WeatherVariable("rain_intensity", "Rain intensity", "mm/h", False, [np.zeros(N_SAMPLES), np.zeros(N_SAMPLES)], [0, 0])
    fig = experiment.plot_variable(flat, ["11-11-24", "06-02-25"], "Yadnarie")
    ax = fig.axes[0]

    assert _violin_bodies(ax) == []  # no body, but the medians and means are still drawn
    assert len(ax.lines) >= 2
    plt.close(fig)


def test_plot_variable_skips_a_campaign_without_data(data):
    pm2p5 = next(var for var in experiment.collect_variables(data) if var.key == "pm2p5")
    fig = experiment.plot_variable(pm2p5, ["11-11-24", "06-02-25"], "Yadnarie")
    ax = fig.axes[0]

    assert [text.get_text() for text in ax.texts] == ["no data"]  # campaign 1 has no PM2.5
    plt.close(fig)


def test_plot_variable_labels_wind_direction_with_compass_points(data):
    wind_direction = next(var for var in experiment.collect_variables(data) if var.key == "wind_direction")
    fig = experiment.plot_variable(wind_direction, ["11-11-24", "06-02-25"], "Yadnarie")
    ax = fig.axes[0]

    assert [text.get_text() for text in ax.get_yticklabels()] == ["N", "E", "S", "W", "N"]
    assert ax.get_ylim() == (0, 360)
    plt.close(fig)


def test_plot_wind_rose_draws_one_panel_per_campaign(data):
    pytest.importorskip("windrose")
    fig = experiment.plot_wind_rose(data, ["11-11-24", "06-02-25"], "Yadnarie")

    assert len(fig.axes) >= 2  # one rose per campaign (plus the legend's own axes)
    assert [ax.get_ylim() for ax in fig.axes[:2]].count(fig.axes[0].get_ylim()) == 2  # shared scale
    plt.close(fig)


def test_plot_wind_rose_is_skipped_when_the_site_records_no_wind_direction(data):
    pytest.importorskip("windrose")
    data.sim_data_total.wind_direction = {}

    assert experiment.plot_wind_rose(data, ["11-11-24", "06-02-25"], "Yadnarie") is None
