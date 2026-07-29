"""
Experimental location assessment workflow: the weather half.

Characterises the *site* rather than a model: for one location, it summarises the weather observed
during each campaign's reflectance-measurement window, so differences in soiling between campaigns
can be read against the meteorology that drove them. The measured-soiling half lives in
``experiment_soiling`` and is written to the same run folder from this module's ``write_outputs``.

Per run it writes, to results/experiment/{site}/{run_name}/:
  - one violin figure per weather variable, with one violin per campaign -- temperature, relative
    humidity, wind speed, wind direction, DNI and rain intensity (each only if the site records it),
    plus one figure per particulate-matter channel the site measures (PM10, PM2.5, PM17, TSP, ...);
  - wind_rose.pdf, one rose per campaign on a shared scale, where the site records wind direction;
  - weather_summary.csv: one row per campaign with its name ("Yadnarie 11-11-24"), duration in days,
    and, for every plotted variable, numeric mean/std columns plus a formatted "24.3 +/- 5.1" column;
  - soiling_summary.csv, soiling_by_tilt.csv and three soiling figures (see experiment_soiling).

Wind direction is treated as the circular, frequently multi-modal quantity it is: calm samples are
excluded (a vane reports nothing meaningful in still air, and some loggers write 0 deg for it), the
mean and spread are circular, and the rose is there because no single mean describes a distribution
with two modes.

The measurement window comes for free from the shared loader: model_pipeline.load_data ends with
trim_experiment_data(..., "reflectance_data"), so every weather array is already clipped to exactly
the period the reflectance measurements span. This module is a summarising/plotting layer on top of
the LoadedData it returns.

This module exposes ``run()`` -- it is driven by the unified CLI
(``python -m analysis_scripts.cli experiment ...``); see cli.py for the configuration surface and
model_pipeline.py for the shared loading kernel.
"""

import os
import dataclasses

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import model_pipeline as mp
from . import experiment_soiling
from .campaign_summary import FIGURE_RC, SITE_DISPLAY_NAMES, campaign_duration_days, campaign_label, format_mean_std, site_display_name, summarize

__all__ = ["FIGURE_RC", "SITE_DISPLAY_NAMES", "campaign_duration_days", "campaign_label", "format_mean_std", "site_display_name", "summarize"]

WORKFLOW = "experiment"
SUMMARY_CSV = "weather_summary.csv"

# (SimulationInputs attribute, axis label, unit, circular). Sites record different subsets of these
# -- qut has no wind direction or humidity, no site currently records DNI -- so the list is the
# vocabulary to look for, not a requirement (see collect_variables).
BASE_WEATHER_SPECS = [
    ("air_temp", "Temperature", "°C", False),
    ("relative_humidity", "Relative humidity", "%", False),
    ("wind_speed", "Wind speed", "m/s", False),
    ("wind_direction", "Wind direction", "°", True),
    ("dni", "DNI", "W/m²", False),
    ("rain_intensity", "Rain intensity", "mm/h", False),
]
DUST_UNIT = "µg/m³"

# A vane cannot report a meaningful direction in near-still air, and some loggers write 0 deg for
# it: at yadnarie every WD==0 sample is also WS==0, 12.6% of one campaign, which read as a due-north
# maximum that is not weather. Circular variables are therefore masked below this wind speed [m/s].
CALM_THRESHOLD = 0.5

VIOLIN_CAPTION = "violin: sample density, bar: quartiles, line: 5th-95th percentile, ○: median"
LINEAR_CAPTION = f"{VIOLIN_CAPTION}, ▲: mean"
CIRCULAR_CAPTION = f"{VIOLIN_CAPTION}, ◆: circular mean\ncalms (< {CALM_THRESHOLD} m/s) excluded; quartiles are linear in degrees and do not wrap"
ROSE_CAPTION = f"% of time per sector, coloured by wind speed [m/s]; calms (< {CALM_THRESHOLD} m/s) excluded"


@dataclasses.dataclass
class WeatherVariable:
    """One plotted quantity and its per-campaign samples.

    `key` is both the figure filename stem and the CSV column stem, so a column in the table and the
    PDF beside it are named the same thing."""

    key: str  # "air_temp", "pm2p5"
    label: str  # "Temperature", "PM2.5"
    unit: str
    circular: bool
    values: list  # one np.ndarray per campaign, aligned with range(len(data.files))
    n_calm: list = None  # per campaign, samples blanked as calm (circular variables only)


def dust_slug(name: str) -> str:
    """Filesystem- and column-safe stem for a dust channel: "PM2.5" -> "pm2p5", "TSP" -> "tsp",
    "PM17-PM10" -> "pm17_minus_pm10". Never emits "." or "-"."""
    return str(name).strip().lower().replace("-", "_minus_").replace(".", "p")


def stat_columns(var: WeatherVariable) -> tuple[str, str]:
    """Numeric column names for a variable's mean and standard deviation.

    Circular quantities get "_circmean"/"_circstd" so the header itself says the statistic is not the
    arithmetic one -- there is no separate legend that could drift out of sync with the data."""
    return (f"{var.key}_circmean", f"{var.key}_circstd") if var.circular else (f"{var.key}_mean", f"{var.key}_std")


def calm_mask(sim, f: int, n_samples: int) -> np.ndarray:
    """Boolean mask of the samples too still for a wind vane to be believed (see CALM_THRESHOLD).

    All-False when the site records no wind speed, or records it on a different grid than the
    variable being masked -- better to summarise an unmasked direction than a misaligned one."""
    speeds = getattr(sim, "wind_speed", None) or {}
    if f not in speeds:
        return np.zeros(n_samples, dtype=bool)
    ws = np.asarray(speeds[f], dtype=float)
    if ws.size != n_samples:
        return np.zeros(n_samples, dtype=bool)
    return np.isfinite(ws) & (ws < CALM_THRESHOLD)


def collect_variables(data: mp.LoadedData) -> list[WeatherVariable]:
    """The weather variables this site actually has data for, each with its per-campaign samples.

    Base variables come from BASE_WEATHER_SPECS, dust from the union of the campaigns'
    dust_concentration_channels -- union rather than intersection because a channel measured in two
    of three campaigns still deserves a figure with two boxes. A variable no campaign has finite
    samples for is dropped entirely rather than producing an empty figure and empty columns."""
    sim = data.sim_data_total
    n = len(data.files)
    variables = []

    for attr, label, unit, circular in BASE_WEATHER_SPECS:
        # `or {}` covers both "the attribute was never created" (the site's Weather sheet has no such
        # column) and "the dict exists but is empty".
        series = getattr(sim, attr, None) or {}
        values = [np.asarray(series[f], dtype=float) if f in series else np.empty(0) for f in range(n)]
        n_calm = [0] * n
        if circular:
            # Blank rather than drop, so the samples stay aligned with the campaign's time grid.
            masks = [calm_mask(sim, f, values[f].size) for f in range(n)]
            n_calm = [int(np.count_nonzero(mask & np.isfinite(values[f]))) for f, mask in enumerate(masks)]
            values = [np.where(mask, np.nan, v) for mask, v in zip(masks, values)]
        if any(np.isfinite(v).any() for v in values):
            variables.append(WeatherVariable(attr, label, unit, circular, values, n_calm))

    # dict.fromkeys keeps first-campaign Weather-column order while de-duplicating across campaigns.
    channels = [sim.dust_concentration_channels.get(f, {}) for f in range(n)]
    for name in dict.fromkeys(key for channel in channels for key in channel):
        values = [np.asarray(channel.get(name, np.empty(0)), dtype=float) for channel in channels]
        if any(np.isfinite(v).any() for v in values):
            variables.append(WeatherVariable(dust_slug(name), name, DUST_UNIT, False, values, [0] * n))

    return variables


def summary_table(cfg: mp.PipelineConfig, data: mp.LoadedData, variables: list[WeatherVariable]) -> pd.DataFrame:
    """One row per campaign: its name, the duration of its measurement window, and for every
    variable the numeric mean/std plus the formatted "mean +/- std" display string."""
    sim = data.sim_data_total
    rows = []
    for f in range(len(data.files)):
        row = {"campaign": campaign_label(cfg.location, sim, f), "duration_days": campaign_duration_days(sim, f)}
        for var in variables:
            mean, std, _n = summarize(var.values[f], var.circular)
            mean_col, std_col = stat_columns(var)
            row[mean_col] = mean
            row[std_col] = std
            row[var.key] = format_mean_std(mean, std)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_variable(var: WeatherVariable, tick_labels: list, site: str) -> plt.Figure:
    """Violin of one variable with one violin per campaign. Returns the figure; the caller saves it.

    A violin rather than a box because these distributions are not unimodal -- yadnarie's November
    wind direction has a northerly mode and an ESE-SE mode, and a box (or any single mean) hides
    that by reporting a centre that sits in the gap between them. The overlaid bar/line/markers
    carry the same quartiles, 5-95 range and mean the CSV reports."""
    n = len(var.values)
    finite = [v[np.isfinite(v)] for v in var.values]
    drawn = [i for i, v in enumerate(finite) if v.size]
    # A kernel density needs some spread; a campaign of one repeated value (rain that never fell)
    # gets the marker overlay alone rather than a degenerate body.
    spread = [i for i in drawn if np.ptp(finite[i]) > 0]

    with plt.rc_context(FIGURE_RC):
        # Width grows with the campaign count, but xlim is pinned to the full range so a
        # single-campaign site gets a normally proportioned violin rather than one stretched
        # across the axes.
        fig, ax = plt.subplots(figsize=(max(4.0, 1.4 * n + 2.0), 4.0))
        if spread:
            parts = ax.violinplot([finite[i] for i in spread], positions=[i + 1 for i in spread], widths=0.7, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor("C0")
                body.set_edgecolor("C0")
                body.set_alpha(0.35)

        for i in drawn:
            p5, q1, med, q3, p95 = np.percentile(finite[i], [5, 25, 50, 75, 95])
            ax.vlines(i + 1, p5, p95, color="0.35", linewidth=1, zorder=2)
            ax.vlines(i + 1, q1, q3, color="0.25", linewidth=4, zorder=2)
            ax.plot(i + 1, med, marker="o", markersize=4, markerfacecolor="white", markeredgecolor="0.25", zorder=3)
            # An arithmetic mean would be wrong for a circular quantity, so wind direction gets its
            # circular mean instead, drawn as a diamond to say the statistic differs.
            mean, _std, _n_finite = summarize(finite[i], var.circular)
            ax.plot(i + 1, mean, marker="D" if var.circular else "^", markersize=5, color="C3", zorder=4)

        if var.circular:
            ax.set_ylim(0, 360)
            ax.set_yticks([0, 90, 180, 270, 360])
            ax.set_yticklabels(["N", "E", "S", "W", "N"])

        for i in range(n):
            if i not in drawn:
                ax.annotate(
                    "no data", (i + 1, 0.5), xycoords=("data", "axes fraction"), ha="center", va="center", fontsize=8, color="0.5", rotation=90
                )

        ax.set_xlim(0.5, n + 0.5)
        ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel(f"{var.label} [{var.unit}]")
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_title(CIRCULAR_CAPTION if var.circular else LINEAR_CAPTION, color="0.35")
        fig.suptitle(f"{site} — {var.label}")
        fig.tight_layout()
    return fig


def plot_wind_rose(data: mp.LoadedData, tick_labels: list, site: str) -> plt.Figure | None:
    """One wind rose per campaign on a shared radial scale, or None if the site records no wind.

    The companion to the wind-direction violin: a rose is the only honest answer to "where does the
    wind come from", because it shows every mode instead of collapsing a multi-modal distribution
    onto one mean. Bars are % of time, split by wind speed, with calms excluded (see CALM_THRESHOLD)
    and reported per panel instead.

    heliosoil.utilities.wind_rose draws a single campaign per figure with its own scale; this lays
    the campaigns out side by side, on one scale, so they can actually be compared."""
    # Optional dependency, and only this figure needs it. Importing it is what registers the
    # "windrose" matplotlib projection used below, so the name itself is deliberately unused.
    import windrose  # noqa: F401

    sim = data.sim_data_total
    directions = getattr(sim, "wind_direction", None) or {}
    speeds = getattr(sim, "wind_speed", None) or {}
    n = len(data.files)
    panels = [f for f in range(n) if f in directions and f in speeds]
    if not panels:
        return None

    with plt.rc_context(FIGURE_RC):
        fig = plt.figure(figsize=(4.2 * len(panels) + 1.0, 4.6))
        axes = []
        for position, f in enumerate(panels, start=1):
            wd = np.asarray(directions[f], dtype=float)
            ws = np.asarray(speeds[f], dtype=float)
            keep = np.isfinite(wd) & np.isfinite(ws) & ~calm_mask(sim, f, wd.size)
            ax = fig.add_subplot(1, len(panels), position, projection="windrose")
            axes.append(ax)
            calm_pct = 100.0 * (1.0 - keep.sum() / max(wd.size, 1))
            if keep.any():
                ax.bar(wd[keep], ws[keep], normed=True, opening=0.8, edgecolor="white", nsector=16)
            ax.set_title(f"{tick_labels[f]}\ncalm {calm_pct:.0f}% of the time", pad=18)
            ax.tick_params(labelsize=7)

        # One radial scale across the panels, or a sector that dominates one campaign looks the
        # same size as a minor sector in another. The rings have to be shared too: left on their
        # own limits they end up bunched against the centre and label a scale that is no longer
        # the one drawn.
        top = max(ax.get_ylim()[1] for ax in axes)
        rings = np.linspace(0, top, 5)[1:]
        for ax in axes:
            ax.set_ylim(0, top)
            ax.set_yticks(rings)
            ax.set_yticklabels([f"{r:.0f}%" for r in rings])
        axes[-1].set_legend(title="m/s", bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8, title_fontsize=8)
        fig.suptitle(f"{site} — Wind rose")
        fig.text(0.5, 0.005, ROSE_CAPTION, ha="center", fontsize=8, color="0.35")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def write_outputs(cfg: mp.PipelineConfig, data: mp.LoadedData, run_dir: str, log) -> tuple[pd.DataFrame, list[WeatherVariable]]:
    """Write one {key}.pdf per variable (plus wind_rose.pdf) and the summary CSV, and return
    (summary, variables)."""
    variables = collect_variables(data)
    if not variables:
        raise ValueError(f"{cfg.location!r} has no usable weather data in its campaigns' measurement windows.")

    sim = data.sim_data_total
    labels = [campaign_label(cfg.location, sim, f) for f in range(len(data.files))]
    # The site name is identical on every violin and already in the title, so the ticks carry the
    # date alone -- repeating the site is what would force the labels to be rotated.
    tick_labels = [label.rsplit(" ", 1)[-1] for label in labels]
    site = site_display_name(cfg.location)

    for var in variables:
        for f, values in enumerate(var.values):
            n_finite = int(np.isfinite(values).sum())
            n_calm = (var.n_calm or [0] * len(var.values))[f]
            if n_calm:
                # Worth stating: at yadnarie this is an eighth of a campaign, and left in it reads
                # as a due-north maximum.
                log.info(f"{labels[f]}: {var.label}, {n_calm} samples blanked as calm (< {CALM_THRESHOLD} m/s).")
            if n_finite == 0:
                log.warning(f"{labels[f]}: no {var.label} data; leaving its violin and summary blank.")
            else:
                log.info(f"{labels[f]}: {var.label}, {n_finite} of {values.size} samples finite.")

        fig = plot_variable(var, tick_labels, site)
        fig.savefig(f"{run_dir}/{var.key}.pdf", bbox_inches="tight")
        plt.close(fig)

    try:
        rose = plot_wind_rose(data, tick_labels, site)
    except ImportError:
        # The rose is a supplement, not the deliverable: a site's violins and table should not be
        # lost because an optional plotting package is missing.
        log.warning("wind_rose.pdf skipped: the optional 'windrose' package is not installed.")
        rose = None
    if rose is not None:
        rose.savefig(f"{run_dir}/wind_rose.pdf", bbox_inches="tight")
        plt.close(rose)

    summary = summary_table(cfg, data, variables)
    summary.to_csv(os.path.join(run_dir, SUMMARY_CSV), index=False, float_format="%.4g")

    soiling = experiment_soiling.write_outputs(cfg, data, run_dir, log, tick_labels, site)
    return summary, variables, soiling


def _prepare_config(cfg: mp.PipelineConfig) -> mp.PipelineConfig:
    """Adjust the config to describe *measured* weather rather than model inputs.

    k_factor is a deposition-model tuning factor, not a measurement correction, so it is forced to
    1.0 (k_factor=None) and dust_concentration_channels then holds exactly the Weather-sheet columns.
    dust_type is pinned to a channel the site actually has: SimulationInputs falls through
    resolve_dust_concentration to weather[dust_type] and raises KeyError otherwise, and
    PipelineConfig defaults to PM10 while mountisa and qut are TSP-only."""
    if not os.path.isdir(cfg.data_dir):
        raise ValueError(f"no data directory for location {cfg.location!r} ({cfg.data_dir}).")
    available = mp.available_dust_types(cfg)
    if not available:
        raise ValueError(f"{cfg.location!r} has no particulate-matter columns in its campaigns' Weather sheets.")
    dust_type = cfg.dust_type if cfg.dust_type in available else available[0]
    return dataclasses.replace(cfg, dust_type=dust_type, k_factor=None)


def run(cfg: mp.PipelineConfig, *, run_name: str | None, force: bool, run_metadata: dict) -> None:
    """Summarise the weather each of `cfg`'s campaigns was measured under, and the soiling they
    actually recorded.

    Writes one violin per weather variable (a violin per campaign), the wind rose, the two soiling
    tables and the three soiling figures under results/experiment/{site}/{run_name}/, the run
    settings + version to run_config.json, and the per-campaign sample counts and any data gaps to
    run.log."""
    cfg = _prepare_config(cfg)  # must precede load_data: it pins a dust_type the site has
    data = mp.load_data(cfg)

    run_name = mp.resolve_run_dir(cfg, WORKFLOW, run_name, force=force)
    run_dir = mp.run_dir_path(cfg, WORKFLOW, run_name)
    # Record the config the run actually used: the CLI built run_metadata before _prepare_config
    # overrode dust_type and k_factor, and a reproducibility snapshot of settings that were not in
    # force would be worse than none.
    mp.write_run_metadata(run_dir, {**run_metadata, "config": dataclasses.asdict(cfg)})
    log, close_log = mp.open_run_log(run_dir)
    try:
        summary, variables, soiling = write_outputs(cfg, data, run_dir, log)
        print(f"\n{site_display_name(cfg.location)}: {len(data.files)} campaign(s), {len(variables)} weather variable(s)\n")
        print(summary[["campaign", "duration_days", *[var.key for var in variables]]].to_string(index=False))
        if soiling is not None:
            # Only the formatted display columns, matching the weather table; by_tilt is reported by
            # row count rather than printed (qut alone is 4 campaigns x 5 tilts).
            print(f"\nSoiling: {len(soiling.mirrors)} mirror-campaign(s), {len(soiling.by_tilt)} tilt group(s)\n")
            print(
                soiling.summary[["campaign", "n_mirrors", "elapsed_days", "rho_initial", "rho_final", "loss_pp", "soiling_rate_pp_day"]].to_string(
                    index=False
                )
            )
        print(f"\nWrote {run_dir}")
    finally:
        close_log()
