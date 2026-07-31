"""
Measured-soiling section of the experimental location assessment workflow.

Summarises what the mirrors actually did during each campaign, from the reflectance measurements
alone -- no fitted model anywhere. It is the companion to the weather section in ``experiment``: the
two write to the same run folder and agree on `campaign` and `duration_days`, so a campaign's
meteorology and its soiling can be read side by side.

Per run it adds, to results/experiment/{site}/{run_name}/:
  - soiling_summary.csv : one row per campaign, mirrors aggregated;
  - soiling_by_tilt.csv : one row per campaign and tilt;
  - soiling_rate_vs_tilt.pdf, soiling_rate_by_orientation.pdf, reflectance_series.pdf;
  - campaign_{f+1}/reflectance_tilt_NN.pdf : the same measurements as reflectance_series, one figure
    per campaign and tilt, so the mirrors in it differ only by orientation. Same folder and filename
    the simulate workflow uses for its measured-vs-predicted counterpart.

Everything derives from one tidy per-mirror frame (`mirror_table`); both tables are groupby
aggregations of it and every figure is a filter on it.

Loss is measured on the *soiling factor* rho/rho_nominal rather than on raw reflectance, so campaigns
that started at different reflectance are comparable and, at the two sites with a re-cleaned
reference mirror (carwarp, yadnarie), day-to-day instrument drift is divided out. Raw rho is reported
alongside it. The rate is the whole-campaign endpoint rate: first to last *observed* measurement,
per mirror. Per mirror rather than per campaign because a mirror with a gap at either end would
otherwise be voided outright -- as it is in heliosoil.utilities' own positional
`reflectance_data.soiling_rate`, which this module therefore does not use.

Column names follow heliosoil.utilities.soiling_rates_summary, the (broken, unused) precedent for
this table: Elapsed Time (days) -> elapsed_days, Tilt -> tilt_deg, Initial/Final Reflectance ->
rho_initial/rho_final, Total Loss -> loss_pp, Soiling Rate -> soiling_rate_pp_day. Spelling is
lowercase-underscore to match weather_summary.csv.
"""

import os
import re
import dataclasses

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heliosoil.utilities as smu
import heliosoil.paper_specific_utilities as psu

from . import model_pipeline as mp
from .campaign_summary import FIGURE_RC, campaign_duration_days, campaign_label, format_mean_std, summarize

SUMMARY_CSV = "soiling_summary.csv"
BY_TILT_CSV = "soiling_by_tilt.csv"

# Not "N/A", which _orientation_colors uses internally: pandas lists that among its default
# na_values, so a CSV written with it reads back as NaN and the distinction is lost. Nothing here
# indexes those maps directly (both lookups go through .get with a default), so the spelling is free.
NO_ORIENTATION = "unknown"
_ORIENTATION_CODES = frozenset({"N", "S", "E", "W", "NE", "SE", "SW", "NW"})
# Compass order, so a legend reads as a compass rather than as workbook column order.
_COMPASS_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

_TILT_IN_NAME = re.compile(r"_T0*(\d+)")
# A mirror tilted past vertical faces the ground; real at carwarp (OE_M1_T180) but worth saying out
# loud, because it stretches the tilt axis and pools into an orientation bar with upward-facing ones.
_TILT_FACING_DOWN = 90.0

# The quantities every table reports, as (column stem, whether it gets mean/std/display columns).
_METRIC_KEYS = ["elapsed_days", "rho_initial", "rho_final", "sf_initial", "sf_final", "loss_pp", "soiling_rate_pp_day"]

VS_TILT_CAPTION = "marker: one mirror; line: per-tilt mean ± 1 s.d.; whole-campaign first-to-last endpoint rate"
BY_ORIENTATION_CAPTION = (
    "bar: mean over that orientation's mirrors ± 1 s.d.; ●: one mirror; number in bar: n\n"
    "groups pool every tilt, so compare orientations only where their tilts match (see soiling_by_tilt.csv "
    "and reflectance_by_tilt.pdf)"
)
AT_TILT_CAPTION = "trace: one mirror at this tilt, labelled by orientation; bars: 95% CI; raw ρ, unshifted"


@dataclasses.dataclass
class SoilingOutputs:
    """The per-mirror frame and the two tables derived from it."""

    mirrors: pd.DataFrame
    summary: pd.DataFrame
    by_tilt: pd.DataFrame


# ------------------------------ mirror geometry ------------------------------


def mirror_orientation(name) -> str | None:
    """Compass code from a mirror column name: "OSE_M2_T30" -> "SE", or None when the name does not
    encode one.

    A whitelist rather than heliosoil.horizontal_impaction.parse_orientation_names' raise, because a
    summary table must degrade rather than abort: qut's mirrors are named "Mirror_1".."Mirror_5",
    which the same split yields "irror" for."""
    code = str(name).split("_")[0][1:].upper()
    return code if code in _ORIENTATION_CODES else None


def mirror_tilts(rdat, f: int, mirror_names: list, log=None) -> np.ndarray:
    """Per-mirror tilt [deg] for campaign `f`, as a (n_mirrors,) array.

    Read from the Tilts sheet with a NaN-safe reduction, falling back to the tilt encoded in the
    mirror name. Tilt is constant per mirror at every site, so the median over time is exact rather
    than an approximation.

    The usual idiom is tilts[f][:, 0], which is NaN for a mirror installed after the campaign opened
    -- yadnarie's four OSE_* columns carry 182 leading NaNs in the untrimmed sheet. On the data this
    workflow sees the two agree, because load_data trims to the window where every mirror is valid
    and the leading NaNs are cut away with it; the median is the safe form for anyone who calls this
    on an untrimmed campaign, or for a gap that outlives the trim."""
    n = len(mirror_names)
    tilts = np.full(n, np.nan)

    sheet = getattr(rdat, "tilts", None) or {}
    if f in sheet:
        rows = np.asarray(sheet[f], dtype=float)
        for m in range(min(n, rows.shape[0])):
            finite = rows[m][np.isfinite(rows[m])]
            if finite.size:
                tilts[m] = float(np.median(finite))
                if log is not None and float(np.ptp(finite)) > 0.5:
                    log.warning(f"{mirror_names[m]}: tilt varies by {np.ptp(finite):.1f}° over the campaign; using the median {tilts[m]:.1f}°.")
    elif log is not None:
        log.warning(f"campaign {f}: no Tilts sheet data; falling back to the tilt encoded in each mirror name.")

    for m, name in enumerate(mirror_names):
        match = _TILT_IN_NAME.search(str(name))
        from_name = float(match.group(1)) if match else np.nan
        if not np.isfinite(tilts[m]):
            tilts[m] = from_name
        elif log is not None and np.isfinite(from_name) and abs(from_name - tilts[m]) > 1.0:
            log.warning(f"{name}: Tilts sheet says {tilts[m]:.1f}° but the name says {from_name:.0f}°; using the sheet.")

    if log is not None:
        for m, tilt in enumerate(tilts):
            if np.isfinite(tilt) and tilt > _TILT_FACING_DOWN:
                log.warning(f"{mirror_names[m]}: tilt {tilt:.0f}° is past vertical (facing downward).")
    return tilts


# ------------------------------ the per-mirror frame ------------------------------


def nominal_reflectance_fallback(cfg: mp.PipelineConfig, log=None) -> tuple[float, str]:
    """(value, source) for the scalar nominal reflectance used where a campaign has no reference
    mirror, read from the site's parameter workbook.

    Degrades to (1.0, "none") rather than raising: that makes the soiling factor equal raw
    reflectance, which the `nominal_source` column then says out loud -- better than killing an
    otherwise-complete run, and better than NaN, which would silently poison every number.

    A blank cell is treated the same as a missing row. float(nan) does not raise, so without this
    check a blank would pass for a value: yadnarie's workbook has exactly that, harmless today only
    because both its campaigns carry a reference mirror and never reach the fallback."""
    reason = None
    try:
        table = pd.read_excel(cfg.parameter_file, index_col="Parameter")
        value = float(table.loc["nominal_reflectance"].Value)
        if np.isfinite(value) and value > 0:
            return value, "parameter file"
        reason = f"nominal_reflectance is {value}"
    except Exception as err:
        reason = f"{type(err).__name__}: {err}"
    if log is not None:
        log.warning(f"no usable nominal_reflectance in {cfg.parameter_file} ({reason}); the soiling factor falls back to raw reflectance.")
    return 1.0, "none"


def mirror_table(cfg: mp.PipelineConfig, data: mp.LoadedData, *, fallback: float | None = None, log=None) -> pd.DataFrame:
    """One row per (campaign, mirror): its geometry, its endpoints, and its endpoint soiling rate.

    `fallback` is the scalar nominal reflectance for campaigns without a reference mirror; None reads
    it from the parameter workbook (injectable so callers -- and tests -- can avoid the Excel read)."""
    rdat, sim = data.reflect_data_total, data.sim_data_total
    fallback_source = "parameter file"
    if fallback is None:
        fallback, fallback_source = nominal_reflectance_fallback(cfg, log=log)

    rows = []
    for f in range(len(data.files)):
        label = campaign_label(cfg.location, sim, f)
        names = list(rdat.mirror_names[f])
        rho = np.asarray(rdat.average[f], dtype=float)
        times = np.asarray(rdat.times[f])
        tilts = mirror_tilts(rdat, f, names, log=log)

        nominal = smu._nominal_reflectance_series(rdat, f, fallback)
        has_reference = nominal is not fallback
        source = "reference mirror" if has_reference else fallback_source
        denominator = np.asarray(nominal, dtype=float)
        sf = rho / np.where(denominator > 0, denominator, np.nan)

        if log is not None:
            log.info(f"{label}: {len(names)} mirror(s), {len(times)} measurement time(s), nominal reflectance from {source}.")

        for m, name in enumerate(names):
            row = {
                "campaign": label,
                "campaign_index": f,
                "mirror": name,
                "tilt_deg": tilts[m],
                "orientation": mirror_orientation(name) or NO_ORIENTATION,
                "nominal_source": source,
            }
            valid = np.flatnonzero(np.isfinite(sf[:, m]))
            row["n_points"] = int(valid.size)
            if valid.size < 2:
                if log is not None:
                    log.warning(f"{label}: {name} has fewer than 2 finite measurements; excluded from the soiling summary.")
                rows.append({**row, **dict.fromkeys(_METRIC_KEYS, np.nan)})
                continue

            i0, i1 = int(valid[0]), int(valid[-1])
            elapsed_days = float((times[i1] - times[i0]) / np.timedelta64(1, "D"))
            loss_pp = float((sf[i0, m] - sf[i1, m]) * 100.0)
            row.update(
                elapsed_days=elapsed_days,
                rho_initial=float(rho[i0, m]),
                rho_final=float(rho[i1, m]),
                sf_initial=float(sf[i0, m]),
                sf_final=float(sf[i1, m]),
                loss_pp=loss_pp,
                soiling_rate_pp_day=loss_pp / elapsed_days if elapsed_days > 0 else np.nan,
            )
            rows.append(row)

            if log is not None and i0 > 0:
                # The documented reason elapsed_days can differ from the campaign's duration_days.
                started = float((times[i0] - times[0]) / np.timedelta64(1, "D"))
                log.info(f"{label}: {name} first measured {started:.1f} d into the campaign; its rate uses its own {elapsed_days:.1f} d window.")
            if log is not None and i0 == 0 and i1 == len(times) - 1:
                # Cross-check against the library's own endpoint rate, which is on raw rho and is not
                # NaN-safe. They should agree here; a mismatch means one of the two moved.
                library = np.asarray(rdat.soiling_rate.get(f, []), dtype=float)
                raw_rate = (rho[i0, m] - rho[i1, m]) / elapsed_days * 100.0
                if m < library.size and np.isfinite(library[m]) and abs(library[m] - raw_rate) > 1e-9:
                    log.debug(f"{label}: {name} raw-rho endpoint rate {raw_rate:.6g} != reflect_data.soiling_rate {library[m]:.6g}.")

    return pd.DataFrame(rows)


# ------------------------------ the two tables ------------------------------


def _aggregate(frame: pd.DataFrame, keys: list) -> dict:
    """The mean/std/display triple for every metric, aggregated over `frame`'s mirrors."""
    out = {}
    for key in keys:
        mean, std, _n = summarize(frame[key])
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
        out[key] = format_mean_std(mean, std)
    return out


def summary_table(data: mp.LoadedData, mirrors: pd.DataFrame) -> pd.DataFrame:
    """One row per campaign, mirrors aggregated.

    `duration_days` is deliberately the same quantity weather_summary.csv reports, so the two tables
    join on `campaign` and the shared column proves they came from the same trimmed window. It can
    exceed `elapsed_days` when a mirror was installed mid-campaign -- both are reported so the gap is
    visible rather than hidden."""
    sim, rdat = data.sim_data_total, data.reflect_data_total
    rows = []
    for f in range(len(data.files)):
        frame = mirrors[mirrors["campaign_index"] == f]
        usable = frame[frame["n_points"] >= 2]
        rows.append(
            {
                "campaign": frame["campaign"].iloc[0] if len(frame) else campaign_label(data.site_name, sim, f),
                "duration_days": campaign_duration_days(sim, f),
                "n_mirrors": int(len(usable)),
                "n_measurements": int(len(rdat.times[f])),
                "nominal_source": frame["nominal_source"].iloc[0] if len(frame) else "none",
                **_aggregate(usable, _METRIC_KEYS),
            }
        )
    return pd.DataFrame(rows)


def by_tilt_table(mirrors: pd.DataFrame, log=None) -> pd.DataFrame:
    """One row per (campaign, tilt), mirrors at that tilt aggregated."""
    usable = mirrors[mirrors["n_points"] >= 2]
    if log is not None and usable["tilt_deg"].isna().any():
        unknown = sorted(usable[usable["tilt_deg"].isna()]["mirror"].unique())
        log.warning(f"no tilt could be determined for {unknown}; they are grouped under a blank tilt.")

    rows = []
    for (f, campaign, tilt), frame in usable.groupby(["campaign_index", "campaign", "tilt_deg"], dropna=False, sort=True):
        codes = sorted({c for c in frame["orientation"] if c != NO_ORIENTATION})
        rows.append(
            {
                "campaign": campaign,
                "campaign_index": f,
                "tilt_deg": tilt,
                "orientation": ",".join(codes) if codes else NO_ORIENTATION,
                "n_mirrors": int(len(frame)),
                **_aggregate(frame, _METRIC_KEYS),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # NaN tilts sort last rather than being dropped, so no mirror silently vanishes from the table.
    table = table.sort_values(["campaign_index", "tilt_deg"], na_position="last").reset_index(drop=True)
    return table.drop(columns="campaign_index")


# ------------------------------ figures ------------------------------
# Every figure is built inside plt.rc_context(FIGURE_RC) and returned unsaved: heliosoil's
# paper_specific_utilities raises the global type sizes on import, and rc_context is what keeps both
# that out of these figures and these figures' overrides out of the other workflows.

_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _campaign_style(i: int) -> tuple[str, str]:
    """Colour and marker for campaign `i`. Both, not just colour, so the figure survives greyscale."""
    return f"C{i}", _MARKERS[i % len(_MARKERS)]


def plot_soiling_rate_vs_tilt(mirrors: pd.DataFrame, tick_labels: list, site: str) -> plt.Figure:
    """Endpoint soiling rate against mirror tilt, one series per campaign.

    Tilt is on a numeric axis rather than a categorical one because the figure's whole claim is that
    the rate varies physically with tilt; the cost is that a mirror past vertical (carwarp's 180°)
    stretches the axis."""
    usable = mirrors[(mirrors["n_points"] >= 2) & mirrors["tilt_deg"].notna()]
    campaigns = sorted(mirrors["campaign_index"].unique())
    tilts_present = sorted(usable["tilt_deg"].unique())
    spread = max(float(np.ptp(tilts_present)), 1.0) if tilts_present else 1.0

    with plt.rc_context(FIGURE_RC):
        fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * max(len(tilts_present), 1) + 3.0), 4.0))
        for i, f in enumerate(campaigns):
            frame = usable[usable["campaign_index"] == f]
            if frame.empty:
                continue
            color, marker = _campaign_style(i)
            # A deterministic offset, essential where campaigns share a tilt set (qut's four
            # campaigns are all measured at 0/15/30/45/65).
            dx = (i - (len(campaigns) - 1) / 2) * 0.012 * spread
            ax.plot(frame["tilt_deg"] + dx, frame["soiling_rate_pp_day"], marker, ls="none", ms=4, alpha=0.45, color=color, zorder=2)
            grouped = frame.groupby("tilt_deg")["soiling_rate_pp_day"]
            means, stds = grouped.mean(), grouped.std()
            ax.errorbar(means.index + dx, means, yerr=stds, marker=marker, ms=5, capsize=3, lw=1.2, color=color, label=tick_labels[f], zorder=3)

        # Rain-washed campaigns give negative rates, so the sign has to be readable.
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
        if tilts_present:
            ax.set_xticks(tilts_present)
        ax.set_xlabel("Tilt [°]")
        ax.set_ylabel("Soiling rate [p.p./day]")
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)
        if len(campaigns) > 1:  # a one-entry legend just repeats the title
            ax.legend()
        ax.set_title(VS_TILT_CAPTION, color="0.35")
        fig.suptitle(f"{site} — Soiling rate vs tilt")
        fig.tight_layout()
    return fig


def plot_soiling_rate_by_orientation(mirrors: pd.DataFrame, files: list, tick_labels: list, site: str) -> plt.Figure | None:
    """Endpoint soiling rate grouped by campaign and split by mirror orientation.

    Returns None when no mirror at this site encodes an orientation (qut names its mirrors
    "Mirror_1".."Mirror_5"), so the caller writes no file rather than an empty figure."""
    usable = mirrors[(mirrors["n_points"] >= 2) & (mirrors["orientation"] != NO_ORIENTATION)]
    if usable.empty:
        return None

    campaigns = sorted(mirrors["campaign_index"].unique())
    present = [code for code in _COMPASS_ORDER if code in set(usable["orientation"])]
    present += sorted(set(usable["orientation"]) - set(present))  # anything exotic, after the compass
    colors = psu._orientation_colors(files)
    width = 0.8 / max(len(present), 1)

    with plt.rc_context(FIGURE_RC):
        fig, ax = plt.subplots(figsize=(max(4.5, 1.6 * len(campaigns) + 2.0), 4.0))
        for j, code in enumerate(present):
            offset = (j - (len(present) - 1) / 2) * width
            heights, errors, positions, counts = [], [], [], []
            for g, f in enumerate(campaigns):
                values = usable[(usable["campaign_index"] == f) & (usable["orientation"] == code)]["soiling_rate_pp_day"]
                mean, std, n_finite = summarize(values)
                heights.append(mean)
                errors.append(std)
                positions.append(g + offset)
                counts.append(n_finite)
                # The individual mirrors, so a mean of two is not read as a solid measurement.
                ax.plot([g + offset] * len(values), values, "o", ms=3, color="0.2", alpha=0.6, zorder=3)
            ax.bar(
                positions,
                heights,
                width=width,
                yerr=errors,
                capsize=3,
                color=colors.get(code, "0.4"),
                edgecolor="white",
                label=psu._ORIENTATION_NAMES.get(code, code),
                zorder=2,
            )
            # A bar over a single mirror is not a mean; saying n in the bar makes that unmissable.
            for x, n_finite in zip(positions, counts):
                if n_finite:
                    ax.annotate(str(n_finite), (x, 0), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=6, color="white")

        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
        ax.set_xticks(range(len(campaigns)))
        ax.set_xticklabels([tick_labels[f] for f in campaigns])
        ax.set_xlim(-0.5, len(campaigns) - 0.5)
        ax.set_ylabel("Soiling rate [p.p./day]")
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(ncol=min(len(present), 4))
        ax.set_title(BY_ORIENTATION_CAPTION, color="0.35")
        fig.suptitle(f"{site} — Soiling rate by orientation")
        fig.tight_layout()
    return fig


def _disambiguate(stems: list) -> list:
    """`stems` with a trailing index on any label that occurs more than once, in the order given.

    Geometry does not identify a mirror on its own -- two mirrors can share an orientation and a tilt
    -- so without this a legend would name two traces the same and stop being a map onto them."""
    return [f"{stem} ({stems[:i].count(stem) + 1})" if stems.count(stem) > 1 else stem for i, stem in enumerate(stems)]


def series_labels(frame: pd.DataFrame, names: list) -> list:
    """Legend labels for one campaign's mirrors: geometry ("SE 30°") rather than the workbook column
    name ("OSE_M2_T30"), which says the same two things in a spelling only the workbook cares about.

    A mirror missing half its geometry is labelled by the half it has, and by its raw name only when
    it has neither -- qut's "Mirror_4" with no tilt on the sheet and none in the name."""
    stems = []
    for name in names:
        code = frame["orientation"].get(name, NO_ORIENTATION)
        tilt = frame["tilt_deg"].get(name, np.nan)
        parts = [code if code != NO_ORIENTATION else "", f"{tilt:.0f}°" if np.isfinite(tilt) else ""]
        stems.append(" ".join(part for part in parts if part) or str(name))
    return _disambiguate(stems)


def plot_reflectance_series(data: mp.LoadedData, mirrors: pd.DataFrame, tick_labels: list, site: str, fallback: float) -> plt.Figure:
    """Measured reflectance against time, one panel per campaign, one series per mirror.

    y is raw measured reflectance, deliberately NOT shifted to start at 1 the way plot_for_paper and
    plot_reflectance_by_tilt do. That normalisation is a model-comparison device; here it would
    contradict the raw rho the table reports and would blank every mirror installed mid-campaign,
    whose first sample is NaN."""
    rdat = data.reflect_data_total
    n = len(data.files)
    colors = psu._orientation_colors(data.files)
    styles = ["-", "--", "-.", ":"]

    with plt.rc_context(FIGURE_RC):
        # squeeze=False keeps a single-campaign site (ablrf) returning an array, not a bare Axes.
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 1.0, 4.2), sharey=True, squeeze=False)
        axes = axes[0]
        for f, ax in enumerate(axes):
            times = np.asarray(rdat.times[f])
            days = (times - times[0]) / np.timedelta64(1, "D")
            rho = np.asarray(rdat.average[f], dtype=float)
            sigma = np.asarray(rdat.sigma_of_the_mean[f], dtype=float)
            frame = mirrors[mirrors["campaign_index"] == f].set_index("mirror")
            names = list(rdat.mirror_names[f])
            labels = series_labels(frame, names)

            for m, name in enumerate(names):
                code = frame["orientation"].get(name, NO_ORIENTATION)
                color = colors.get(code, f"C{m}") if code != NO_ORIENTATION else f"C{m}"
                label = labels[m]
                # NaN points are simply not drawn, so a mirror installed mid-campaign starts partway
                # across its panel -- the truth, rather than a gap-filled line.
                ax.errorbar(
                    days, rho[:, m], yerr=1.96 * sigma[:, m], marker="o", ms=3, lw=1, capsize=3, color=color, ls=styles[m % len(styles)], label=label
                )

            nominal = smu._nominal_reflectance_series(rdat, f, fallback)
            if nominal is not fallback:
                ax.plot(days, np.asarray(nominal, dtype=float)[:, 0], color="0.6", ls=":", lw=1, label="nominal (reference)")
            else:
                ax.axhline(fallback, color="0.6", ls=":", lw=1, label="nominal (parameter file)")

            ax.set_title(tick_labels[f])
            ax.set_xlabel("Days since first measurement")
            ax.grid(alpha=0.4)
            ax.set_axisbelow(True)
            # Per-panel, not figure-level: carwarp and yadnarie measure different mirror sets in
            # different campaigns, so one shared legend would be actively wrong. "best" because the
            # traces descend by a campaign-dependent amount, so no fixed corner is always free.
            ax.legend(fontsize=7, ncol=2, loc="best")

        axes[0].set_ylabel("Reflectance ρ [-]")
        fig.suptitle(f"{site} — Measured reflectance")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def tilt_figure_name(tilt: float) -> str:
    """Filename of one campaign's measured-reflectance figure at one tilt.

    Deliberately simulate.tilt_figure_name's stem minus the role/fold suffix, which names the fit
    that drew a figure and has no counterpart in a workflow with no fit: a campaign_{f+1} folder
    here and one there then hold the same filenames for the same tilts."""
    return f"reflectance_tilt_{tilt:02.0f}.pdf"


def plot_reflectance_at_tilt(
    data: mp.LoadedData, mirrors: pd.DataFrame, f: int, tilt: float, campaign: str, site: str, fallback: float
) -> plt.Figure | None:
    """Measured reflectance against time for every mirror of campaign `f` at one tilt, or None when
    that campaign has no mirror there.

    The per-tilt cut of plot_reflectance_series, and the answer to the caveat printed on
    soiling_rate_by_orientation.pdf -- its bars pool every tilt, so orientations may only be compared
    where their tilts match. Here the figure *is* one tilt within one campaign, so its traces differ
    by orientation and by nothing else, and the title says what was held fixed.

    The measured-data counterpart of paper_specific_utilities.plot_reflectance_by_tilt, which the
    simulate workflow writes into the same campaign_{f+1}/reflectance_tilt_NN.pdf slot. Two
    deliberate differences, both because there is no model here: no dashed prediction and no fit
    statistics, and y is raw rho rather than each trace shifted to start at 1 -- that shift is a
    model-comparison device, and here it would contradict the raw rho the tables report and blank any
    mirror whose first sample is NaN."""
    rdat = data.reflect_data_total
    # n_points >= 1 rather than the tables' >= 2: one measurement still draws a marker, and only a
    # mirror with nothing at all should be missing from its tilt's figure.
    frame = mirrors[(mirrors["campaign_index"] == f) & (mirrors["tilt_deg"] == tilt) & (mirrors["n_points"] >= 1)]
    if frame.empty:
        return None

    times = np.asarray(rdat.times[f])
    days = (times - times[0]) / np.timedelta64(1, "D")
    rho = np.asarray(rdat.average[f], dtype=float)
    sigma = np.asarray(rdat.sigma_of_the_mean[f], dtype=float)
    columns = {name: m for m, name in enumerate(rdat.mirror_names[f])}
    colors = psu._orientation_colors(data.files)

    # Orientation alone: the tilt is in the title, so repeating it in every legend entry would be
    # noise. A mirror without one keeps its name, which is all that is left to say about it.
    stems = [code if code != NO_ORIENTATION else str(name) for name, code in zip(frame["mirror"], frame["orientation"])]

    with plt.rc_context(FIGURE_RC):
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        for (_, mirror), label in zip(frame.iterrows(), _disambiguate(stems)):
            m = columns[mirror["mirror"]]
            code = mirror["orientation"]
            color = colors.get(code, f"C{m}") if code != NO_ORIENTATION else f"C{m}"
            ax.errorbar(days, rho[:, m], yerr=1.96 * sigma[:, m], marker="o", ms=4, lw=1.2, capsize=3, color=color, label=label)

        nominal = smu._nominal_reflectance_series(rdat, f, fallback)
        if nominal is not fallback:
            ax.plot(days, np.asarray(nominal, dtype=float)[:, 0], color="0.6", ls=":", lw=1, label="nominal (reference)")
        else:
            ax.axhline(fallback, color="0.6", ls=":", lw=1, label="nominal (parameter file)")

        ax.set_xlabel("Days since first measurement")
        ax.set_ylabel("Reflectance ρ [-]")
        ax.grid(alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.set_title(AT_TILT_CAPTION, color="0.35")
        fig.suptitle(f"{site} — {campaign}, tilt {tilt:.0f}°")
        fig.tight_layout()
    return fig


def write_tilt_figures(data: mp.LoadedData, mirrors: pd.DataFrame, run_dir: str, tick_labels: list, site: str, fallback: float, log=None) -> list:
    """Write campaign_{f+1}/reflectance_tilt_NN.pdf for every (campaign, tilt) present and return
    the paths written, relative to `run_dir`.

    Tilts come from the per-mirror frame rather than from a fixed list, so there is nothing for a
    caller to mis-specify and a campaign gets exactly the tilts it measured."""
    usable = mirrors[mirrors["tilt_deg"].notna() & (mirrors["n_points"] >= 1)]
    if log is not None and usable.empty:
        log.warning("no per-tilt reflectance figures: no mirror has both a tilt and a finite measurement.")

    written = []
    for f in sorted(usable["campaign_index"].unique()):
        frame = usable[usable["campaign_index"] == f]
        campaign_dir = os.path.join(run_dir, f"campaign_{f + 1}")
        for tilt in sorted(frame["tilt_deg"].unique()):
            # The date alone, as on the other figures' ticks: the site is already in the suptitle,
            # and campaign_label ("Yadnarie 12-11-24") would put it there twice.
            fig = plot_reflectance_at_tilt(data, mirrors, f, tilt, tick_labels[f], site, fallback)
            if fig is None:  # unreachable from here, but plot_reflectance_at_tilt is public
                continue
            os.makedirs(campaign_dir, exist_ok=True)
            name = tilt_figure_name(tilt)
            fig.savefig(os.path.join(campaign_dir, name), bbox_inches="tight")
            plt.close(fig)
            written.append(f"campaign_{f + 1}/{name}")
        if log is not None:
            log.info(f"{frame['campaign'].iloc[0]}: {len(frame['tilt_deg'].unique())} per-tilt reflectance figure(s) in campaign_{f + 1}/.")
    return written


# ------------------------------ entry point ------------------------------


def write_outputs(cfg: mp.PipelineConfig, data: mp.LoadedData, run_dir: str, log, tick_labels: list, site: str) -> SoilingOutputs | None:
    """Write the two soiling tables, the three site-wide figures and the per-campaign per-tilt
    reflectance figures, or return None when the site has no usable reflectance data to summarise."""
    if data.reflect_data_total is None:
        log.warning("no reflectance data loaded; the soiling summary is skipped.")
        return None

    fallback, _source = nominal_reflectance_fallback(cfg, log=log)
    # sigma_of_the_mean scales the error bars in reflectance_series.pdf but the experiment CLI does
    # not expose number_of_measurements, so the value in force belongs on the record.
    log.info(f"sigma_of_the_mean assumes number_of_measurements={cfg.number_of_measurements}.")

    mirrors = mirror_table(cfg, data, fallback=fallback, log=log)
    if mirrors.empty or not (mirrors["n_points"] >= 2).any():
        log.warning("no mirror has two or more finite measurements; the soiling summary is skipped.")
        return None

    summary = summary_table(data, mirrors)
    by_tilt = by_tilt_table(mirrors, log=log)
    summary.to_csv(f"{run_dir}/{SUMMARY_CSV}", index=False, float_format="%.4g")
    by_tilt.to_csv(f"{run_dir}/{BY_TILT_CSV}", index=False, float_format="%.4g")

    for _, row in summary.iterrows():
        log.info(f"{row['campaign']}: soiling rate {row['soiling_rate_pp_day']} p.p./day over {row['n_mirrors']} mirror(s).")

    fig = plot_soiling_rate_vs_tilt(mirrors, tick_labels, site)
    fig.savefig(f"{run_dir}/soiling_rate_vs_tilt.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plot_soiling_rate_by_orientation(mirrors, data.files, tick_labels, site)
    if fig is None:
        log.warning("soiling_rate_by_orientation.pdf skipped: this site's mirror names carry no orientation.")
    else:
        fig.savefig(f"{run_dir}/soiling_rate_by_orientation.pdf", bbox_inches="tight")
        plt.close(fig)

    fig = plot_reflectance_series(data, mirrors, tick_labels, site, fallback)
    fig.savefig(f"{run_dir}/reflectance_series.pdf", bbox_inches="tight")
    plt.close(fig)

    write_tilt_figures(data, mirrors, run_dir, tick_labels, site, fallback, log=log)

    return SoilingOutputs(mirrors=mirrors, summary=summary, by_tilt=by_tilt)
