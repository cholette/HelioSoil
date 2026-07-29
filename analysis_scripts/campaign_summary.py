"""
Shared nucleus of the experimental location assessment workflow: how a campaign is named, how a
sample of it is summarised, and how its figures are typeset.

These few names are all that the workflow's two halves -- the weather section in ``experiment`` and
the measured-soiling section in ``experiment_soiling`` -- have in common. They live here so the two
can share them without importing each other, which would be a cycle.

Nothing here reads data or draws anything; see the two sibling modules for that.
"""

import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd

# Site folder names are lowercase and unspaced, which .title() alone renders as "Mountisa"/"Qut".
SITE_DISPLAY_NAMES = {"ablrf": "ABLRF", "carwarp": "Carwarp", "mountisa": "Mount Isa", "qut": "QUT", "wodonga": "Wodonga", "yadnarie": "Yadnarie"}

# heliosoil.paper_specific_utilities raises the global tick/label/legend sizes to 16/18/14pt on
# import (they suit its 12x15in multi-panel paper figures) and model_pipeline imports it
# transitively. The workflow's figures are a third that size, so they set their own type explicitly
# rather than inherit it -- and rc_context keeps the override from leaking back into the other
# workflows. Every figure in either sibling module must be built inside `with plt.rc_context(FIGURE_RC)`.
FIGURE_RC = {"axes.labelsize": 11, "axes.titlesize": 8, "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.titlesize": 12, "legend.fontsize": 8}


def site_display_name(location: str) -> str:
    """Human-readable site name for labels and titles: "mountisa" -> "Mount Isa"."""
    return SITE_DISPLAY_NAMES.get(str(location).strip().lower(), str(location).replace("_", " ").title())


def campaign_label(location: str, sim, f: int) -> str:
    """Campaign name for tables and titles, e.g. "Yadnarie 11-11-24": the site plus the day the
    campaign's measurement window opens.

    The date is the first timestamp of the *trimmed* window. sim.start_datetime is set at import and
    never updated by trim_experiment_data, so it is stale by hours-to-days here."""
    return f"{site_display_name(location)} {pd.Timestamp(sim.time[f].iloc[0]):%d-%m-%y}"


def campaign_duration_days(sim, f: int) -> float:
    """Length of the campaign's measurement window in days."""
    return (sim.time[f].iloc[-1] - sim.time[f].iloc[0]) / np.timedelta64(1, "D")


def summarize(values, circular: bool = False) -> tuple[float, float, int]:
    """(mean, std, n_finite) over the finite samples of `values`.

    Wind direction is circular, so its mean and standard deviation are taken on the circle -- an
    arithmetic mean of 355 deg and 5 deg is 180 deg, i.e. exactly backwards. std is NaN for a single
    sample (no spread is defined), and both are NaN when there is nothing to summarise."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    if circular:
        mean = float(circmean(v, high=360, low=0))
        std = float(circstd(v, high=360, low=0)) if v.size > 1 else np.nan
    else:
        mean = float(np.mean(v))
        std = float(np.std(v, ddof=1)) if v.size > 1 else np.nan
    return mean, std, int(v.size)


def format_mean_std(mean: float, std: float) -> str:
    """Paper-ready display string, e.g. "24.3 ± 5.1"; the mean alone when there is no spread to
    report, and blank when there is no data at all."""
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(std):
        return f"{mean:.3g}"
    return f"{mean:.3g} ± {std:.3g}"
