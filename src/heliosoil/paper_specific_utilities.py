import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import pickle
from typing import Union
import heliosoil.base_models as smb
import heliosoil.fitting as smf
from heliosoil.utilities import _nominal_reflectance_anchor, _nominal_reflectance_series, gravitational_settling_factor, logger

# %% Plot for the paper
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
plt.rc("legend", fontsize=14)
plt.rc("axes", labelsize=18)

_ORIENTATION_NAMES = {
    "N": "North",
    "S": "South",
    "E": "East",
    "W": "West",
    "NE": "Northeast",
    "SE": "Southeast",
    "SW": "Southwest",
    "NW": "Northwest",
    "N/A": "N/A",
}


def _prediction_variance(mod, e):
    """Campaign `e`'s soiling-factor prediction variance, or None when the fitted model has
    no noise process to draw an interval from.

    A least-squares fit estimates the mean parameters only -- the sum of squares does not
    depend on the sigmas (see horizontal_impaction.ConstantMeanWindDeposition.fit_ls) -- so
    predict_soiling_factor leaves the variance dict empty. Callers draw the mean prediction
    and simply omit the shaded interval rather than failing on the missing key."""
    variance = getattr(mod.helios, "soiling_factor_prediction_variance", None)
    return variance.get(e) if variance else None


def _orientation_colors(files):
    """Color-by-orientation map, keyed by which site name shows up in `files` (e.g.
    sdat.files). Shared by plot_for_paper and plot_reflectance_by_tilt so every
    figure colors a given orientation the same way."""
    if any("augusta" in f.lower() for f in files):
        return {"NW": "blue", "SE": "red"}
    if any("yadnarie" in f.lower() for f in files):
        return {"NE": "blue", "SE": "red", "SW": "green", "NW": "magenta", "N/A": "blue"}
    return {"N": "blue", "S": "red", "E": "green", "W": "magenta", "N/A": "blue"}


def plot_for_paper(
    mod,
    rdat,
    sdat,
    train_experiments,
    train_mirrors,
    orientation,
    rows_with_legend=[3],
    num_legend_cols=6,
    legend_shift=(0, 0),
    plot_rh=True,
    yticks=None,
    auto_yticks=False,
    ytick_spacing=0.05,
    ylabel_row=None,
    figsize=(12, 15),
    lgd_size=10,
    ci_alpha=0.1,
):
    """
    Plot reflectance data and model predictions for a set of experiments and mirror tilts.

    This function generates a figure with multiple subplots, where each subplot shows the
    reflectance data and model predictions for a specific experiment and mirror tilt. The
    figure also includes plots of the dust concentration and wind speed for each experiment.

    Parameters:
        mod (heliosoil.base_models.SoilingModel): The soiling model to use for the predictions.
        rdat (heliosoil.base_models.ReflectanceData): The reflectance data for the experiments.
        sdat (heliosoil.base_models.SimulationData): The simulation data for the experiments.
        train_experiments (list): The names of the experiments used for training the model.
        train_mirrors (list): The names of the mirrors used for training the model.
        orientation (list): The orientation of each mirror in the experiments.
        rows_with_legend (list, optional): Unused -- the y-label row is now `ylabel_row` and the
            legend is a single figure-level one. Kept so existing call sites keep working.
        num_legend_cols (int, optional): The number of columns in the legend.
        legend_shift (tuple, optional): A tuple specifying the x and y shift of the legend.
        plot_rh (bool, optional): Whether to plot the relative humidity.
        yticks (list, optional): Explicit y-axis tick values, applied to every tilt row. Takes
            precedence over auto_yticks, so passing both keeps the explicit ticks.
        auto_yticks (bool, optional): If True (and `yticks` is None), each tilt row gets its own
            tick set, spaced `ytick_spacing` apart, spanning that row's measured/predicted data.
            Ticks and limits are shared by every column of the row, so all campaigns in a row
            stay directly comparable.
        ytick_spacing (float, optional): Tick spacing used when auto_yticks is True.
        ylabel_row (int, optional): Tilt-row index carrying the "Normalized reflectance" y-label
            (leftmost column). Defaults to the middle tilt row.
        ci_alpha (float, optional): Unused -- the 95% prediction interval is now drawn as dotted
            bounds in each mirror's own colour rather than as a translucent fill, so there is no
            fill opacity to set. Kept so existing call sites keep working.

    Returns:
        tuple: The figure and axis objects for the generated plot.
    """

    if any("augusta".lower() in value.lower() for value in sdat.files):
        plot_rh = False  # the RH sensor is broken since the beginning of Port Augusta experiments

    mod.predict_soiling_factor(sdat, reflectance_data=rdat)  # ensure predictions are fresh

    exps = list(mod.helios.tilt.keys())
    tilts = np.unique(mod.helios.tilt[0])
    # Per-mirror nominal reflectance anchor (reference-mirror-derived when rdat
    # carries one, else the model's fixed constant broadcast to every mirror) --
    # indexed the same way as rdat.average/rdat.mirror_names (kk below), not the
    # model heliostat index (hh). np.broadcast_to makes the fallback-scalar case
    # indexable identically to the per-mirror-array case.
    r0_by_exp = {e: np.broadcast_to(_nominal_reflectance_anchor(rdat, e, mod.helios.nominal_reflectance), rdat.rho0[e].shape) for e in exps}

    if plot_rh:
        fig, ax = plt.subplots(nrows=len(tilts) + 2, ncols=len(exps), figsize=figsize, sharex="col")
    else:
        fig, ax = plt.subplots(nrows=len(tilts) + 1, ncols=len(exps), figsize=figsize, sharex="col")

    dust_max = max([max(sdat.dust_concentration[f]) for f in exps])  # max dust concentration for setting y-axes

    if plot_rh:
        hum_max = max([max(sdat.relative_humidity[f]) for f in exps])  # max relative humidity for setting y-axes

    # Define color for each orientation
    colors = _orientation_colors(sdat.files)

    # Data extent of each tilt row, pooled over every column/mirror in the row, for auto_yticks.
    # Measured (with its 2-sigma bars) and predicted means only: the prediction interval band is
    # decoration and can be far wider than the curves it shades.
    row_data_limits = {jj: [np.inf, -np.inf] for jj in range(len(tilts))}

    def _update_row_limits(jj, *arrays):
        for arr in arrays:
            arr = np.asarray(arr, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                row_data_limits[jj][0] = min(row_data_limits[jj][0], float(finite.min()))
                row_data_limits[jj][1] = max(row_data_limits[jj][1], float(finite.max()))

    ref_output = {}
    for ii, e in enumerate(exps):
        for jj, t in enumerate(tilts):
            tr = rdat.times[e]
            tr = (tr - tr[0]).astype("timedelta64[s]").astype(np.float64) / 3600 / 24

            (idx,) = np.where(rdat.tilts[e][:, 0] == t)
            (idxs,) = np.where(mod.helios.tilt[e][:, 0] == t)

            if not any(rdat.tilts[e][:, 0] == t):
                print("Tilt Not Found")
                continue

            if t == 0 and any("augusta".lower() in value.lower() for value in sdat.files):
                idx = idx[1:]  # In the Port Augusta data the first mirror is cleaned every time and used as control reference
            if t == 0 and any("augusta".lower() in value.lower() for value in sdat.files):
                idxs = idxs[1:]  # In the Port Augusta data the first mirror is cleaned every time and used as control reference

            if t == 0 and any("mildura".lower() in value.lower() for value in sdat.files):
                idx = idx[
                    2:
                ]  # In the Mildura data the first mirror is cleaned every time and used as control reference and the 2nd is used for Heliostat comparison
            if t == 0 and any("mildura".lower() in value.lower() for value in sdat.files):
                idxs = idxs[
                    2:
                ]  # In the Mildura data the first mirror is cleaned every time and used as control reference and the 2nd is used for Heliostat comparison

            ts = sdat.time[e].values[0 : rdat.prediction_indices[e][-1] + 1]
            ts = (ts - ts[0]).astype("timedelta64[s]").astype(np.float64) / 3600 / 24

            # idx (reflectance-data mirror index) and idxs (model heliostat index) are
            # parallel arrays -- both built from the same imported mirror list, in the
            # same order -- so zip pairs each mirror's measurement with its own prediction.
            for kk, hh in zip(idx, idxs):  # measured (solid) + predicted (dashed), per mirror
                m = rdat.average[e][:, kk].squeeze().copy()
                s = rdat.sigma_of_the_mean[e][:, kk].squeeze()
                m += 1 - m[0]  # shift up so that all start at 1.0 for visual comparison
                error_two_sigma = 1.96 * s

                color = colors[orientation[ii][kk]]
                orientation_name = _ORIENTATION_NAMES.get(orientation[ii][kk], orientation[ii][kk])
                if np.ndim(ax) == 1:
                    ax = np.vstack(ax)  # create a fictious 2D array with only one column
                ax[jj, ii].errorbar(tr, m, yerr=error_two_sigma, label=orientation_name, color=color, linestyle="-")
                _update_row_limits(jj, m - error_two_sigma, m + error_two_sigma)

                if (e in train_experiments) and (rdat.mirror_names[e][kk] in train_mirrors):
                    a = ax[jj, e]
                    a.axvline(x=tr[0], ls=":", color="red")
                    a.axvline(x=tr[-1], ls=":", color="red")
                    a.patch.set_facecolor(color="yellow")
                    a.patch.set_alpha(0.2)

                # This mirror's own prediction: for tilt-only models (constant-mean,
                # semi-physical) every mirror at this tilt predicts identically, so these
                # lines overlap exactly; for wind-driven models (Delta_gamma = azimuth -
                # wind_direction) they genuinely differ by orientation. Unlabeled here (the
                # per-orientation legend entry comes from the solid measured line above);
                # the "Predicted" line-style key is added once, separately, below.
                yh = r0_by_exp[e][kk] * mod.helios.soiling_factor[e][hh, 0 : rdat.prediction_indices[e][-1] + 1]
                yh = yh + (1.0 - yh[0])
                ax[jj, ii].plot(ts, yh, color=color, linestyle="--")
                _update_row_limits(jj, yh)

                # THIS mirror's 95% prediction interval, as dotted bounds in the mirror's own
                # colour. It used to be a single black filled band taken from one representative
                # mirror, which said neither whose interval it was nor how wide a coverage it
                # meant. With two or three orientations sharing a panel, one translucent fill per
                # orientation would compound into a grey mass over the very curves being compared,
                # so the bounds are drawn as lines instead: colour keeps carrying orientation
                # (nothing new to learn) and dotted separates the interval from the dashed mean.
                # Omitted entirely for a model with no noise process (see _prediction_variance).
                variance = _prediction_variance(mod, e)
                if variance is not None:
                    sigma_predict = r0_by_exp[e][kk] * np.sqrt(variance[hh, 0 : rdat.prediction_indices[e][-1] + 1])
                    ax[jj, ii].plot(ts, yh - 1.96 * sigma_predict, color=color, linestyle=":", linewidth=1.0)
                    ax[jj, ii].plot(ts, yh + 1.96 * sigma_predict, color=color, linestyle=":", linewidth=1.0)
                    # Deliberately not fed to _update_row_limits: the interval can be several times
                    # the height of the curves it brackets, and letting it set the row's y-range
                    # would flatten every measured series into the middle of the panel.

            # Representative reference output for this (campaign, tilt), unshifted -- what callers
            # read back out of the figure. Unrelated to the intervals drawn above.
            ref_output[(e, int(t))] = (r0_by_exp[e][idx[0]] * mod.helios.soiling_factor[e][idxs[0], 0 : rdat.prediction_indices[e][-1] + 1]).copy()
            ax[jj, ii].grid("on")

            if jj == 0:
                ax[jj, ii].set_title(f"Campaign {e + 1}, Tilt: {t:.0f}" + r"$^{\circ}$")
            else:
                ax[jj, ii].set_title(rf"Tilt: {t:.0f}" + r"$^{\circ}$")

        new_var = sdat.dust_concentration[e][0 : rdat.prediction_indices[e][-1] + 1]
        dust_conc = new_var
        ws = sdat.wind_speed[e][0 : rdat.prediction_indices[e][-1] + 1]
        ws_max = np.max(ws)
        dust_type = sdat.dust_type[e][0 : rdat.prediction_indices[e][-1] + 1]
        if plot_rh:
            a2 = ax[-2, ii]
        else:
            a2 = ax[-1, ii]

        a2.plot(ts, dust_conc, color="brown")
        a2.tick_params(axis="y", labelcolor="brown")
        a2.set_ylim((0, 1.01 * dust_max))

        a2a = a2.twinx()
        a2a.plot(ts, ws, color="green")
        a2a.tick_params(axis="y", labelcolor="green")
        a2a.set_ylim((0, 1.01 * ws_max))
        # a2a.set_ylim((0, 30))
        a2a.set_yticks((0, ws_max / 2, ws_max))
        a2a.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        if plot_rh:
            # THE LINE BELOW AVOID THE LAST ELEMENT (SO IT HAS THE SAME DIMENSION)
            rel_hum = sdat.relative_humidity[e][0 : rdat.prediction_indices[e][-1] + 1]
            a3 = ax[-1, ii]
            a3.plot(ts, rel_hum, color="blue")
            a3.tick_params(axis="y", labelcolor="blue")
            a3.set_ylim((0, 1.01 * hum_max))

        if ii == 0:
            fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
            a2.set_ylabel(fs.format(dust_type), color="brown")

            if plot_rh:
                a3.set_ylabel("Relative \nHumidity (%)", color="blue")
        else:
            a2.set_yticklabels([])

        if ii == len(exps) - 1:
            fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
            a2a.set_ylabel("Wind Speed (m/s)", color="green")

    # for ii,row in enumerate(ax):
    # for jj,a in enumerate(row):
    #     if ii < len(tilts):
    #         if yticks is None:
    #             a.set_ylim((0.85,1.01))
    #             a.set_yticks((0.85,0.90,0.95,1.0))
    #         else:
    #             a.set_ylim((min(yticks),max(yticks)))
    #             a.set_yticks(yticks)

    #         if jj > 0:
    #             a.set_yticklabels([])
    #         if (ii in rows_with_legend) and (jj==0):
    #             ang = rdat.reflectometer_incidence_angle[jj]
    #             a.set_ylabel(r"Normalized reflectance $\rho(0)-\rho(t)$ at "+str(ang)+"$^{{\circ}}$")
    #         if (ii in rows_with_legend) and (jj==0):
    #             # a.legend(loc='center',ncol=2,bbox_to_anchor=(0.25,0.5))
    #             h_legend,labels_legend = a.get_legend_handles_labels()
    #     elif ii == len(tilts):
    #         a.set_yticks((0,150,300))
    #     else:
    #         a.set_yticks((0,50,100))

    #     if ii == ax.shape[0]-1:
    #         a.set_xlabel('Days')
    #     a.grid('on')

    handles_leg = []
    labels_leg = []

    # One tick set per tilt row, shared by every column of that row, so the row's campaigns stay
    # directly comparable while each row is free to span only the range its own data occupies.
    # `yticks` wins when both are given: an explicit tick list is a deliberate choice, so passing
    # one overrides auto_yticks rather than being silently ignored.
    auto_row_ticks = {}
    tick_decimals = max(2, int(np.ceil(-np.log10(ytick_spacing))))
    if auto_yticks and yticks is None:
        for jj in range(len(tilts)):
            vmin, vmax = row_data_limits[jj]
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                continue  # tilt absent from the data (see the "Tilt Not Found" skip above)
            # Limits hug the data (plus a small margin); ticks are the exact multiples of
            # ytick_spacing that fall inside. Snapping the *limits* out to multiples instead
            # would donate a whole empty interval whenever the data just clears a tick.
            pad = max(0.05 * (vmax - vmin), 0.05 * ytick_spacing)
            lo, hi = vmin - pad, vmax + pad
            ticks = np.arange(np.ceil(lo / ytick_spacing), np.floor(hi / ytick_spacing) + 1) * ytick_spacing
            if ticks.size < 2:
                # Row too narrow to contain two ticks at this spacing: stretch the limits just
                # far enough to reach the two multiples nearest the data (rather than out to
                # whole intervals on both sides, which would leave a large empty band).
                k = np.floor(hi / ytick_spacing)
                ticks = np.array([(k - 1) * ytick_spacing, k * ytick_spacing])
                lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
            auto_row_ticks[jj] = (np.round(ticks, tick_decimals), (lo, hi))

    # Centre the y-label on the block of tilt rows instead of an arbitrary legend row -- with an
    # odd number of tilts this is the exact middle row, which is where the label reads as centred.
    label_row = (len(tilts) - 1) // 2 if ylabel_row is None else ylabel_row

    for ii, row in enumerate(ax):
        for jj, a in enumerate(row):
            if ii < len(tilts):
                if ii in auto_row_ticks:
                    ticks, ylim = auto_row_ticks[ii]
                    a.set_ylim(ylim)
                    a.set_yticks(ticks)
                    a.yaxis.set_major_formatter(FormatStrFormatter(f"%.{tick_decimals}f"))
                elif yticks is None:
                    a.set_ylim((0.85, 1.01))
                    a.set_yticks((0.85, 0.90, 0.95, 1.0))
                else:
                    a.set_ylim((min(yticks), max(yticks)))
                    a.set_yticks(yticks)

                if jj > 0:
                    a.set_yticklabels([])

                if ii == label_row and jj == 0:
                    ang = rdat.reflectometer_incidence_angle[jj]
                    a.set_ylabel(r"Normalized reflectance $\rho(0)-\rho(t)$ at " + str(ang) + r"$^{\circ}$")

                # Get legend handles and labels
                hs, ls = a.get_legend_handles_labels()
                handles_leg.extend(hs)
                labels_leg.extend(ls)

                # # Remove duplicates from the legend
                # for handle, label in zip(h_legend, labels_legend):
                #     if label not in seen_labels:
                #         seen_labels.add(label)
                #     else:
                #         # Remove the duplicate handle from the list
                #         h_legend.remove(handle)
                #         labels_legend.remove(label)

            elif ii == len(tilts):
                a.set_yticks((0, 150, 300))
            else:
                a.set_yticks((0, 50, 100))

            # Bottom row carries the shared (sharex="col") time axis, whichever panel it is:
            # relative humidity when plot_rh, otherwise dust concentration / wind speed.
            if ii == ax.shape[0] - 1:
                a.set_xlabel("Days")

    # Remove duplicates by filtering unique labels and keeping corresponding handles
    unique_labels = []
    unique_handles = []

    for handle, label in zip(handles_leg, labels_leg):
        if label not in unique_labels:  # Check if the label is already in the unique list
            unique_labels.append(label)  # Add unique label
            unique_handles.append(handle)  # Add corresponding handle

    # Line-style key, separate from the per-orientation color key above: neutral color
    # since solid/dashed (not color) is what distinguishes measured from predicted.
    unique_handles.append(Line2D([0], [0], color="black", linestyle="-"))
    unique_labels.append("Measured")
    unique_handles.append(Line2D([0], [0], color="black", linestyle="--"))
    unique_labels.append("Predicted")
    # Names the coverage as well as the role: "Prediction Interval" alone left the reader to guess
    # whether the shaded region was one sigma, two, or something else.
    unique_handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=1.0))
    unique_labels.append("95% prediction interval")

    fig.legend(
        unique_handles,
        unique_labels,
        ncol=num_legend_cols,
        bbox_to_anchor=(0.9025 + legend_shift[0], 1.025 + legend_shift[1]),
        bbox_transform=fig.transFigure,
    )
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    fig.tight_layout()
    return fig, ax, ref_output


def plot_reflectance_by_tilt(mod, rdat, sdat, experiment_index, tilt, orientation_codes, ax=None, figsize=(8, 5), ci_alpha=0.15):
    """
    Plot measured (solid, with 95% CI error bars) vs. predicted (dashed) reflectance
    for every mirror at a single tilt angle, within a single campaign -- a focused,
    one-tilt-at-a-time counterpart to plot_for_paper's full grid.

    Args:
        mod: a fitted soiling model exposing helios.soiling_factor /
            helios.soiling_factor_prediction_variance / helios.nominal_reflectance
            (predictions are refreshed in-place via mod.predict_soiling_factor).
        rdat (heliosoil.base_models.ReflectanceMeasurements): reflectance data,
            with import_tilts=True so rdat.tilts is populated.
        sdat (heliosoil.base_models.SimulationInputs): simulation data.
        experiment_index (int): campaign/file index (key into rdat/sdat/mod.helios).
        tilt (float): tilt angle [deg] to select mirrors by, matched against
            rdat.tilts[experiment_index][:, 0] (each mirror's tilt is assumed
            constant over the campaign, as in plot_for_paper).
        orientation_codes (list[str]): this campaign's per-mirror orientation code
            (e.g. from orientation_code()), in the same order as
            rdat.mirror_names[experiment_index].
        ax (matplotlib.axes.Axes, optional): axes to draw on; a new figure/axes is
            created (and tight_layout'd) if None.
        ci_alpha (float, optional): Unused -- the 95% prediction interval is drawn as dotted
            bounds per mirror rather than a fill. Kept so existing call sites keep working.

    Returns:
        (fig, ax, stats): stats is the regression_performance_stats dict (MBE/MAE/
        RMSE/R2/N) for this campaign restricted to mirrors at `tilt`. (None, None,
        None) if this campaign has no mirror at `tilt`.
    """
    mod.predict_soiling_factor(sdat, reflectance_data=rdat)  # ensure predictions are fresh
    e = experiment_index
    # Per-mirror nominal reflectance anchor, indexed like rdat.average/rdat.mirror_names
    # (kk below) -- see plot_for_paper for the same pattern.
    r0 = np.broadcast_to(_nominal_reflectance_anchor(rdat, e, mod.helios.nominal_reflectance), rdat.rho0[e].shape)

    (idx,) = np.where(rdat.tilts[e][:, 0] == tilt)
    (idxs,) = np.where(mod.helios.tilt[e][:, 0] == tilt)
    if len(idx) == 0:
        return None, None, None

    colors = _orientation_colors(sdat.files)
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    tr = rdat.times[e]
    tr = (tr - tr[0]).astype("timedelta64[s]").astype(np.float64) / 3600 / 24
    last_pred_idx = rdat.prediction_indices[e][-1]
    ts = sdat.time[e].values[0 : last_pred_idx + 1]
    ts = (ts - ts[0]).astype("timedelta64[s]").astype(np.float64) / 3600 / 24

    for kk, hh in zip(idx, idxs):  # measured (solid) + predicted (dashed), per mirror
        m = rdat.average[e][:, kk].squeeze().copy()
        s = rdat.sigma_of_the_mean[e][:, kk].squeeze()
        m += 1 - m[0]  # shift up so that all mirrors start at 1.0, for visual comparison

        code = orientation_codes[kk]
        orientation_name = _ORIENTATION_NAMES.get(code, code)
        color = colors.get(code, "black")
        ax.errorbar(tr, m, yerr=1.96 * s, label=f"{orientation_name} (measured)", color=color, linestyle="-", marker="o", markersize=3, capsize=3)

        yh = r0[kk] * mod.helios.soiling_factor[e][hh, 0 : last_pred_idx + 1]
        yh = yh + (1.0 - yh[0])
        ax.plot(ts, yh, color=color, linestyle="--", label=f"{orientation_name} (predicted)")

        # This mirror's own 95% prediction interval, as dotted bounds in its colour rather than
        # one black filled band from a representative mirror -- same reasoning as plot_for_paper.
        # Omitted for a model with no noise process (see _prediction_variance).
        variance = _prediction_variance(mod, e)
        if variance is not None:
            sigma_predict = r0[kk] * np.sqrt(variance[hh, 0 : last_pred_idx + 1])
            ax.plot(ts, yh - 1.96 * sigma_predict, color=color, linestyle=":", linewidth=1.0)
            ax.plot(ts, yh + 1.96 * sigma_predict, color=color, linestyle=":", linewidth=1.0)

    stats = regression_performance_stats(mod, rdat, [e], mirrors=idx)

    ax.set_xlabel("Days")
    ax.set_ylabel("Norm. reflectance")
    ax.set_title(f"Campaign {e + 1}, Tilt: {tilt:.0f}" + r"$^{\circ}$" + f"  (daily-rate MAE={stats['MAE']:.4f}, RMSE={stats['RMSE']:.4f})")
    ax.grid(True)
    # Line-style key for the dotted bounds, added once: colour already carries orientation, so
    # repeating every orientation a third time would triple the legend to say one thing.
    handles, labels = ax.get_legend_handles_labels()
    if _prediction_variance(mod, e) is not None:
        handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=1.0))
        labels.append("95% prediction interval")
    ax.legend(handles, labels, loc="best", fontsize=9)
    if created_fig:
        fig.tight_layout()

    return fig, ax, stats


def plot_for_heliostats(
    mod,
    rdat,
    sdat,
    train_experiments,
    train_mirrors,
    orientation,
    rows_with_legend=[3],
    num_legend_cols=6,
    legend_shift=(0, 0),
    plot_rh=True,
    yticks=None,
    figsize=None,
    lgd_size=10,
):
    mod.predict_soiling_factor(sdat, reflectance_data=rdat)  # ensure predictions are fresh

    exps = list(mod.helios.soiling_factor.keys())
    # tilts = np.unique(mod.helios.tilt[0]) # this is for the mirror rig with fixed tilt
    hels = rdat.mirror_names[0]  # [0] assuming all campaigns uses the same heliostats, otherwise modify here!

    if plot_rh:
        fig, ax = plt.subplots(nrows=len(hels) + 2, ncols=len(exps), figsize=(12, 15), sharex="col")
    else:
        fig, ax = plt.subplots(nrows=len(hels) + 1, ncols=len(exps), figsize=(12, 15), sharex="col")

    ws_max = max([max(sdat.wind_speed[f]) for f in exps])  # max wind speed for setting y-axes
    dust_max = max([max(sdat.dust_concentration[f]) for f in exps])  # max dust concentration for setting y-axes

    if plot_rh:
        hum_max = max([max(sdat.relative_humidity[f]) for f in exps])  # max relative humidity for setting y-axes

    ref_output = []
    for ii, e in enumerate(exps):
        exp_output = {}
        # Per-mirror nominal reflectance anchor, indexed like rdat.average (jj below,
        # shared here with the model heliostat index) -- see plot_for_paper.
        r0 = np.broadcast_to(_nominal_reflectance_anchor(rdat, e, mod.helios.nominal_reflectance), rdat.rho0[e].shape)
        for jj, t in enumerate(hels):
            tr = rdat.times[e]
            tr_day = (tr - tr[0]).astype("timedelta64[s]").astype(np.float64) / 3600 / 24

            ts = sdat.time[e].values[0 : rdat.prediction_indices[e][-1] + 1]
            ts_day = (ts - ts[0]).astype("timedelta64[s]").astype(np.float64) / 3600 / 24

            m = rdat.average[e][:, jj].squeeze().copy()
            s = rdat.sigma_of_the_mean[e][:, jj].squeeze()

            valid_idx = np.where(~np.isnan(m))[0]
            if np.isnan(m).any():
                tr = tr[valid_idx]
                m = m[valid_idx]
                s = s[valid_idx]
                tr_day = tr_day[valid_idx]
                ts_day = ts_day[(ts >= tr[0]) & (ts <= tr[-1])]
                ts = ts[(ts >= tr[0]) & (ts <= tr[-1])]

            m += 1 - m[0]  # shift up so that all start at 1.0 for visual comparison
            error_two_sigma = 1.96 * s

            if np.ndim(ax) == 1:
                ax = np.vstack(ax)  # create a fictious 2D array with only one column
            ax[jj, ii].errorbar(tr_day, m, yerr=error_two_sigma)

            ym = (
                r0[jj] * mod.helios.soiling_factor[e][jj, rdat.prediction_indices[e][valid_idx[0]] : rdat.prediction_indices[e][-1] + 1]
            )  # simulate reflectance losses between valid prediction indices
            # THE BELOW SIMPLY RESETS TO 0 THE CONFIDENCE INTERVAL IF A MEASUREMENT IS PERFORMED LATER - CHECK IF IT WORKS AS INTENDED BY MIKE
            var_predict = (
                mod.helios.soiling_factor_prediction_variance[e][jj, rdat.prediction_indices[e][valid_idx[0]] : rdat.prediction_indices[e][-1] + 1]
                - mod.helios.soiling_factor_prediction_variance[e][jj, rdat.prediction_indices[e][valid_idx[0]]]
            )

            exp_output[t] = {"Reflectance": ym.copy(), "Variance": var_predict.copy(), "Time": ts.copy()}

            if ym.ndim == 1:
                ym += 1.0 - ym[0]
            else:
                ym += 1.0 - ym[:, 0]

            sigma_predict = r0[jj] * np.sqrt(var_predict)
            Lp = ym - 1.96 * sigma_predict
            Up = ym + 1.96 * sigma_predict
            ax[jj, ii].plot(ts_day, ym, label="Prediction Mean", color="black")
            ax[jj, ii].fill_between(ts_day, Lp, Up, color="black", alpha=0.1, label=r"Prediction Interval")
            ax[jj, ii].grid("on")

            if jj == 0:
                ax[jj, ii].set_title(f"Campaign {e + 1}, Heliostat {t}")
            else:
                ax[jj, ii].set_title(f"Heliostat {t}")
        ref_output.append(exp_output)

        new_var = sdat.dust_concentration[e][0 : rdat.prediction_indices[e][-1] + 1]
        dust_conc = new_var
        ws = sdat.wind_speed[e][0 : rdat.prediction_indices[e][-1] + 1]
        dust_type = sdat.dust_type[e][0 : rdat.prediction_indices[e][-1] + 1]
        if plot_rh:
            a2 = ax[-2, ii]
        else:
            a2 = ax[-1, ii]

        a2.plot(ts_day, dust_conc, color="black")
        a2.tick_params(axis="y", labelcolor="black")
        a2.set_ylim((0, 1.01 * dust_max))

        a2a = a2.twinx()
        a2a.plot(ts_day, ws, color="green")
        a2a.tick_params(axis="y", labelcolor="green")
        a2a.set_ylim((0, 1.01 * ws_max))
        a2a.set_yticks((0, ws_max / 2, ws_max))
        a2a.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        if plot_rh:
            # THE LINE BELOW AVOID THE LAST ELEMENT (SO IT HAS THE SAME DIMENSION)
            rel_hum = sdat.relative_humidity[e][0 : rdat.prediction_indices[e][-1] + 1]
            a3 = ax[-1, ii]
            a3.plot(ts_day, rel_hum, color="blue")
            a3.tick_params(axis="y", labelcolor="blue")
            a3.set_ylim((0, 1.01 * hum_max))

        if ii == 0:
            fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
            a2.set_ylabel(fs.format(dust_type), color="black")

            if plot_rh:
                a3.set_ylabel("Relative \nHumidity (%)", color="blue")
        else:
            a2.set_yticklabels([])

        if ii == len(exps) - 1:
            fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
            a2a.set_ylabel("Wind Speed (m/s)", color="green")

    h_legend, labels_legend = [], []
    for ii, row in enumerate(ax):
        for jj, a in enumerate(row):
            if ii < len(hels):
                if yticks is None:
                    a.set_ylim((0.95, 1.01))
                    a.set_yticks((0.95, 0.97, 0.99, 1.01))
                else:
                    a.set_ylim((min(yticks), max(yticks)))
                    a.set_yticks(yticks)

                if jj > 0:
                    a.set_yticklabels([])
                if (ii in rows_with_legend) and (jj == 0):
                    ang = rdat.reflectometer_incidence_angle[jj]
                    a.set_ylabel(r"Normalized reflectance $\rho(0)-\rho(t)$ at " + str(ang) + r"$^{\circ}$")
                if (ii in rows_with_legend) and (jj == 0):
                    # a.legend(loc='center',ncol=2,bbox_to_anchor=(0.25,0.5))
                    h_legend, labels_legend = a.get_legend_handles_labels()
            elif ii == len(hels):
                a.set_yticks((0, 150, 300))
            else:
                a.set_yticks((0, 50, 100))

            if ii == ax.shape[0] - 1:
                a.set_xlabel("Days")
            a.grid("on")

    labels_legend, lab_idx = np.unique(labels_legend, return_index=True)
    fig.legend(
        [h_legend[ii] for ii in lab_idx],
        labels_legend,
        ncol=num_legend_cols,
        bbox_to_anchor=(0.9025 + legend_shift[0], 1.025 + legend_shift[1]),
        bbox_transform=fig.transFigure,
        fontsize=lgd_size,
    )

    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    fig.tight_layout()
    return fig, ax, ref_output


def soiling_rate(alphas: np.ndarray, alphas2: np.ndarray, save_file: str, M: int = 1000):
    """
    Simulates the soiling rate based on the given parameters and a saved model.

    Args:
        alphas (np.ndarray): Daily sum of alpha values corresponding to the specified percentiles.
        alphas2 (np.ndarray): Daily sum of alpha^2 values corresponding to the specified percentiles.
        save_file (str): File path to the saved model.
        M (int, optional): Number of simulations to run. Defaults to 1000.

    Returns:
        np.ndarray: Simulated daily soiling rates.
    """

    # load in parameters
    with open(save_file, "rb") as f:
        data = pickle.load(f)
        # sim_data_train = data["simulation_data"]
        imodel = data["model"]
        log_param_hat = data["transformed_parameters"]
        log_param_hat_cov = data["transformed_parameter_covariance"]

    # `type(...) is ...` (not isinstance) since ConstantMeanWindDeposition is also a
    # ConstantMeanDeposition subclass, but has a 5-parameter vector on a mixed
    # log/linear scale that np.exp(log_param_hat) below cannot unpack correctly.
    assert type(imodel) is smf.ConstantMeanDeposition, (
        "Model in saved file must be plain constant-mean type (not a subclass such as "
        "ConstantMeanWindDeposition, whose parameter vector has a different length/scale)."
    )
    mu_tilde, sigma_dep = np.exp(log_param_hat)

    # simulate
    sims = np.zeros((M, len(alphas)))
    inc_factor = imodel.helios.inc_ref_factor[0]
    for m in range(M):
        log_param = np.random.multivariate_normal(mean=log_param_hat, cov=log_param_hat_cov)
        mut, sigt = imodel.transform_scale(log_param)

        mean_loss_rate = inc_factor * mut * alphas  # loss rate in one timestep
        var_loss_rate = (inc_factor * sigt) ** 2 * alphas2  # loss variance in one timestep

        # sample and sum days
        s = np.sqrt(var_loss_rate)
        dt_loss = np.random.normal(loc=mean_loss_rate, scale=s)

        # add samples to flattened list
        sims[m, :] = dt_loss * 100

    return sims


def daily_soiling_rate(
    sim_dat: smb.SimulationInputs, model_save_file: str, percents: Union[list, np.ndarray] = None, M: int = 10000, dust_type="TSP"
):
    """
    Calculates the daily soiling rate based on simulation inputs and a saved model.

    Args:
        sim_dat (smb.simulation_inputs): Simulation input data.
        model_save_file (str): File path to the saved model.
        percents (list or np.ndarray, optional): Percentiles of interest for the daily sum of alpha and alpha^2. Defaults to None.
        M (int, optional): Number of simulations to run. Defaults to 10000.
        dust_type (str, optional): Type of dust to use. Defaults to "TSP".

    Returns:
        tuple:
            sims (np.ndarray): Simulated daily soiling rates.
            sa (np.ndarray): Daily sum of alpha values corresponding to the specified percentiles.
            sa2 (np.ndarray): Daily sum of alpha^2 values corresponding to the specified percentiles.
    """
    # This assumes a horizontal reflector

    # get daily sums for \alpha and \alpha^2
    df = [pd.read_excel(f, "Weather") for f in sim_dat.files]
    df = pd.concat(df)
    df.sort_values(by="Time", inplace=True)

    prototype_pm = getattr(sim_dat.dust, dust_type)[0]
    df["alpha"] = df[dust_type] / prototype_pm
    df["date"] = df["Time"].dt.date
    df["alpha2"] = df["alpha"] ** 2
    daily_sum_alpha = (df.groupby("date")["alpha"].sum()).values
    daily_sum_alpha2 = (df.groupby("date")["alpha2"].sum()).values

    # get daily sums corresponding to percentiles of interest
    idx = np.argsort(daily_sum_alpha)
    daily_sum_alpha = daily_sum_alpha[idx]
    daily_sum_alpha2 = daily_sum_alpha2[idx]
    p = np.percentile(daily_sum_alpha, percents, method="lower")
    mask = np.isin(daily_sum_alpha, p)
    sa = daily_sum_alpha[mask]
    sa2 = daily_sum_alpha2[mask]

    # simulate
    sims = soiling_rate(sa, sa2, model_save_file, M=M)

    return sims, sa, sa2


def daily_loss_pairs(mod, rdat, files, mirrors, cumulative=False):
    """Measured and predicted loss for `mirrors` over `files`, as two flat parallel arrays [p.p.].

    The quantity fit_quality_plots scatters against the 1:1 line: by default the change in loss
    between consecutive measurements, or the cumulative loss since the campaign's first
    measurement when `cumulative`. Split out from the plotting so a caller that needs the pooled
    range over several panels -- as save_fit_quality_figures does, to give standalone figures the
    common axes a shared-axis subplot grid used to give them for free -- can get it without
    drawing anything first.
    """
    pi, meas, r0, sf = rdat.prediction_indices, rdat.average, rdat.rho0, mod.helios.soiling_factor
    y, y_hat = [], []
    for f in files:
        mm = meas[f][:, mirrors]
        sfm = sf[f][mirrors, :]
        if cumulative:
            cumulative_loss = 100 * (r0[f][mirrors] - mm)
            cumulative_loss -= cumulative_loss[0, :]
            cumulative_loss_prediction = 100 * r0[f][mirrors][:, np.newaxis] * (1 - sfm[:, pi[f] - pi[f][0]])
            cumulative_loss_prediction -= cumulative_loss_prediction[:, 0][:, np.newaxis]
            y += [cumulative_loss.flatten()]
            y_hat += [cumulative_loss_prediction.transpose().flatten()]
        else:
            delta_rho_prediction = 100 * r0[f][mirrors] * sfm[:, pi[f] - pi[f][0]].transpose()
            y += [-100 * np.diff(mm, axis=0).flatten()]
            y_hat += [-np.diff(delta_rho_prediction, axis=0).flatten()]

    return np.concatenate(y, axis=0), np.concatenate(y_hat, axis=0)


def fit_quality_panels(train_experiments, train_mirrors, test_mirrors, test_experiments):
    """The (title, slug, experiments, mirrors) splits a fit's quality is read on.

    One definition, used both by the panelled summarize_fit_quality and by the standalone-figure
    writer, so the two can never drift apart on what a panel means.

    The held-out-mirror split exists only when mirrors were actually held out. A model whose noise
    process is orientation-dependent, and any least-squares fit, trains on every common mirror (see
    heliosoil.utilities.default_training_mirrors), so a fixed three-way split would carry an empty
    one.
    """
    train_mirrors = list(train_mirrors)
    test_mirrors = list(test_mirrors) if test_mirrors is not None else []

    panels = [("Training mirror(s)", "train_mirrors", train_experiments, train_mirrors)]
    if test_mirrors:
        panels.append(("Test mirror(s) \n (training interval)", "test_mirrors", train_experiments, test_mirrors))
    panels.append(("Test experiments\n (all tilts)", "test_experiments", test_experiments, train_mirrors + test_mirrors))
    return panels


def fit_quality_plots(
    mod,
    rdat,
    files,
    mirrors,
    ax=None,
    min_loss=None,
    max_loss=None,
    include_fits=True,
    data_ls="b.",
    data_label="Data",
    replot=True,
    vertical_adjust=0,
    cumulative=False,
):
    """
    Plots the fit quality of a model by generating three subplots:
    1. Fit quality on the training mirror(s)
    2. Fit quality on the test mirror(s) (using the training interval)
    3. Fit quality on the test experiments (using all tilts)

    The function takes in the model, reference data, training and test data, and other parameters to control the plot appearance and save the figure.

    Args:
        mod (object): The trained model to evaluate.
        rdat (object): The reference data to compare the model predictions against.
        files (list): The list of training experiment IDs.
        mirrors (list): The list of training or test mirror IDs.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure and axis will be created.
        min_loss (float, optional): The minimum loss value to use for the plot axes.
        max_loss (float, optional): The maximum loss value to use for the plot axes.
        include_fits (bool, optional): Whether to include linear fit lines in the plots.
        data_ls (str, optional): The line style for the data points.
        data_label (str, optional): The label for the data points.
        replot (bool, optional): Whether to replot the ideal 1:1 line.
        vertical_adjust (float, optional): Vertical adjustment for the annotation.
        cumulative (bool, optional): Whether to plot cumulative loss instead of daily loss.

    Returns:
        tuple: The generated figure and axis objects.
    """
    if min_loss is None:
        min_loss = 0
    if max_loss is None:
        max_loss = 0
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # No mirrors (or no experiments) to draw: every model that trains on the full mirror set
    # leaves the "held-out mirrors" panel empty, and an empty selection would otherwise reach
    # np.min/np.max of an empty array below. Label the axes and return rather than raising, so
    # the caller's panel still renders.
    if len(np.atleast_1d(mirrors)) == 0 or len(files) == 0:
        ax.annotate("no mirrors held out", xy=(0.5, 0.5), xycoords="axes fraction", ha="center", va="center", fontsize=11, color="grey")
        ax.set_box_aspect(1)
        return fig, ax

    y_flat, y_hat_flat = daily_loss_pairs(mod, rdat, files, mirrors, cumulative=cumulative)
    min_loss = float(np.min([min_loss, y_flat.min(), y_hat_flat.min()]))
    max_loss = float(np.max([max_loss, y_flat.max(), y_hat_flat.max()]))
    rmse = np.sqrt(np.mean((y_flat - y_hat_flat) ** 2))
    ax.plot(y_flat, y_hat_flat, data_ls, label=data_label + f", RMSE={rmse:.3f}")
    if include_fits:
        R = np.corrcoef(y_flat, y_hat_flat)[0, 1]
        p = np.polyfit(y_flat, y_hat_flat, deg=1)

    w = max_loss - min_loss
    min_loss -= 0.1 * w
    max_loss += 0.1 * w
    if replot:
        ax.plot([min_loss, max_loss], [min_loss, max_loss], "k-", label="Ideal")
    ax.set_ylim((min_loss, max_loss))
    ax.set_xlim((min_loss, max_loss))
    ax.set_box_aspect(1)

    if include_fits:
        linear_fit_values = p[1] + p[0] * np.array([min_loss, max_loss])
        ax.plot([min_loss, max_loss], linear_fit_values, "r:", label=data_label + "_fit")
        ax.annotate(rf"$R$={R:.2f}", xy=(0.05, 0.92 + vertical_adjust), xycoords="axes fraction")
        ax.set_ylabel(f"Predicted={p[0]:.2f}*measured + {p[1]:.2f}", fontsize=12)
        ax.legend(loc="lower right")
    else:
        ax.set_ylabel("Predicted", fontsize=12)
        ax.legend(loc="best")
    ax.set_xlabel(r"Measured", fontsize=12)
    # Only lay out a figure this call created: when drawing into a caller-supplied axes the
    # figure is theirs (and typically still has panels to fill), so tightening it here would
    # fight the caller's own layout.
    if created_fig:
        fig.tight_layout()

    return fig, ax


def summarize_fit_quality(
    model,
    ref,
    train_experiments,
    train_mirrors,
    test_mirrors,
    test_experiments,
    min_loss=None,
    max_loss=None,
    save_file=None,
    figsize=(8, 6),
    include_fits=True,
):
    """
    Summarize the fit quality of a model as predicted-vs-measured panels:
    1. Fit quality on the training mirror(s)
    2. Fit quality on the test mirror(s), training interval -- ONLY when mirrors were held out
    3. Fit quality on the test experiments (using all tilts)

    Panel 2 is omitted when `test_mirrors` is empty. A model whose noise process is
    orientation-dependent, and any least-squares fit, trains on every common mirror (see
    heliosoil.utilities.default_training_mirrors), so there are no held-out mirrors to plot and
    a fixed three-panel layout would leave a blank axes. The panel count therefore follows the
    split that actually exists.

    Args:
        model (object): The trained model to evaluate.
        ref (object): The reference data to compare the model predictions against.
        train_experiments (list): The list of training experiment IDs.
        train_mirrors (list): Column indices of the training mirrors.
        test_mirrors (list): Column indices of the held-out mirrors; may be empty.
        test_experiments (list): The list of test experiment IDs.
        min_loss (float, optional): Lower axis limit; None lets each panel derive it from data.
        max_loss (float, optional): Upper axis limit; None lets each panel derive it from data.
        save_file (str, optional): Path stem to save to ("<stem>_fit_quality.pdf"). None returns
            the figure without writing it, which is what a caller managing its own output tree
            wants.
        figsize (tuple): The figure size in inches.
        include_fits (bool): Whether to include linear fit lines in the plots.

    Returns:
        tuple: The generated figure and the array of axes (length 2 or 3).
    """
    panels = fit_quality_panels(train_experiments, train_mirrors, test_mirrors, test_experiments)
    has_held_out_mirrors = any(slug == "test_mirrors" for _title, slug, _exps, _mirrors in panels)

    fig, ax = plt.subplots(nrows=1, ncols=len(panels), sharex=True, sharey=True, figsize=figsize, squeeze=False)
    ax = ax[0]  # squeeze=False keeps a 2D array even for one column; take the single row

    for a, (title, _slug, experiments, mirrors) in zip(ax, panels):
        fit_quality_plots(model, ref, experiments, mirrors, ax=a, min_loss=min_loss, max_loss=max_loss, include_fits=include_fits)
        a.set_title(title, fontsize=14)

    if not has_held_out_mirrors:
        fig.suptitle("every common mirror was trained on -- no held-out-mirror panel", fontsize=10, color="grey", y=1.02)

    fig.tight_layout(pad=5)
    if save_file is not None:
        fig.savefig(save_file + "_fit_quality.pdf", bbox_inches="tight")

    return fig, ax


def save_fit_quality_figures(
    model,
    ref,
    train_experiments,
    train_mirrors,
    test_mirrors,
    test_experiments,
    directory,
    prefix="fit_quality",
    figsize=(5.5, 5.5),
    include_fits=True,
):
    """The same splits as summarize_fit_quality, written one standalone figure per split.

    Panelling them side by side shrinks each square scatter to a third of the page, which is
    exactly where the 1:1 line, the regression line and the annotation all have to be read
    against each other. One file per split keeps each at full size; the trade summarize_fit_quality
    made -- shared axes across panels -- is bought back explicitly by computing the pooled loss
    range over every split first and passing it to all of them, so the figures stay directly
    comparable to one another.

    Args:
        model, ref, train_experiments, train_mirrors, test_mirrors, test_experiments: as
            summarize_fit_quality.
        directory (str): directory to write into; created if it does not exist.
        prefix (str): filename stem; each file is "<prefix>_<slug>.pdf".
        figsize (tuple): size of each individual figure, in inches.
        include_fits (bool): whether to draw the linear fit line and its R annotation.

    Returns:
        list[str]: the paths written, in panel order.
    """
    panels = fit_quality_panels(train_experiments, train_mirrors, test_mirrors, test_experiments)

    # Pooled range first, across every split that has data to contribute -- a split with no
    # mirrors or no experiments draws the "nothing held out" annotation instead of a scatter and
    # has no range of its own.
    min_loss, max_loss = 0.0, 0.0
    for _title, _slug, experiments, mirrors in panels:
        if len(np.atleast_1d(mirrors)) == 0 or len(experiments) == 0:
            continue
        y, y_hat = daily_loss_pairs(model, ref, experiments, mirrors)
        min_loss = float(np.min([min_loss, y.min(), y_hat.min()]))
        max_loss = float(np.max([max_loss, y.max(), y_hat.max()]))

    os.makedirs(directory, exist_ok=True)
    paths = []
    for title, slug, experiments, mirrors in panels:
        # A split with nothing in it is not written at all. In a subplot grid an empty panel at
        # least holds its place in a row; as a file of its own it is a figure saying only that it
        # is empty. fit_quality_panels already drops the held-out-mirror split when no mirror was
        # held out, so this catches the remaining case: a fit trained on every campaign, which
        # leaves no test experiments.
        if len(np.atleast_1d(mirrors)) == 0 or len(experiments) == 0:
            continue
        fig, a = plt.subplots(figsize=figsize)
        fit_quality_plots(model, ref, experiments, mirrors, ax=a, min_loss=min_loss, max_loss=max_loss, include_fits=include_fits)
        # The panel titles carry a newline for the narrow subplot columns; a standalone figure has
        # the width for one line.
        a.set_title(" ".join(title.split()), fontsize=13)
        fig.tight_layout()
        path = f"{directory}/{prefix}_{slug}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def daily_rate_residuals(model, reflectance_data, experiments, mirrors=None):
    """
    Predicted minus measured **daily soiling rate**, one value per (mirror, measurement interval),
    pooled over `experiments`.

    This is the quantity regression_performance_stats reduces to MBE/MAE/RMSE; it is exposed
    separately so a caller can look at the error's *distribution* (e.g. grouped by tilt) rather
    than only its first moments. See that function's docstring for why the rate, not the
    reflectance, is the error quantity, and for the nominal-reflectance convention.

    The rate is ``-diff(soiling_factor) / diff(time_days)``, with the measured soiling factor
    taken as ``measured_reflectance / nominal_reflectance`` so both sides are in the same
    dimensionless (soiling-factor per day) units. Differencing is done per experiment so it never
    crosses a campaign boundary. Requires model.helios.soiling_factor to already be populated for
    `experiments`.

    Args:
        model: a fitted soiling model exposing helios.soiling_factor and
            helios.nominal_reflectance.
        reflectance_data: ReflectanceMeasurements with matching prediction_indices, average, and
            times for each file in `experiments`.
        experiments (list): file indices to pool together.
        mirrors (array-like of int, optional): column indices to include. Default: all mirrors.

    Returns:
        np.ndarray: the finite residuals, flattened. Empty when nothing survives (e.g. a campaign
        with a single measurement, or an all-NaN mirror) -- callers must handle size 0 rather
        than assume a mean is defined.
    """
    pi = reflectance_data.prediction_indices
    meas = reflectance_data.average
    sf = model.helios.soiling_factor
    times = reflectance_data.times

    resid_list = []
    for f in experiments:
        m = meas[f] if mirrors is None else meas[f][:, mirrors]
        sfm = sf[f] if mirrors is None else sf[f][mirrors, :]
        r0 = _nominal_reflectance_series(reflectance_data, f, model.helios.nominal_reflectance)
        if mirrors is not None and not np.isscalar(r0):
            r0 = r0[:, mirrors]

        sf_pred = sfm[:, pi[f]].transpose()  # predicted soiling factor, (n_time, n_mirror)
        # Measured soiling factor = measured reflectance / nominal (clean) reflectance, so both
        # sides of the rate residual are in the same dimensionless soiling-factor units.
        sf_meas = m / r0

        # Elapsed days between consecutive measurements (same convention as loss_table_from_sim).
        t = np.asarray(times[f])
        if np.issubdtype(t.dtype, np.datetime64):
            dt_days = np.diff(t) / np.timedelta64(1, "D")
        else:
            dt_days = np.diff(t.astype(float))
        dt_days = dt_days[:, None]

        # Daily soiling rate = drop in soiling factor per day (a drop is positive soiling).
        rate_pred = -np.diff(sf_pred, axis=0) / dt_days
        rate_meas = -np.diff(sf_meas, axis=0) / dt_days
        resid_list.append((rate_pred - rate_meas).flatten())

    # Mask non-finite entries (NaN measurements, zero-length intervals) so a few gaps don't void
    # the whole set.
    resid = np.concatenate(resid_list) if resid_list else np.array([])
    return resid[np.isfinite(resid)]


def regression_performance_stats(model, reflectance_data, experiments, mirrors=None):
    """
    Summarize a model fit with regression metrics computed on two different quantities so the
    error metrics judge soiling *dynamics* while R2 stays a familiar reflectance goodness-of-fit:

    - MBE, MAE, RMSE are computed on the **daily soiling rate**: the drop in soiling factor
      between consecutive measurements divided by the elapsed days,
      ``rate = -diff(soiling_factor) / diff(time_days)``. The measured soiling factor is
      ``measured_reflectance / nominal_reflectance``, so predicted and measured rates are in the
      same dimensionless (soiling-factor per day) units. This measures how well each prediction
      predicts the *change* in soiling loss over each interval rather than how well it fits the
      overall reflectance shape/level. Differencing is done per experiment so it never crosses a
      campaign boundary. The residuals themselves are available from `daily_rate_residuals`, which
      computes them; this function only reduces them to their first moments.
    - R2 is the coefficient of determination on **reflectance** (predicted vs. measured), pooled
      over `experiments`. Called with the training vs. test experiments it yields the in-sample /
      out-of-sample reflectance R2 respectively.

    Requires model.helios.soiling_factor to already be populated for `experiments`
    (e.g. by calling model.predict_soiling_factor(...) or model.plot_soiling_factor(...)
    beforehand) with tilt/azimuth set consistently with `reflectance_data`.

    Predicted reflectance is nominal_reflectance * soiling_factor, not
    reflectance_data.rho0 * soiling_factor: soiling_factor's own initial condition is
    already anchored to each mirror's rho0 via compute_soiling_factor (cumulative_soil0
    = (1 - rho0/nominal_reflectance)/inc_factor), so nominal_reflectance * soiling_factor
    exactly recovers rho0 at t=0 for every mirror; multiplying by rho0 again would not.
    `nominal_reflectance` here is reflectance_data's reference-mirror-derived,
    per-mirror/time value when available (heliosoil.utilities._nominal_reflectance_series),
    else model.helios.nominal_reflectance. Dividing the measurement by this same time-varying
    nominal removes reference-mirror drift from the measured rate, matching the drift-agnostic
    model soiling factor.

    Args:
        model: a fitted soiling model exposing helios.soiling_factor and
            helios.nominal_reflectance.
        reflectance_data: ReflectanceMeasurements with matching prediction_indices, average, and
            times for each file in `experiments`.
        experiments (list): file indices (as used to key simulation_inputs/reflectance_data)
            to pool together when computing the statistics.
        mirrors (array-like of int, optional): column indices to include. Default: all mirrors.

    Returns:
        dict with keys "MBE", "MAE", "RMSE" (daily soiling rate, soiling-factor per day),
        "R2" (reflectance), and "N" (number of pooled reflectance observations). An empty
        `experiments` yields NaN statistics over N=0 rather than an error: a split can
        legitimately be empty (a model trained on every campaign has no test set), and that
        is an absent statistic, not a failure.
    """
    if len(experiments) == 0:
        return {"MBE": np.nan, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "N": 0}

    pi = reflectance_data.prediction_indices
    meas = reflectance_data.average
    sf = model.helios.soiling_factor

    # Reflectance-level pairing (predicted vs. measured) -> R2 only.
    pred_list, meas_list = [], []
    for f in experiments:
        m = meas[f] if mirrors is None else meas[f][:, mirrors]
        sfm = sf[f] if mirrors is None else sf[f][mirrors, :]
        r0 = _nominal_reflectance_series(reflectance_data, f, model.helios.nominal_reflectance)
        if mirrors is not None and not np.isscalar(r0):
            r0 = r0[:, mirrors]

        sf_pred = sfm[:, pi[f]].transpose()  # predicted soiling factor, (n_time, n_mirror)
        pred_list.append((r0 * sf_pred).flatten())  # predicted reflectance, for R2
        meas_list.append(m.flatten())  # measured reflectance, for R2

    # Daily soiling-rate error metrics, over the already-finite-masked residuals. If nothing
    # survives (e.g. a campaign with a single measurement), report NaN rather than warn on an
    # empty mean.
    rate_resid = daily_rate_residuals(model, reflectance_data, experiments, mirrors=mirrors)
    if rate_resid.size:
        mbe, mae, rmse = np.mean(rate_resid), np.mean(np.abs(rate_resid)), np.sqrt(np.mean(rate_resid**2))
    else:
        mbe = mae = rmse = np.nan

    # Reflectance coefficient of determination (in-sample or out-of-sample per `experiments`).
    pred_flat = np.concatenate(pred_list)
    meas_flat = np.concatenate(meas_list)
    resid = pred_flat - meas_flat
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((meas_flat - np.mean(meas_flat)) ** 2)

    return {"MBE": mbe, "MAE": mae, "RMSE": rmse, "R2": 1 - ss_res / ss_tot, "N": meas_flat.size}


def daily_soiling_tilt_all_data(
    sim_dat: smb.SimulationInputs, model_save_file: str, M: int = 1000, dust_type="TSP", tilt: float = None, trim_percents=None, azimuth: float = 0.0
):
    """Monte-Carlo daily loss for a hypothetical mirror at a fixed `tilt`.

    Delegates to virtual_mirror_daily_loss, so the loss comes from the fitted model's own forward
    equations rather than a second reduction of the same physics. Two consequences for callers
    that predate that: the days are the simulation window's (as trimmed when the inputs were
    built) rather than every row of the raw "Weather" sheet, which moves the numbers slightly; and
    a wind-driven model is now supported, taking `azimuth` into account instead of being rejected.

    The daily alpha sums are still returned, unchanged in meaning, for callers that plot them --
    but they no longer drive the simulation.
    """
    df = [pd.read_excel(f, "Weather") for f in sim_dat.files]
    df = pd.concat(df)
    df.sort_values(by="Time", inplace=True)

    prototype_pm = getattr(sim_dat.dust, dust_type)[0]
    df["alpha"] = df[dust_type] / prototype_pm
    df["date"] = df["Time"].dt.date
    df["alpha2"] = df["alpha"] ** 2
    daily_sum_alpha = (df.groupby("date")["alpha"].sum()).values
    daily_sum_alpha2 = (df.groupby("date")["alpha2"].sum()).values

    if trim_percents is not None:
        xl, xu = np.percentile(daily_sum_alpha, trim_percents, axis=None)
        mask = (daily_sum_alpha >= xl) & (daily_sum_alpha <= xu)
        daily_sum_alpha = daily_sum_alpha[mask]
        daily_sum_alpha2 = daily_sum_alpha2[mask]

    if tilt is not None:
        daily_sum_alpha = daily_sum_alpha * gravitational_settling_factor(tilt)
        daily_sum_alpha2 = daily_sum_alpha2 * gravitational_settling_factor(tilt) ** 2

    with open(model_save_file, "rb") as f:
        model = pickle.load(f)["model"]
    sims, _expected = virtual_mirror_daily_loss(
        model, sim_dat, model_save_file, [(0.0 if tilt is None else float(tilt), float(azimuth))], M=M, trim_percents=trim_percents
    )
    return sims[:, 0, :], daily_sum_alpha, daily_sum_alpha2


def supports_loss_curve(model) -> bool:
    """Whether `model` can back a loss-vs-tilt curve (see plot_loss_vs_tilt).

    Two requirements, for two different reasons.

    The curve evaluates the model forward on virtual mirrors at tilts/azimuths that were never
    measured, so the model's delta_soiled_area must be a function of (tilt, azimuth, dust, wind)
    alone. That is every constant-mean model -- plain or wind-driven -- but not the physical
    ones, whose flux comes from a per-mirror particle-size distribution (helios.pdfqN) built by
    a separate deposition-velocity step that a synthetic mirror has never been through.

    It also samples the deposition noise, so the fit must have left a noise process behind. A
    least-squares fit estimates the mean parameters only and sets every sigma to None, which
    means there is no interval to draw and no `sigma` to sample."""
    if not isinstance(model, smb.ConstantMeanBase):
        return False
    sigma_names = getattr(model, "_sigma_param_names", ("sigma_dep",))
    return any(getattr(model, name, None) is not None for name in sigma_names)


def _virtual_mirror_daily_bases(model, sim_dat, grid):
    """Daily-summed per-parameter bases for a grid of virtual (tilt, azimuth) mirrors.

    delta_soiled_area is affine in the mean parameters and its variance is linear in the squared
    sigmas, so the whole Monte Carlo can be driven from bases evaluated ONCE rather than by
    re-running the forward model per draw. The bases are read off the forward model by probing it
    at unit parameter vectors -- the same trick, and for the same reason, as
    ConstantMeanWindDeposition.mean_design_matrix: no second copy of the physics to drift out of
    sync, and per-component dust channels and the clipped settling factor come along for free.

    Returns (mean_bases, var_bases, days): bases have shape (n_parameters, n_virtual, n_days).
    """
    mean_names = list(getattr(model, "_mean_param_names", ("mu_tilde",)))
    sigma_names = list(getattr(model, "_sigma_param_names", ("sigma_dep",)))
    files = list(sim_dat.time.keys())

    # A shallow copy with its own helios: the caller's fitted model keeps its real mirrors and
    # its own predictions, which it may still be being used for.
    work = copy.copy(model)
    work.helios = copy.copy(model.helios)
    work.helios.delta_soiled_area, work.helios.delta_soiled_area_variance = {}, {}
    work.helios.tilt, work.helios.azimuth = {}, {}
    for f in files:
        n_times = len(sim_dat.time[f])
        work.helios.tilt[f] = np.array([[t] * n_times for t, _ in grid], dtype=float)
        work.helios.azimuth[f] = np.array([[a] * n_times for _, a in grid], dtype=float)

    day_index = {f: pd.Series(np.asarray(sim_dat.time[f])).dt.date.values for f in files}

    def daily_sum(per_file):
        """Concatenate each file's per-day totals along the day axis."""
        return np.concatenate([pd.DataFrame(per_file[f].T).groupby(day_index[f]).sum().to_numpy().T for f in files], axis=1)

    def probe(assignments, attribute):
        for name, value in assignments.items():
            setattr(work, name, value)
        work.calculate_delta_soiled_area(sim_dat, verbose=False)
        return daily_sum(getattr(work.helios, attribute))

    mean_bases, var_bases = [], []
    # Means: one at 1, the rest at 0, every sigma off so no (discarded) variance is built.
    for name in mean_names:
        mean_bases.append(probe({**dict.fromkeys(mean_names, 0.0), name: 1.0, **dict.fromkeys(sigma_names, None)}, "delta_soiled_area"))
    # Sigmas: one at 1, the rest off. The variance basis does not depend on the means.
    for name in sigma_names:
        var_bases.append(probe({**dict.fromkeys(mean_names, 0.0), **dict.fromkeys(sigma_names, None), name: 1.0}, "delta_soiled_area_variance"))

    days = np.concatenate([np.unique(day_index[f]) for f in files])
    return np.array(mean_bases), np.array(var_bases), days


def sampling_covariance(param_cov, max_sd=10.0):
    """A covariance safe to draw fitted parameters from, and the indices it had to hold fixed.

    An MLE covariance comes from inverting a numerical Hessian, and for a parameter the data does
    not identify that inversion is free to return nonsense: a NEGATIVE variance, or one so large
    that a draw on the log scale overflows exp() to inf and poisons every downstream number with
    NaN. Both occur in practice on wind models whose noise terms fit to ~1e-10.

    Two repairs, in order. The matrix is symmetrised and its negative eigenvalues clipped to zero
    (the nearest positive-semi-definite matrix), then any parameter still carrying a non-finite or
    absurd standard deviation is pinned: its row and column are zeroed so it is held at its point
    estimate. Pinning is the honest reading -- an unidentified parameter has no usable uncertainty
    to propagate, and holding it fixed states that, where sampling it would invent a spread the
    likelihood never supported.

    Args:
        param_cov: the fitted parameters' covariance, on whatever scale they are expressed in.
        max_sd: standard deviations above this are treated as unidentified. The default is well
            past any real uncertainty on a log-scale parameter (e^10 is a 22000x spread) and well
            below where exp() overflows.

    Returns:
        (cov, pinned): the repaired covariance and the indices held fixed.
    """
    cov = np.array(param_cov, dtype=float)
    cov = 0.5 * (cov + cov.T)  # symmetrise: an inverted Hessian is only symmetric up to roundoff
    cov[~np.isfinite(cov)] = 0.0

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if (eigenvalues < 0).any():
        cov = (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
        cov = 0.5 * (cov + cov.T)

    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    pinned = np.flatnonzero(~np.isfinite(sd) | (sd > max_sd))
    for i in pinned:
        cov[i, :] = 0.0
        cov[:, i] = 0.0

    if pinned.size:
        logger.warning(
            f"Parameter(s) at index {pinned.tolist()} are not identified by this fit (non-finite or "
            f"implausible standard error); holding them at their estimate instead of sampling them."
        )
    return cov, pinned


def virtual_mirror_daily_loss(model, sim_dat, model_save_file, grid, M=1000, trim_percents=None, rng=None):
    """Monte-Carlo daily reflectance loss for virtual mirrors at each (tilt, azimuth) in `grid`.

    Each draw samples the fitted parameters from their covariance and then the deposition noise
    around the mean they imply, so the spread carries both parameter uncertainty and the model's
    own noise process -- the same two sources `soiling_rate` combines, but evaluated through the
    model's forward equations instead of a daily-dust reduction. That is what lets a wind-driven
    model be curved at all: its geometry factors carry the wind speed and direction of each
    timestep, so they cannot be pulled outside a daily sum the way cos(tilt) can.

    Args:
        model: a fitted constant-mean (plain or wind) model; see supports_loss_curve.
        sim_dat: SimulationInputs carrying the weather the loss is evaluated over.
        model_save_file: path stem of the pickled fit, read for the fitted parameters and their
            covariance on the model's own transformed scale.
        grid: sequence of (tilt_deg, azimuth_deg) virtual mirrors.
        M: Monte-Carlo draws.
        trim_percents: optional [low, high] percentile range of daily totals to keep, applied per
            virtual mirror, so a curve dominated by a few dust events can be told from a typical
            day.
        rng: optional np.random.Generator, for reproducibility.

    Returns:
        (sims, expected): two quantities that answer two different questions and should not be
        substituted for one another.

        `sims`, shape (M, n_virtual, n_days_kept), is the sampled daily loss in percentage points
        -- parameter draw plus deposition noise. Its spread answers "how much does one day differ
        from another?", which is several times the mean and nearly the same for every orientation.

        `expected`, shape (M, n_virtual), is each draw's mean daily loss with the zero-mean noise
        term left out. Its spread is the uncertainty in the expected loss -- parameter uncertainty
        alone -- which is what belongs in a band beside a mean curve. Dropping the noise also
        means two virtual mirrors that are genuinely identical (every orientation at tilt 0, say)
        land on exactly the same line rather than differing by the Monte Carlo's own error.
    """
    rng = np.random.default_rng() if rng is None else rng
    with open(model_save_file, "rb") as f:
        saved = pickle.load(f)
    log_param_hat = np.asarray(saved["transformed_parameters"], dtype=float)
    # Repaired before sampling: an unidentified parameter's Hessian-derived variance can come back
    # negative or astronomically large, either of which turns a draw into inf/NaN (see
    # sampling_covariance). Those parameters are held at their estimate instead.
    log_param_cov, _pinned = sampling_covariance(np.asarray(saved["transformed_parameter_covariance"], dtype=float))

    mean_bases, var_bases, _days = _virtual_mirror_daily_bases(model, sim_dat, grid)
    n_mean = mean_bases.shape[0]

    if trim_percents is not None:
        # Trim on the mean-parameter basis total, i.e. on how much soiling that day drives, so
        # the kept days are the typical ones for this mirror rather than for some other measure.
        total = mean_bases.sum(axis=0)  # (n_virtual, n_days)
        keep = np.ones(total.shape, dtype=bool)
        for i in range(total.shape[0]):
            lo, hi = np.percentile(total[i], trim_percents)
            keep[i] = (total[i] >= lo) & (total[i] <= hi)
        n_keep = int(keep.sum(axis=1).min())
    else:
        keep = n_keep = None

    inc_factor = float(np.atleast_1d(model.helios.inc_ref_factor[next(iter(sim_dat.time.keys()))]).ravel()[0])

    sims = np.empty((M, mean_bases.shape[1], n_keep if n_keep is not None else mean_bases.shape[2]))
    expected = np.empty((M, sims.shape[1]))
    for m in range(M):
        params = np.atleast_1d(model.transform_scale(rng.multivariate_normal(log_param_hat, log_param_cov)))
        # Backstop: a repaired covariance makes this vanishingly unlikely, but one non-finite
        # parameter would otherwise turn the whole curve into NaN silently. Fall back to the point
        # estimate for that entry rather than propagating it.
        params = np.where(np.isfinite(params), params, np.atleast_1d(model.transform_scale(log_param_hat)))
        mean_daily = np.tensordot(params[:n_mean], mean_bases, axes=1)
        var_daily = np.tensordot(np.asarray(params[n_mean:], dtype=float) ** 2, var_bases, axes=1)
        sample = rng.normal(loc=mean_daily, scale=np.sqrt(np.clip(var_daily, 0.0, None)))

        def select(x):
            return x if keep is None else np.array([x[i, keep[i]][:n_keep] for i in range(x.shape[0])])

        sims[m] = select(inc_factor * sample * 100.0)
        expected[m] = select(inc_factor * mean_daily * 100.0).mean(axis=1)

    return sims, expected


CARDINAL_AZIMUTHS = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}

# Categorical hues for the loss curve's orientations, assigned in this fixed order and never
# cycled. Deliberately NOT _orientation_colors: that map is keyed by each site's own measured
# mirror codes (yadnarie's are intercardinal -- NE/SE/SW/NW), so cardinal labels fall through it
# and every curve comes out the same colour. Validated as a 4-slot categorical palette on the
# all-pairs list (worst CVD dE 9.2, worst normal-vision dE 16.3), since all four curves share one
# axes and are directly compared.
_LOSS_CURVE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")


def _is_azimuth_invariant(model, sim_dat) -> bool:
    """Whether this model's loss depends on mirror azimuth at all.

    Probed rather than inferred from the component list: two virtual mirrors at the same tilt and
    different azimuths either produce the same bases or they do not, and asking the model directly
    cannot fall out of step with which mechanisms happen to be active. A plain constant-mean model
    (and a wind model built only from gravitational/turbulent terms) is invariant, and drawing it
    as four identical curves would be four ways of saying one thing."""
    bases, _var, _days = _virtual_mirror_daily_bases(model, sim_dat, [(45.0, 0.0), (45.0, 90.0)])
    return bool(np.allclose(bases[:, 0, :], bases[:, 1, :], rtol=1e-12, atol=0.0))


def plot_loss_vs_tilt(
    model,
    sim_dat: smb.SimulationInputs,
    model_save_file: str,
    tilts=(0, 10, 30, 45, 60, 75, 90),
    azimuths=None,
    M: int = 1000,
    percentiles=(5, 95),
    ax=None,
    figsize=(7, 4.5),
    rng=None,
):
    """Predicted daily reflectance loss as a function of (fixed) mirror tilt and orientation.

    One curve per orientation. For a tilt-only model every orientation gives the same answer, so
    the four collapse to a single curve rather than four identical ones; for a wind-driven model
    they separate, and that separation is the orientation contrast the wind mechanisms exist to
    capture. The curves re-converge at tilt 0 and past vertical, where no mechanism can tell one
    azimuth from another.

    Two spreads are drawn, and they are not the same quantity (see virtual_mirror_daily_loss):

      - the shaded band is PARAMETER uncertainty -- the percentile range of each draw's expected
        daily loss. It is pooled across orientations rather than drawn per curve because it is
        common-mode: a draw that raises mu_tilde lifts every orientation together, so the curves'
        spacing is far better determined than their shared level;
      - the dashed envelope is the DAY-TO-DAY range of the sampled loss. It crosses zero, and that
        is a real outcome rather than an artefact: the deposition noise is symmetric, so on a
        low-dust day the sampled change is a small apparent gain in reflectance.

    Those two, with the mean curve, are what the figure this replaces carried (an "all data" mean
    with its 5-95% day-to-day band), plus the parameter uncertainty it did not separate out. What
    that figure additionally drew, and this one does not, is a second "trimmed" curve over the
    central 5-95% of dust days: because the mean is taken over days either way, trimming the tails
    moves it by under 1% at every tilt -- two orders of magnitude inside the parameter band -- so
    the second curve was a restatement of the first. (Trimming does visibly narrow the day-to-day
    range, but that is a statement about the range, which is drawn from the untrimmed record
    precisely because the worst days are the point of it.) virtual_mirror_daily_loss still takes
    trim_percents for a caller that wants the trimmed quantity directly.

    Loss enters through max(0, cos(tilt)) (see heliosoil.utilities.gravitational_settling_factor),
    so a mirror at or past vertical settles nothing; for a tilt-only model the curve, its band and
    its day-to-day envelope all reach exactly zero at 90 deg, since a surface that deposits
    nothing also carries no deposition noise.

    Args:
        model: the fitted model to evaluate; see supports_loss_curve for what qualifies.
        sim_dat: SimulationInputs carrying the weather to evaluate over.
        model_save_file: path stem of the pickled fit (as written by model.save), which must
            hold `transformed_parameters` and `transformed_parameter_covariance`.
        tilts: tilt angles [deg] to evaluate.
        azimuths: {label: azimuth_deg} to draw; defaults to the four cardinal directions.
        M: Monte-Carlo draws.
        percentiles: band edges drawn around each curve.
        ax: axes to draw on; a new figure is created when None.
        rng: optional np.random.Generator, for reproducibility.

    Returns:
        (fig, ax, table): `table` carries one row per (orientation, tilt) with the expected daily
        loss, its parameter-uncertainty band and the day-to-day range, in percentage points per day.
    """
    azimuths = dict(CARDINAL_AZIMUTHS) if azimuths is None else dict(azimuths)
    # A model with no azimuth term would draw every orientation as the same curve. Collapse to the
    # one curve it actually has rather than stacking four identical lines under a legend that
    # implies they could differ.
    azimuth_invariant = _is_azimuth_invariant(model, sim_dat)
    if azimuth_invariant:
        azimuths = {"all orientations": next(iter(azimuths.values()))}
    grid = [(float(t), float(a)) for a in azimuths.values() for t in tilts]

    sims_all, expected_all = virtual_mirror_daily_loss(model, sim_dat, model_save_file, grid, M=M, rng=rng)

    # The band comes from `expected` -- each draw's mean daily loss without the deposition noise --
    # so it is the uncertainty in the expected loss. Banding the raw samples instead would plot
    # the day-to-day spread, which runs several times the curve's own height, is nearly identical
    # for every orientation, and drawn as a filled region buries the comparison the figure exists
    # for. That range is drawn as an unfilled envelope instead, and reported per row in the CSV.
    expected_all = np.asarray(expected_all)  # (M, n_virtual)
    sims_all = np.asarray(sims_all)

    rows = []
    for i, (tilt, azimuth) in enumerate(grid):
        label = next(name for name, value in azimuths.items() if value == azimuth)
        lo_all, hi_all = np.percentile(expected_all[:, i], percentiles)
        day_lo, day_hi = np.percentile(sims_all[:, i, :], percentiles, axis=None)
        rows.append(
            {
                "orientation": label,
                "azimuth": azimuth,
                "tilt": tilt,
                # The noise-free expectation, not the sample mean: see virtual_mirror_daily_loss.
                "mean_pp_per_day": float(expected_all[:, i].mean()),
                "low_pp_per_day": float(lo_all),
                "high_pp_per_day": float(hi_all),
                # The day-to-day range: the answer to a different question ("how bad can one day
                # get?"), which is why it is an outline rather than a second fill.
                "daily_low_pp_per_day": float(day_lo),
                "daily_high_pp_per_day": float(day_hi),
            }
        )
    table = pd.DataFrame(rows)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ONE band, not one per orientation. The parameter uncertainty is common-mode: a draw that
    # raises mu_tilde lifts every orientation together, so the curves' spacing is far better
    # determined than their common level. Drawing a band per orientation would stack four
    # near-identical wide regions into a grey mass and hide the very separation being compared,
    # while implying each orientation's level is independently that uncertain. Pooled across
    # orientations at each tilt, and drawn behind everything, it reads as what it is: how well the
    # level is pinned down, with the curves carrying the comparison.
    sorted_tilts = sorted(float(t) for t in tilts)

    def envelope(low_column, high_column):
        """Pooled low/high across orientations at each tilt, in tilt order."""
        pooled = table.groupby("tilt")[[low_column, high_column]].agg({low_column: "min", high_column: "max"}).reindex(sorted_tilts)
        return np.asarray(pooled[low_column], float), np.asarray(pooled[high_column], float)

    band_low, band_high = envelope("low_pp_per_day", "high_pp_per_day")
    ax.fill_between(sorted_tilts, band_low, band_high, color="#9aa0a6", alpha=0.18, linewidth=0, zorder=0)

    # The day-to-day range, as an unfilled envelope. It spans several times the curve's own height
    # and crosses zero -- the deposition noise is symmetric, so a low-dust day can sample as a
    # small apparent GAIN in reflectance, which is a real outcome of the fitted noise process and
    # not an artefact to hide. Drawn as thin dashed outlines rather than a fill so it frames the
    # curves instead of burying them, and pooled across orientations because it is very nearly the
    # same for all of them.
    day_low, day_high = envelope("daily_low_pp_per_day", "daily_high_pp_per_day")
    for edge in (day_low, day_high):
        ax.plot(sorted_tilts, edge, color="#9aa0a6", linestyle="--", linewidth=1.0, zorder=1)

    # Colour is the orientation, and the only encoding the curves carry -- hues taken in fixed
    # order, never cycled.
    for slot, label in enumerate(azimuths):
        part = table[table["orientation"] == label].sort_values("tilt")
        col = {name: np.asarray(part[name], dtype=float) for name in table.columns if name != "orientation"}
        color = "#2a2a28" if azimuth_invariant else _LOSS_CURVE_COLORS[slot % len(_LOSS_CURVE_COLORS)]

        ax.plot(col["tilt"], col["mean_pp_per_day"], color=color, linewidth=2, label=label if not azimuth_invariant else None, zorder=3)

        # Deliberately no direct labels. They are the usual way to keep identity off colour alone,
        # but there is nowhere on this figure to put them: the orientations converge exactly at
        # tilt 0 (no mechanism separates them on a horizontal mirror) and are still within a few
        # tenths of a point of each other at maximum tilt, so labels collide at either end. The
        # legend carries identity instead, and the loss_vs_tilt.csv written beside the figure is
        # the table view that the palette's one sub-3:1 hue obliges.

    # Line-style key plus the band, which nothing else identifies. Orientation identity is carried
    # by the direct labels, so it is not repeated here when there is more than one curve.
    style_handles = [
        Line2D([0], [0], color="#2a2a28", linewidth=2),
        Patch(facecolor="#9aa0a6", alpha=0.18, linewidth=0),
        Line2D([0], [0], color="#9aa0a6", linestyle="--", linewidth=1.0),
    ]
    style_labels = [
        "expected loss (all days)",
        f"parameter uncertainty ({percentiles[0]}-{percentiles[1]}%)",
        f"day-to-day range ({percentiles[0]}-{percentiles[1]}%)",
    ]
    if not azimuth_invariant:
        handles, labels = ax.get_legend_handles_labels()
        style_handles, style_labels = handles + style_handles, labels + style_labels
    ax.legend(style_handles, style_labels, fontsize=8, frameon=False, ncol=2 if not azimuth_invariant else 1)

    ax.set_xlabel("Tilt (degrees)")
    ax.set_ylabel("Expected loss \n (p.p./day)")
    ax.set_xlim((min(tilts), max(tilts)))
    # No floor at zero. A negative daily rate is a real outcome, not an artefact: the deposition
    # noise is symmetric, so on a low-dust day the sampled change is a small apparent GAIN, and a
    # wind model's omegas are fitted on a linear scale and may themselves draw negative. Flooring
    # the axis would silently hide both.
    ax.axhline(0.0, color="#c8ccd0", linewidth=0.8, zorder=0)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if created_fig:
        fig.tight_layout()

    return fig, ax, table


# Ordinal ramp for the dust-day scenarios. One hue, light to dark, because Low -> Maximum is
# ordered magnitude rather than four separate identities; the light end clears 2:1 against the
# surface so the palest scenario does not recede into it. Validated as a 4-step ordinal ramp.
_SCENARIO_COLORS = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")


def point_estimate_daily_loss(model, sim_dat, model_save_file, grid):
    """Mean and standard deviation of the daily loss at the FITTED parameters, without sampling.

    The analytic normal the fit implies for each day, against which a sampled distribution can be
    compared: the sampled one additionally carries parameter uncertainty, so it is a mixture over
    draws rather than this single normal, and the gap between them is how much not knowing the
    parameters widens the answer beyond the noise process alone.

    Returns:
        (mean, sd): arrays of shape (n_virtual, n_days), in percentage points per day.
    """
    with open(model_save_file, "rb") as f:
        saved = pickle.load(f)
    params = np.atleast_1d(model.transform_scale(np.asarray(saved["transformed_parameters"], dtype=float)))

    mean_bases, var_bases, _days = _virtual_mirror_daily_bases(model, sim_dat, grid)
    n_mean = mean_bases.shape[0]
    inc_factor = float(np.atleast_1d(model.helios.inc_ref_factor[next(iter(sim_dat.time.keys()))]).ravel()[0])

    mean = inc_factor * np.tensordot(params[:n_mean], mean_bases, axes=1) * 100.0
    variance = np.tensordot(np.asarray(params[n_mean:], dtype=float) ** 2, var_bases, axes=1)
    sd = inc_factor * np.sqrt(np.clip(variance, 0.0, None)) * 100.0
    return mean, sd


def plot_loss_distributions(
    model,
    sim_dat: smb.SimulationInputs,
    model_save_file: str,
    percents=(5, 50, 95, 100),
    labels=("Low", "Medium", "High", "Maximum"),
    tilt: float = 0.0,
    azimuth: float = 0.0,
    M: int = 20000,
    bins: int = 200,
    ax=None,
    figsize=(6.5, 4.5),
    rng=None,
):
    """Distribution of the daily reflectance loss on a low, median, high and worst dust day.

    Each scenario is one real day from the record -- the day sitting at that percentile of the
    fit's own expected daily loss -- and its histogram is that day's loss over `M` draws of the
    parameters and the deposition noise.

    Overlaid on each is the POINT ESTIMATE: the normal N(mu, sigma^2) obtained by holding every
    parameter at its fitted (maximum-likelihood) value -- the single number the fit reports, no
    covariance drawn from -- and evaluating that day's mean and deposition variance from it. It is
    therefore the deposition noise alone. The histogram is the same day with the parameters also
    sampled from their covariance, so it is a mixture of such normals rather than one, and is the
    wider of the two by however much not knowing the parameters costs. The gap between them
    separates "this day is variable" from "we do not know the parameters well"; where they
    coincide, the fit's uncertainty is negligible next to the day-to-day noise.

    Drawn for a horizontal mirror by default, matching the figure this replaces. At tilt 0 the
    answer is azimuth-free for every model in the package (no mechanism separates orientations on
    a horizontal surface), so the default needs no orientation choice to be well posed.

    Args:
        model: the fitted model to evaluate; see supports_loss_curve for what qualifies.
        sim_dat: SimulationInputs carrying the weather to evaluate over.
        model_save_file: path stem of the pickled fit (as written by model.save).
        percents: percentiles of the daily record picking each scenario's day.
        labels: scenario names, in the same order as `percents`.
        tilt, azimuth: the virtual mirror's geometry [deg].
        M: Monte-Carlo draws.
        bins: histogram bins per scenario.
        ax: axes to draw on; a new figure is created when None.
        rng: optional np.random.Generator, for reproducibility.

    Returns:
        (fig, ax, table): one row per scenario, all in p.p./day -- the day's percentile, then the
        sampled mean, sd and 5-95% range (parameters sampled + noise), then the same three for the
        point estimate (parameters at their fitted values, noise only).
    """
    if len(percents) != len(labels):
        raise ValueError(f"percents and labels must be the same length, got {len(percents)} and {len(labels)}.")

    grid = [(float(tilt), float(azimuth))]
    sims, _expected = virtual_mirror_daily_loss(model, sim_dat, model_save_file, grid, M=M, rng=rng)
    point_mean, point_sd = point_estimate_daily_loss(model, sim_dat, model_save_file, grid)
    day_loss, day_sims = point_mean[0], np.asarray(sims)[:, 0, :]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    rows, selected_days = [], []
    for slot, (percent, label) in enumerate(zip(percents, labels)):
        # The day AT that percentile, not an interpolated one: the scenario has to be a real day so
        # its dust, wind and duration are internally consistent.
        target = np.percentile(day_loss, percent, method="lower")
        day = int(np.argmin(np.abs(day_loss - target)))
        selected_days.append(day)
        samples = day_sims[:, day]
        color = _SCENARIO_COLORS[slot % len(_SCENARIO_COLORS)]

        ax.hist(samples, bins=bins, density=True, alpha=0.45, color=color, label=label, linewidth=0)
        # The point-estimate normal for the same day, drawn over its own support.
        mu, sigma = float(point_mean[0, day]), float(point_sd[0, day])
        if sigma > 0:
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
            ax.plot(x, np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)), color=color, linewidth=1.5)

        low, high = np.percentile(samples, [5, 95])
        # Both intervals are 5-95%, so the columns sit side by side honestly: 1.645 sigma, not the
        # more familiar 1.96, which would be a 2.5-97.5% interval and would read as the
        # point estimate being the wider of the two purely by construction. The sd columns sit
        # beside them because mean +/- sd is the form these numbers get quoted in, and reading it
        # off a percentile range silently assumes the distribution is normal -- which the sampled
        # one is not: it is a mixture over parameter draws.
        rows.append(
            {
                "scenario": label,
                "percentile": percent,
                "sampled_mean_pp_per_day": float(samples.mean()),
                "sampled_sd_pp_per_day": float(samples.std(ddof=1)),
                "sampled_low_pp_per_day": float(low),
                "sampled_high_pp_per_day": float(high),
                "point_estimate_mean_pp_per_day": mu,
                "point_estimate_sd_pp_per_day": sigma,
                "point_estimate_low_pp_per_day": mu - 1.645 * sigma,
                "point_estimate_high_pp_per_day": mu + 1.645 * sigma,
            }
        )

    # Zero is a meaningful place on this axis: mass to its left is days the fitted noise process
    # says can read as a small apparent GAIN in reflectance.
    ax.axvline(0.0, color="#c8ccd0", linewidth=0.8, zorder=0)

    # Frame on where the samples actually are. The overlaid normals are drawn out to +-4 sigma so
    # their tails meet the axis cleanly, which on the widest scenario reaches well past any real
    # data and would otherwise leave a third of the panel empty.
    low_edge, high_edge = np.percentile(day_sims[:, selected_days], [0.2, 99.8])
    margin = 0.04 * (high_edge - low_edge)
    ax.set_xlim(min(low_edge - margin, 0.0), high_edge + margin)
    # Two keys: the scenario colours above, then what each MARK means. Both style entries lead
    # with the mark ("histogram:", "curve:") so the neutral swatch cannot be mistaken for a fifth
    # scenario, and the curve's key names what it holds fixed rather than merely that it is a
    # normal -- the difference between it and the histogram beside it is why both are drawn.
    handles, legend_labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#2a2a28", alpha=0.45, linewidth=0))
    legend_labels.append("histogram: parameters sampled + noise")
    handles.append(Line2D([0], [0], color="#2a2a28", linewidth=1.5))
    legend_labels.append("curve: point estimate (fitted parameters, noise only)")
    ax.legend(handles, legend_labels, fontsize=8, frameon=False)

    ax.set_xlabel("Loss (percentage points per day)")
    ax.set_ylabel("Probability density")
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if created_fig:
        fig.tight_layout()

    return fig, ax, pd.DataFrame(rows)


def plot_experiment_PA(simulation_inputs, reflectance_data, experiment_index, figsize=(7, 12)):
    sim_data = simulation_inputs
    reflect_data = reflectance_data
    f = experiment_index

    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=figsize)
    # fmt = r"${0:s}^\circ$"
    fmt = "${0:s}$"
    ave = reflect_data.average[f]
    t = reflect_data.times[f]
    std = reflect_data.sigma[f]
    # names = ["M"+str(ii+1) for ii in range(ave.shape[1])]
    names = ["SE1", "SE2", "SE3", "SE4", "SE5", "NW1", "NW2", "NW3", "NW4", "NW5"]

    for ii in range(ave.shape[1]):
        ax[0].errorbar(t, ave[:, ii], yerr=1.96 * std[:, ii], label=fmt.format(names[ii]), marker="o", capsize=4.0)

    ax[0].grid(True)
    label_str = r"Reflectance at {0:.1f} $^{{\circ}}$".format(reflect_data.reflectometer_incidence_angle[f])
    ax[0].set_ylabel(label_str)
    ax[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax[0].set_ylim(0.85, 0.97)

    ax[1].plot(sim_data.time[f], sim_data.dust_conc_mov_avg[f], color="brown", label="Measurements")
    ax[1].axhline(y=sim_data.dust_concentration[f].mean(), color="brown", ls="--", label="Average")
    label_str = r"{0:s} [$\mu g\,/\,m^3$]".format(sim_data.dust_type[0])
    ax[1].set_ylabel(label_str, color="brown")
    ax[1].tick_params(axis="y", labelcolor="brown")
    ax[1].grid(True)
    ax[1].legend()
    title_str = sim_data.dust_type[f] + r" (mean = {0:.2f} $\mu g$/$m^3$)"
    ax[1].set_title(title_str.format(sim_data.dust_concentration[f].mean()), fontsize=15)
    ax[1].set_ylim(0, 50)

    # # Rain intensity, if available
    # if len(sim_data.rain_intensity)>0: # rain intensity is not an empty dict
    #     ax[2].plot(sim_data.time[f],sim_data.rain_intensity[f])
    # else:
    #     rain_nan = np.nan*np.ones(sim_data.time[f].shape)
    #     ax[2].plot(sim_data.time[f],rain_nan)

    # ax[2].set_ylabel(r'Rain [mm/hour]',color='blue')
    # ax[2].tick_params(axis='y', labelcolor='blue')
    # YL = ax[2].get_ylim()
    # ax[2].set_ylim((0,YL[1]))
    # ax[2].grid(True)

    ax[2].plot(sim_data.time[f], sim_data.wind_speed_mov_avg[f], color="green", label="Measurements")
    ax[2].axhline(y=sim_data.wind_speed[f].mean(), color="green", ls="--", label="Average")
    label_str = r"Wind Speed [$m\,/\,s$]"
    ax[2].set_ylabel(label_str, color="green")
    ax[2].set_xlabel("Date")
    ax[2].tick_params(axis="y", labelcolor="green")
    ax[2].grid(True)
    ax[2].legend()
    title_str = "Wind Speed (mean = {0:.2f} m/s)".format(sim_data.wind_speed[f].mean())
    ax[2].set_title(title_str, fontsize=15)
    ax[2].set_ylim(0, 9)

    # if len(sim_data.relative_humidity)>0:
    #     ax[4].plot(sim_data.time[f],sim_data.relative_humidity[f],color='black',label="measurements")
    #     ax[4].axhline(y=sim_data.relative_humidity[f].mean(),color='black',ls='--',label = "Average")
    # else:
    #     rain_nan = np.nan*np.ones(sim_data.time[f].shape)
    #     ax[4].plot(sim_data.time[f],rain_nan)

    # label_str = r'Relative Humidity [%]'
    # ax[4].set_ylabel(label_str,color='black')
    # ax[4].set_xlabel('Date')
    # ax[4].tick_params(axis='y', labelcolor='black')
    # ax[4].grid(True)
    # ax[4].legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    return fig, ax
