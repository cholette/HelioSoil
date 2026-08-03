
# %%
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
main_directory = os.path.abspath(os.path.join(script_dir, ".."))
for root, dirs, files in os.walk(main_directory):
    sys.path.append(root)
# %%

import numpy as np
import pandas as pd
import soiling_model.base_models as smb
import soiling_model.utilities as smu
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as sps
import re

# %%
# CHOOSE RAYGEN LOCATION (Carwarp or Yadnarie)
# raygen_site = "Carwarp"
raygen_site = "Yadnarie"

# CHOOSE WHETHER TO WORK ON HELIOSTATS OR ON THE MIRROR RIG - it works only for Carwarp data
hel_analysis = False
if raygen_site == "Carwarp":
    HELIOSTATS = hel_analysis
else:
    HELIOSTATS = False

# default to False, set to True to save outputs (and override any previous saved output)
save_output = False

# %% Set data input

site_paths = {
    "Carwarp": {
        "save_file": f"{main_directory}/results/wind_results_mildura",
        "d": f"{main_directory}/data/mildura/",
        "compass": 1,
        "all_intervals": np.array([
            [np.datetime64('2024-01-30T10:00:00'), np.datetime64('2024-02-05T08:00:00')],
            [np.datetime64('2024-06-06T17:00:00'), np.datetime64('2024-06-11T08:00:00')]],
            dtype='datetime64[m]')
    },
    "Yadnarie": {
        "save_file": f"{main_directory}/results/wind_results_yadnarie",
        "d": f"{main_directory}/data/yadnarie/",
        "compass": 2,
        "all_intervals": np.array([
            [np.datetime64('2024-11-11T20:30:00'), np.datetime64('2024-11-16T07:30:00')],
            [np.datetime64('2024-11-12T11:40:00'), np.datetime64('2024-11-16T07:30:00')]],
            dtype='datetime64[m]')
    } }

if raygen_site in site_paths:
    save_file = site_paths[raygen_site]['save_file']
    d = site_paths[raygen_site]['d']
    all_intervals = site_paths[raygen_site]['all_intervals']
    print('Analysis of ' + raygen_site + ' data')
else:
    raise ValueError(f"Invalid site '{raygen_site}'. Choose a valid site for analysis.")

reflectometer_incidence_angle = 15 # [deg]
reflectometer_acceptance_angle = 12.5e-3 # [rad] half acceptance angle

time_to_remove_at_end = [0, 0]  # hours to remove at end of each experiment
k_factor = "import"


# %% Get file list and time intervals

files, all_intervals, exp_mirrors, all_mirrors = smu.get_training_data(d, "_experiment_", time_to_remove_at_end=time_to_remove_at_end)

# %% Load simulation (met) data

dust_type = "PM2.5"
sim_data = smb.simulation_inputs(files, k_factors=k_factor, dust_type=dust_type)

# %% Load reflectance data

n_meas = 6.0
reflect_data = smb.reflectance_measurements(
    files,
    sim_data.time,
    number_of_measurements=n_meas,
    reflectometer_incidence_angle=reflectometer_incidence_angle,
    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
    import_tilts=True,
    column_names_to_import=None,
)

# %% Trim to experiment windows

sim_data, reflect_data = smu.trim_experiment_data(sim_data, reflect_data, all_intervals)

# %% Merge SE mirrors into the November 2024 experiment

se_file = d + "experiment_20241112_20241116_SE.xlsx"

nov_idx = next(f for f in sim_data.time.keys()
               if sim_data.start_datetime[f].month == 11 and sim_data.start_datetime[f].year == 2024)

reflect_se = smb.reflectance_measurements(
    [se_file],
    {0: sim_data.time[nov_idx]},
    number_of_measurements=n_meas,
    reflectometer_incidence_angle=reflectometer_incidence_angle,
    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
    import_tilts=True,
    column_names_to_import=None,
)

# align SE measurement times onto Nov times, NaN-pad the missing first time point
nov_times = reflect_data.times[nov_idx]
se_times = reflect_se.times[0]
se_in_nov = np.array([np.argmin(np.abs(nov_times - t)) for t in se_times])

n_nov = len(nov_times)
n_se = reflect_se.average[0].shape[1]
avg_pad = np.full((n_nov, n_se), np.nan)
sig_pad = np.full((n_nov, n_se), np.nan)
avg_pad[se_in_nov] = reflect_se.average[0]
sig_pad[se_in_nov] = reflect_se.sigma[0]

reflect_data.average[nov_idx]      = np.hstack([reflect_data.average[nov_idx], avg_pad])
reflect_data.sigma[nov_idx]        = np.hstack([reflect_data.sigma[nov_idx],   sig_pad])
reflect_data.mirror_names[nov_idx] = reflect_data.mirror_names[nov_idx] + reflect_se.mirror_names[0]
n_nov_times = reflect_data.tilts[nov_idx].shape[1]
se_tilts_broadcast = np.repeat(reflect_se.tilts[0][:, 0:1], n_nov_times, axis=1)
reflect_data.tilts[nov_idx]        = np.vstack([reflect_data.tilts[nov_idx], se_tilts_broadcast])

# %% Plot reflectance and wind speed per experiment

for f in sim_data.time.keys():
    mirror_names = reflect_data.mirror_names[f]
    labels = [re.sub(r"_T\d{2,3}", "", m.replace("O", "").replace("_M", "")) for m in mirror_names]

    # subset of vertical mirrors (tilt == 90 deg) — reflect_data is kept intact
    vertical_idx = [ii for ii in range(len(mirror_names))
                    if np.isclose(np.nanmean(reflect_data.tilts[f][ii]), 90)]
    if not vertical_idx:
        print(f"Experiment {f}: no vertical mirrors found, skipping.")
        continue
    labels_vert = [labels[ii] for ii in vertical_idx]
    avg_vert = reflect_data.average[f][:, vertical_idx]
    sigma_vert = reflect_data.sigma[f][:, vertical_idx]

    ref_times = pd.to_datetime(reflect_data.times[f].astype("datetime64[ns]"))
    sim_times = pd.to_datetime(sim_data.time[f])

    fig, axes = plt.subplots(nrows=3, figsize=(12, 11), sharex=True, constrained_layout=True)

    # --- top panel: reflectance for vertical mirrors only ---
    ax_ref = axes[0]
    for ii, lbl in enumerate(labels_vert):
        if "AV" in lbl:
            continue
        ax_ref.errorbar(
            ref_times,
            avg_vert[:, ii],
            yerr=1.96 * sigma_vert[:, ii],
            marker="o",
            capsize=4,
            label=lbl,
        )
    ax_ref.set_ylabel("Measured Reflectance")
    ax_ref.legend(fontsize=9, loc="lower left")
    ax_ref.grid(True)

    month_str = ref_times[0].strftime("%B %Y")
    ax_ref.set_title(f"{raygen_site} – Reflectance, Wind & Dust/RH – {month_str}")

    # --- middle panel: wind speed (line) + direction (scatter colour) ---
    ax_wind = axes[1]
    ws = sim_data.wind_speed[f]
    wd = getattr(sim_data, "wind_direction", {}).get(f, None)

    if wd is not None and not np.all(np.isnan(wd)):
        sc = ax_wind.scatter(sim_times, ws, c=wd, cmap="hsv", vmin=0, vmax=360,
                             s=4, alpha=0.7, zorder=3)
        cbar = fig.colorbar(sc, ax=ax_wind, pad=0.01)
        cbar.set_label("Wind Direction [°]", fontsize=9)
    else:
        ax_wind.plot(sim_times, ws, color="green", lw=0.8)

    ax_wind.axhline(np.nanmean(ws), color="green", ls="--", lw=1,
                    label=f"Mean = {np.nanmean(ws):.2f} m/s")
    ax_wind.set_ylabel("Wind Speed [m/s]", color="green")
    ax_wind.tick_params(axis="y", labelcolor="green")
    ax_wind.legend(fontsize=9)
    ax_wind.grid(True)

    # --- bottom panel: PM2.5 and PM10 ---
    ax_dust = axes[2]
    pm25 = getattr(sim_data, "pm2p5", {}).get(f, None)
    pm10 = getattr(sim_data, "pm10", {}).get(f, None)

    if pm25 is not None:
        ax_dust.plot(sim_times, pm25, color="brown", lw=0.8, label=f"PM2.5 (mean={np.nanmean(pm25):.1f} µg/m³)")
    if pm10 is not None:
        ax_dust.plot(sim_times, pm10, color="orange", lw=0.8, label=f"PM10 (mean={np.nanmean(pm10):.1f} µg/m³)")

    ax_dust.set_ylabel(r"Dust concentration [$\mu$g/m³]")
    ax_dust.grid(True)

    ax_rh = ax_dust.twinx()
    rh = getattr(sim_data, "relative_humidity", {}).get(f, None)

    if rh is not None:
        ax_rh.plot(sim_times, rh, color="blue", lw=0.8,
                   label=f"RH (mean={np.nanmean(rh):.1f} %)")
        ax_rh.axhline(np.nanmean(rh), color="blue", ls="--", lw=1)

    ax_rh.set_ylabel("Relative Humidity [%]", color="blue")
    ax_rh.tick_params(axis="y", labelcolor="blue")
    ax_rh.set_ylim(0, 100)

    lines_dust, labels_dust = ax_dust.get_legend_handles_labels()
    lines_rh, labels_rh = ax_rh.get_legend_handles_labels()
    ax_dust.legend(lines_dust + lines_rh, labels_dust + labels_rh, fontsize=9)

    ax_dust.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax_dust.tick_params(axis="x", rotation=15)

    plt.show()

# %% Correlation: reflectance loss vs. wind speed and cos(wind direction)

for f in sim_data.time.keys():
    mirror_names_f = reflect_data.mirror_names[f]
    selected_idx = [ii for ii in range(len(mirror_names_f))
                    if np.nanmean(reflect_data.tilts[f][ii]) in (60, 90)]
    if not selected_idx:
        continue

    ref_times = pd.to_datetime(reflect_data.times[f].astype("datetime64[ns]"))
    sim_times = pd.to_datetime(sim_data.time[f])
    ws = sim_data.wind_speed[f]
    wd = getattr(sim_data, "wind_direction", {}).get(f, None)
    avg_sel = reflect_data.average[f][:, selected_idx]
    exp_label = ref_times[0].strftime("%b %Y")

    losses, ws_vals, cos_wd_vals, pm10_vals, mirror_labels = [], [], [], [], []
    mirror_names_sel = [mirror_names_f[ii] for ii in selected_idx]
    pm10 = getattr(sim_data, "pm10", {}).get(f, None)

    for i in range(len(ref_times) - 1):
        mask = (sim_times >= ref_times[i]) & (sim_times < ref_times[i + 1])
        if mask.sum() == 0:
            continue
        mean_ws   = np.nanmean(ws[mask])
        cos_wd    = np.nanmean(np.cos(np.radians(wd[mask]))) if wd is not None else np.nan
        mean_pm10 = np.nanmean(pm10[mask]) if pm10 is not None else np.nan

        dt_days = (ref_times[i + 1] - ref_times[i]).total_seconds() / 86400

        for m, mname in enumerate(mirror_names_sel):
            loss = avg_sel[i, m] - avg_sel[i + 1, m]
            if np.isnan(loss):
                continue
            losses.append(loss * 100 / dt_days)
            ws_vals.append(mean_ws)
            cos_wd_vals.append(cos_wd)
            pm10_vals.append(mean_pm10)
            mirror_labels.append(re.sub(r"_T\d{2,3}", "", mname.replace("O", "").replace("_M", "")))

    losses        = np.array(losses)
    ws_vals       = np.array(ws_vals)
    cos_wd_vals   = np.array(cos_wd_vals)
    pm10_vals     = np.array(pm10_vals)
    mirror_labels = np.array(mirror_labels)

    pm10_ws    = pm10_vals * ws_vals
    pm10_coswd = pm10_vals * cos_wd_vals

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, mlbl in enumerate(np.unique(mirror_labels)):
        mask = mirror_labels == mlbl
        axes[0].scatter(pm10_ws[mask],    losses[mask], label=mlbl, alpha=0.8, color=colors[idx])
        axes[1].scatter(pm10_coswd[mask], losses[mask], label=mlbl, alpha=0.8, color=colors[idx])

    for ax, x, xlabel in [
        (axes[0], pm10_ws,    r"Mean PM10 $\times$ Wind Speed [$\mu$g/m³ · m/s]"),
        (axes[1], pm10_coswd, r"Mean PM10 $\times$ cos(Wind Direction) [$\mu$g/m³]"),
    ]:
        valid = ~np.isnan(x) & ~np.isnan(losses)
        if valid.sum() < 2:
            continue
        r, p = sps.pearsonr(x[valid], losses[valid])
        slope, intercept = np.polyfit(x[valid], losses[valid], 1)
        xfit = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(xfit, slope * xfit + intercept, "k--", lw=1.2, label=f"r = {r:.2f}, p = {p:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Reflectance Loss [p.p./day]")
        ax.legend(fontsize=9)
        ax.grid(True)

    plt.suptitle(f"{raygen_site} – {exp_label} – Vertical mirrors: reflectance loss correlations", fontsize=13)
    plt.tight_layout()
    plt.show()
# %%
