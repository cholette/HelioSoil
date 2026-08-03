# %%
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
main_directory = os.path.abspath(os.path.join(script_dir, ".."))
for root, dirs, files in os.walk(main_directory):
    sys.path.append(root)

import numpy as np
import pandas as pd
import soiling_model.base_models as smb
import soiling_model.utilities as smu
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import scipy.stats as sps
import re
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

# %% functions

def cardinal_to_uv(cardinal: str) -> tuple[float, float]:
    """
    Convert a cardinal direction string to (U, V) unit vector components.
    U: positive = eastward, V: positive = northward.
    Direction is where the wind is GOING (not coming from).
    """
    angles = {
        'N':     0, 'NNE':  22.5, 'NE':   45, 'ENE':  67.5,
        'E':    90, 'ESE': 112.5, 'SE':  135, 'SSE': 157.5,
        'S':   180, 'SSW': 202.5, 'SW':  225, 'WSW': 247.5,
        'W':   270, 'WNW': 292.5, 'NW':  315, 'NNW': 337.5,
    }
    key = cardinal.strip().upper()
    if key not in angles:
        raise ValueError(f"Unknown cardinal direction: '{cardinal}'")

    theta = np.deg2rad(angles[key])
    return np.cos(theta), np.sin(theta)

def parse_mirror_name(name):
    """Extract orientation and tilt from e.g. 'ON_M1_T90' -> ('N', 90)"""
    parts = name.split('_')
    orientation = parts[0][1:]      # 'ON' -> 'N'
    tilt = int(parts[2][1:])        # 'T90' -> 90
    return orientation, tilt

def campaign_title(filepath):
    """Extract month name from filename e.g. 'MountIsa_Data_20200901_...' -> 'September 2020'"""
    basename = os.path.basename(filepath)
    match = re.search(r'_(\d{8})_', basename)
    if match:
        date = datetime.strptime(match.group(1), '%Y%m%d')
        return date.strftime('%B %Y')
    return basename

def print_correlation(label, x, y, mirror):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        print(f"  {mirror:<15} | {label:<45} | not enough data")
        return
    r_p, p_p = pearsonr(x[mask],  y[mask])
    r_s, p_s = spearmanr(x[mask], y[mask])
    print(f"  {mirror:<15} | {label:<45} | Pearson r={r_p:+.3f} (p={p_p:.3f})  Spearman r={r_s:+.3f} (p={p_s:.3f})")

# %% Analysis of wind — Mount Isa

tilts_of_interest = [90, 85, 60, 30, 5, 0]
time_to_remove_at_end = [0, 0, 0, 0, 0, 0]
k_factor = "import"
dust_type = "TSP"

d = os.path.join(main_directory, "data", "mount_isa") + "\\"
save_file = os.path.join(main_directory, "results", "wind_results_mount_isa")
reflectometer_incidence_angle = 15       # [deg]
reflectometer_acceptance_angle = 12.5e-3 # [rad] half acceptance angle

files, all_intervals, exp_mirrors, all_mirrors = smu.get_training_data(d, "MountIsa_Data_", time_to_remove_at_end=time_to_remove_at_end)
sim_data = smb.simulation_inputs(files, k_factors=k_factor, dust_type=dust_type)
wind_mask = {ii: sim_data.wind_speed[ii] > 0.001 for ii in range(len(sim_data.file_name))}

# %%
n_meas = 9.0
reflect_data = smb.reflectance_measurements(
    files,
    sim_data.time,
    number_of_measurements=n_meas,
    reflectometer_incidence_angle=reflectometer_incidence_angle,
    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
    import_tilts=True,
    column_names_to_import=None,
)

# %% Get soiling rates between measurements
soiling_rate = {}
for ii, f in enumerate(reflect_data.file_name):
    dr = np.diff(reflect_data.average[ii], axis=0)
    dt = np.diff(reflect_data.times[ii]).astype('timedelta64[s]').astype(float) / 3600 / 24
    soiling_rate[ii] = -dr / dt[:, np.newaxis]

# %% Mirror normals (positive N, E)
mirror_names = []
for ii, _ in enumerate(reflect_data.file_name):
    mirror_names.extend(reflect_data.mirror_names[ii])
mirror_names = np.unique(mirror_names)

mirror_i = []
mirror_j = []
mirror_keys = []
for m in mirror_names:
    orientation, tilt = parse_mirror_name(m)
    u, v = cardinal_to_uv(orientation)
    mirror_i.append(u)
    mirror_j.append(v)
    mirror_keys.append((orientation, tilt))

mirrors_to_exclude = set()  # add mirror prefixes like 'ON_M1' to exclude specific mirrors
mirrors_of_interest = [(key, m) for key, m in zip(mirror_keys, mirror_names)
                       if key[1] in tilts_of_interest
                       and '_'.join(m.split('_')[:2]) not in mirrors_to_exclude]

print("Mirrors of interest:")
for key, m in mirrors_of_interest:
    print(f"  {m} -> {key}")

# %% Wind direction
wind_nu = {}
wind_nv = {}
wind_u  = {}
wind_v  = {}
for ii, _ in enumerate(sim_data.file_name):
    wind_nu[ii] = -np.sin(np.deg2rad(sim_data.wind_direction[ii]))
    wind_nv[ii] = -np.cos(np.deg2rad(sim_data.wind_direction[ii]))
    wind_u[ii]  = sim_data.wind_speed[ii] * wind_nu[ii]
    wind_v[ii]  = sim_data.wind_speed[ii] * wind_nv[ii]

# %% Projection of wind vector onto normal of mirror
n_mirrors = len(mirror_names)
s_proj  = {}
sn_proj = {}
for ii, _ in enumerate(sim_data.file_name):
    s_proj[ii]  = {}
    sn_proj[ii] = {}
    for jj, key in enumerate(mirror_keys):
        sn_proj[ii][key] = wind_nu[ii] * mirror_i[jj] + wind_nv[ii] * mirror_j[jj]
        s_proj[ii][key]  = wind_u[ii]  * mirror_i[jj] + wind_v[ii]  * mirror_j[jj]

# %% Plot sn_proj for each mirror
for ii, f in enumerate(files):
    fig, axes = plt.subplots(len(mirrors_of_interest), 1,
                             figsize=(10, 3*len(mirrors_of_interest)),
                             sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (key, m) in zip(axes, mirrors_of_interest):
        ax.plot(sim_data.time[ii][wind_mask[ii]], sn_proj[ii][key][wind_mask[ii]], linewidth=0.6)
        ax.set_ylabel(m, fontsize=9)
        ax.grid(True)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=25, ha='right')
    fig.suptitle(f"sn_proj — {campaign_title(f)}")
    plt.tight_layout()
    plt.show()

# %% Plot s_proj for each mirror
for ii, f in enumerate(files):
    fig, axes = plt.subplots(len(mirrors_of_interest), 1,
                             figsize=(10, 3*len(mirrors_of_interest)),
                             sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (key, m) in zip(axes, mirrors_of_interest):
        ax.plot(sim_data.time[ii][wind_mask[ii]], s_proj[ii][key][wind_mask[ii]], linewidth=0.6)
        ax.set_ylabel(m, fontsize=9)
        ax.grid(True)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=25, ha='right')
    fig.suptitle(f"s_proj — {campaign_title(f)}")
    plt.tight_layout()
    plt.show()

# %% Plots
for ii, f in enumerate(files):
    fig, ax = smu.wind_rose(sim_data, ii, wind_mask[ii])
    ax.set_title(campaign_title(f))
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.plot(sim_data.time[ii], sim_data.wind_direction[ii])
    ax1.set_title(campaign_title(f))
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Wind direction (°)")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=25, ha='right')
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(sim_data.time[ii], sim_data.wind_speed[ii])
    ax2.set_title(campaign_title(f))
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Wind speed (m/s)")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=25, ha='right')
    plt.tight_layout()
    plt.show()

# %% Aggregate wind and dust over each soiling interval
wind_speed_mean  = {}
wind_dir_mean    = {}
tsp_mean         = {}
rh_mean          = {}
crosses_midnight = {}
s_proj_mean      = {}
sn_proj_mean     = {}

for ii, _ in enumerate(sim_data.file_name):
    n_intervals = len(reflect_data.times[ii]) - 1
    wind_speed_mean[ii]  = np.zeros(n_intervals)
    wind_dir_mean[ii]    = np.zeros(n_intervals)
    tsp_mean[ii]         = np.full(n_intervals, np.nan)
    rh_mean[ii]          = np.full(n_intervals, np.nan)
    crosses_midnight[ii] = np.array([
        reflect_data.times[ii][kk].astype('datetime64[D]') !=
        reflect_data.times[ii][kk+1].astype('datetime64[D]')
        for kk in range(n_intervals)
    ])
    s_proj_mean[ii]      = {key: np.zeros(n_intervals) for key in mirror_keys}
    sn_proj_mean[ii]     = {key: np.zeros(n_intervals) for key in mirror_keys}

    for kk in range(n_intervals):
        t_start = reflect_data.times[ii][kk]
        t_end   = reflect_data.times[ii][kk+1]
        time_mask = (sim_data.time[ii] >= t_start) & (sim_data.time[ii] < t_end)

        mask = time_mask & wind_mask[ii]

        wind_speed_mean[ii][kk] = np.nanmean(sim_data.wind_speed[ii][mask])
        tsp_mean[ii][kk]        = np.nanmean(sim_data.dust_concentration[ii][mask])
        if len(sim_data.relative_humidity) > 0:
            rh_mean[ii][kk]     = np.nanmean(sim_data.relative_humidity[ii][mask])

        wd_rad = np.deg2rad(sim_data.wind_direction[ii][mask])
        wind_dir_mean[ii][kk] = np.rad2deg(np.arctan2(
            np.nanmean(np.sin(wd_rad)),
            np.nanmean(np.cos(wd_rad))
        )) % 360

        for key in mirror_keys:
            s_proj_mean[ii][key][kk]  = np.nanmean(s_proj[ii][key][mask])
            sn_proj_mean[ii][key][kk] = np.nanmean(sn_proj[ii][key][mask])

# %% Plot wind projection per tilt
for tilt in tilts_of_interest:
    mirrors_tilt = [(key, m) for key, m in mirrors_of_interest if key[1] == tilt]
    n_mirrors_tilt = len(mirrors_tilt)
    n_campaigns = len(files)
    if n_mirrors_tilt == 0:
        continue

    fig, axes = plt.subplots(n_mirrors_tilt, n_campaigns,
                             figsize=(6*n_campaigns, 3*n_mirrors_tilt),
                             sharey='row', sharex='col')
    axes = np.atleast_2d(axes)

    for row, (key, m) in enumerate(mirrors_tilt):
        for ii, f in enumerate(files):
            ax = axes[row, ii]
            if row == 0:
                ax.set_title(campaign_title(f), fontsize=12)
            if ii == 0:
                ax.set_ylabel(f"{m}\nWind proj. (m/s)", fontsize=9)

            ax.plot(sim_data.time[ii], s_proj[ii][key],
                    color='steelblue', linewidth=0.6, alpha=0.8)
            ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha='right')

    fig.suptitle(f"Wind projection onto mirror normal — {tilt}° tilt mirrors", fontsize=13)
    plt.tight_layout()
    plt.show()

# %% Correlations

for ii, f in enumerate(files):
    print(f"\n{'='*100}")
    print(f"Campaign: {campaign_title(f)}")
    print(f"{'='*100}")

    ws  = wind_speed_mean[ii]
    wd  = wind_dir_mean[ii]
    tsp = tsp_mean[ii]
    rh  = rh_mean[ii]

    for jj, (key, m) in enumerate(zip(mirror_keys, mirror_names)):
        if m not in reflect_data.mirror_names[ii]:
            continue

        jj_local = list(reflect_data.mirror_names[ii]).index(m)
        sr    = soiling_rate[ii][:, jj_local]
        proj  = s_proj_mean[ii][key]
        nproj = sn_proj_mean[ii][key]

        print(f"\n  --- {m} ---")
        print_correlation("Wind speed",                          ws,       sr, m)
        print_correlation("Wind direction",                      wd,       sr, m)
        print_correlation("Wind projection",                     proj,     sr, m)
        print_correlation("Wind projection (norm)",              nproj,    sr, m)
        print_correlation("TSP",                                 tsp,      sr, m)
        print_correlation("Wind speed × TSP",                    ws*tsp,   sr, m)
        print_correlation("Wind projection × TSP",               proj*tsp, sr, m)
        print_correlation("Wind projection (norm) × TSP",        nproj*tsp,sr, m)
        print_correlation("RH",                                  rh,       sr, m)
        print_correlation("Wind speed × RH",                     ws*rh,    sr, m)
        print_correlation("Wind projection × RH",                proj*rh,  sr, m)
        print_correlation("Wind projection (norm) × RH",         nproj*rh, sr, m)
        print_correlation("TSP × RH",                            tsp*rh,   sr, m)

# %% Compute all correlations for mirrors of interest and rank them
all_correlations = {
    "Wind speed":                    lambda ii, key: wind_speed_mean[ii],
    "Wind direction":                lambda ii, key: wind_dir_mean[ii],
    "Wind projection":               lambda ii, key: s_proj_mean[ii][key],
    "Wind proj (norm)":              lambda ii, key: sn_proj_mean[ii][key],
    "TSP":                           lambda ii, _: tsp_mean[ii],
    "Wind speed × TSP":              lambda ii, _: wind_speed_mean[ii]   * tsp_mean[ii],
    "Wind proj × TSP":               lambda ii, key: s_proj_mean[ii][key] * tsp_mean[ii],
    "Wind proj (norm) × TSP":        lambda ii, key: sn_proj_mean[ii][key]* tsp_mean[ii],
    "RH":                            lambda ii, _: rh_mean[ii],
    "Wind speed × RH":               lambda ii, _: wind_speed_mean[ii]   * rh_mean[ii],
    "Wind proj × RH":                lambda ii, key: s_proj_mean[ii][key] * rh_mean[ii],
    "Wind proj (norm) × RH":         lambda ii, key: sn_proj_mean[ii][key]* rh_mean[ii],
    "TSP × RH":                      lambda ii, _: tsp_mean[ii] * rh_mean[ii],
}

def top_correlations(mirrors_tilt, n_top=4):
    """
    For a given set of mirrors, compute mean |Pearson r| across all campaigns
    and return the top n_top correlations.
    """
    corr_scores = {label: [] for label in all_correlations}

    for key, m in mirrors_tilt:
        for ii, f in enumerate(files):
            if m not in reflect_data.mirror_names[ii]:
                continue
            jj_local = list(reflect_data.mirror_names[ii]).index(m)
            sr = soiling_rate[ii][:, jj_local]

            for label, x_func in all_correlations.items():
                x = x_func(ii, key)
                mask = np.isfinite(x) & np.isfinite(sr)
                if mask.sum() >= 3:
                    r, _ = pearsonr(x[mask], sr[mask])
                    corr_scores[label].append(abs(r))

    mean_scores = {
        label: np.mean(vals) if vals else 0.0
        for label, vals in corr_scores.items()
    }

    ranked = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Correlation ranking (mean |Pearson r|):")
    for label, score in ranked:
        print(f"    {label:<35} {score:.3f}")

    return [(label, all_correlations[label]) for label, _ in ranked[:n_top]]

# %% Plot top 4 correlations for each tilt
for tilt in tilts_of_interest:
    mirrors_tilt = [(key, m) for key, m in mirrors_of_interest if key[1] == tilt]
    n_mirrors_tilt = len(mirrors_tilt)
    if n_mirrors_tilt == 0:
        continue

    print(f"\n{'='*60}")
    print(f"Tilt {tilt}°")
    top_corr = top_correlations(mirrors_tilt, n_top=4)

    fig, axes = plt.subplots(n_mirrors_tilt, 4,
                             figsize=(20, 4*n_mirrors_tilt),
                             sharey='row')
    axes = np.atleast_2d(axes)

    for row, (key, m) in enumerate(mirrors_tilt):
        for col, (label, x_func) in enumerate(top_corr):
            ax = axes[row, col]

            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{m}\nSoiling rate (%/day)", fontsize=9)

            for ii, f in enumerate(files):
                if m not in reflect_data.mirror_names[ii]:
                    continue
                jj_local = list(reflect_data.mirror_names[ii]).index(m)
                sr = soiling_rate[ii][:, jj_local]
                x  = x_func(ii, key)

                ax.scatter(x, sr, label=campaign_title(f), s=40, zorder=3)

                mask = np.isfinite(x) & np.isfinite(sr)
                if mask.sum() >= 2:
                    z = np.polyfit(x[mask], sr[mask], 1)
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
                    ax.plot(x_line, np.polyval(z, x_line), linewidth=1.2)

                if mask.sum() >= 3:
                    r, p = pearsonr(x[mask], sr[mask])
                    ax.annotate(f"r={r:+.2f}\np={p:.2f}",
                                xy=(0.05, 0.95), xycoords='axes fraction',
                                fontsize=8, va='top')

            ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
            ax.axvline(0, color='k', linewidth=0.6, linestyle='--')
            ax.set_xlabel(label, fontsize=8)

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', ncol=len(files),
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Top 4 correlations — {tilt}° tilt mirrors", fontsize=13)
    plt.tight_layout()
    plt.show()

# %% Plot reflectance for assessed mirrors
for ii, f in enumerate(files):
    lgd_size = 10
    fig, ax = plt.subplots(nrows=2, figsize=(12, 8), gridspec_kw={'hspace': 0.3})

    ave = reflect_data.average[ii]
    t   = reflect_data.times[ii]
    std = reflect_data.sigma[ii]

    lgd_label = [re.sub(r"_T\d{2,3}", "", lg.replace("O", "").replace("_M", ""))
                 for lg in reflect_data.mirror_names[ii]]

    for jj in range(ave.shape[1]):
        m = reflect_data.mirror_names[ii][jj]
        _, tilt = parse_mirror_name(m)
        if tilt not in tilts_of_interest:
            continue
        ax[0].errorbar(t, ave[:, jj], yerr=1.96*std[:, jj],
                       label=lgd_label[jj], marker='o', capsize=4.0)

    ax[0].set_ylabel("Measured Reflectance")
    ax[0].legend(fontsize=lgd_size, loc='center right', bbox_to_anchor=(1.15, 0.5))
    ax[0].grid(True)

    t_wind = sim_data.time[ii][wind_mask[ii]]
    ws     = sim_data.wind_speed[ii][wind_mask[ii]]

    ax[1].plot(t_wind, ws, color='green', label="Measurements")
    ax[1].axhline(y=ws.mean(), color='green', ls='--',
                  label=r"Average = {0:.2f}".format(ws.mean()))
    ax[1].set_ylabel(r'Wind Speed [$m\,/\,s$]', color='green')
    ax[1].tick_params(axis='y', labelcolor='green')
    ax[1].grid(True)
    ax[1].legend(fontsize=lgd_size)

    [axis.tick_params(axis='x', rotation=15) for axis in ax]

    fig.suptitle(f"Reflectance & Wind — {campaign_title(f)}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

# %% Plot reflectance and wind speed per experiment
for f in sim_data.time.keys():
    mirror_names_f = reflect_data.mirror_names[f]
    labels = [re.sub(r"_T\d{2,3}", "", m.replace("O", "").replace("_M", "")) for m in mirror_names_f]

    vertical_idx = [ii for ii in range(len(mirror_names_f))
                    if np.isclose(np.nanmean(reflect_data.tilts[f][ii]), 90)]
    if not vertical_idx:
        print(f"Experiment {f}: no vertical mirrors found, skipping.")
        continue
    labels_vert = [labels[ii] for ii in vertical_idx]
    avg_vert   = reflect_data.average[f][:, vertical_idx]
    sigma_vert = reflect_data.sigma[f][:, vertical_idx]

    ref_times = pd.to_datetime(reflect_data.times[f].astype("datetime64[ns]"))
    sim_times = pd.to_datetime(sim_data.time[f])

    fig, axes = plt.subplots(nrows=3, figsize=(12, 11), sharex=True, constrained_layout=True)

    ax_ref = axes[0]
    for ii, lbl in enumerate(labels_vert):
        if "AV" in lbl:
            continue
        ax_ref.errorbar(ref_times, avg_vert[:, ii], yerr=1.96*sigma_vert[:, ii],
                        marker="o", capsize=4, label=lbl)
    ax_ref.set_ylabel("Measured Reflectance")
    ax_ref.legend(fontsize=9, loc="lower left")
    ax_ref.grid(True)

    month_str = ref_times[0].strftime("%B %Y")
    ax_ref.set_title(f"Reflectance, Wind & Dust/RH — {month_str}")

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

    ax_dust = axes[2]
    tsp = sim_data.dust_concentration[f]
    ax_dust.plot(sim_times, tsp, color="brown", lw=0.8,
                 label=f"TSP (mean={np.nanmean(tsp):.1f} µg/m³)")
    ax_dust.set_ylabel(r"TSP [$\mu$g/m³]")
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
    lines_rh,   labels_rh   = ax_rh.get_legend_handles_labels()
    ax_dust.legend(lines_dust + lines_rh, labels_dust + labels_rh, fontsize=9)

    ax_dust.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax_dust.tick_params(axis="x", rotation=15)

    plt.show()

# %% Plot each correlation with all mirrors

mirrors_of_interest_sorted = sorted(mirrors_of_interest, key=lambda x: x[0][1], reverse=True)
mirrors_by_tilt = {}
for key, m in mirrors_of_interest_sorted:
    mirrors_by_tilt.setdefault(key[1], []).append((key, m))
tilts_sorted = sorted(mirrors_by_tilt.keys(), reverse=True)
max_cols = max(len(v) for v in mirrors_by_tilt.values())

for label, x_func in all_correlations.items():
    fig, axes = plt.subplots(len(tilts_sorted), max_cols,
                             figsize=(4 * max_cols, 3 * len(tilts_sorted)),
                             sharey='row', squeeze=False)

    for row, tilt in enumerate(tilts_sorted):
        mirrors_tilt = mirrors_by_tilt[tilt]

        for col, (key, m) in enumerate(mirrors_tilt):
            ax = axes[row, col]
            ax.set_title(m, fontsize=10)

            for ii, f in enumerate(files):
                if m not in reflect_data.mirror_names[ii]:
                    continue
                jj_local = list(reflect_data.mirror_names[ii]).index(m)
                sr = soiling_rate[ii][:, jj_local]
                x  = x_func(ii, key)

                labeled = False
                for is_night, marker in [(False, 'o'), (True, '^')]:
                    sel = crosses_midnight[ii] == is_night
                    if sel.any():
                        lbl = campaign_title(f) if not labeled else None
                        ax.scatter(x[sel], sr[sel], marker=marker, label=lbl,
                                   color=f"C{ii}", s=40, zorder=3)
                        labeled = True

                mask = np.isfinite(x) & np.isfinite(sr)
                if mask.sum() >= 2:
                    z = np.polyfit(x[mask], sr[mask], 1)
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
                    ax.plot(x_line, np.polyval(z, x_line),
                            color=f"C{ii}", ls='-', linewidth=1.2)

                for is_night, ls in [(False, '--'), (True, '-.')]:
                    sel = crosses_midnight[ii] == is_night
                    m_dn = sel & np.isfinite(x) & np.isfinite(sr)
                    if m_dn.sum() >= 2:
                        z = np.polyfit(x[m_dn], sr[m_dn], 1)
                        x_line = np.linspace(x[m_dn].min(), x[m_dn].max(), 50)
                        ax.plot(x_line, np.polyval(z, x_line),
                                color=f"C{ii}", ls=ls, linewidth=1.0)

                if mask.sum() >= 3:
                    r, p = pearsonr(x[mask], sr[mask])
                    ax.annotate(f"r={r:+.2f}\np={p:.2f}",
                                xy=(0.05, 0.95), xycoords='axes fraction',
                                fontsize=8, va='top')

            ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
            ax.axvline(0, color='k', linewidth=0.6, linestyle='--')
            ax.set_xlabel(label, fontsize=8)
            ax.grid(True)

        axes[row, 0].set_ylabel("Soiling rate (%/day)", fontsize=9)

        for col in range(len(mirrors_tilt), max_cols):
            axes[row, col].set_visible(False)

    marker_handles = [
        Line2D([0], [0], marker='o', color='gray', ls='--', ms=6, label='Day'),
        Line2D([0], [0], marker='^', color='gray', ls='-.', ms=6, label='Night'),
    ]
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles + marker_handles, labels_leg + ['Day', 'Night'],
               loc='lower center', ncol=len(files) + 2,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(label, fontsize=12)
    plt.tight_layout()
    plt.show()

# %% Daily-average correlation analysis

reflect_data_d = smu.daily_average(reflect_data, sim_data.time, dt=None)

soiling_rate_d = {}
for ii, _ in enumerate(reflect_data_d.file_name):
    dr   = np.diff(reflect_data_d.average[ii], axis=0)
    dt_d = np.diff(reflect_data_d.times[ii]).astype('timedelta64[s]').astype(float) / 3600 / 24
    soiling_rate_d[ii] = -dr / dt_d[:, np.newaxis]

ws_d   = {}
wd_d   = {}
tsp_d  = {}
rh_d   = {}
sp_d   = {}
snp_d  = {}

for ii, _ in enumerate(sim_data.file_name):
    n_d = len(reflect_data_d.times[ii]) - 1
    ws_d[ii]  = np.zeros(n_d)
    wd_d[ii]  = np.zeros(n_d)
    tsp_d[ii] = np.full(n_d, np.nan)
    rh_d[ii]  = np.full(n_d, np.nan)
    sp_d[ii]  = {key: np.zeros(n_d) for key in mirror_keys}
    snp_d[ii] = {key: np.zeros(n_d) for key in mirror_keys}

    for kk in range(n_d):
        t0 = reflect_data_d.times[ii][kk]
        t1 = reflect_data_d.times[ii][kk + 1]
        tmask = (sim_data.time[ii] >= t0) & (sim_data.time[ii] < t1)
        mask  = tmask & wind_mask[ii]

        ws_d[ii][kk]  = np.nanmean(sim_data.wind_speed[ii][mask])
        tsp_d[ii][kk] = np.nanmean(sim_data.dust_concentration[ii][mask])
        if len(sim_data.relative_humidity) > 0:
            rh_d[ii][kk] = np.nanmean(sim_data.relative_humidity[ii][mask])

        wd_rad = np.deg2rad(sim_data.wind_direction[ii][mask])
        wd_d[ii][kk] = np.rad2deg(np.arctan2(
            np.nanmean(np.sin(wd_rad)),
            np.nanmean(np.cos(wd_rad))
        )) % 360

        for key in mirror_keys:
            sp_d[ii][key][kk]  = np.nanmean(s_proj[ii][key][mask])
            snp_d[ii][key][kk] = np.nanmean(sn_proj[ii][key][mask])

all_correlations_d = {
    "Wind speed":                    lambda ii, _: ws_d[ii],
    "Wind direction":                lambda ii, _: wd_d[ii],
    "Wind projection":               lambda ii, key: sp_d[ii][key],
    "Wind proj (norm)":              lambda ii, key: snp_d[ii][key],
    "TSP":                           lambda ii, _: tsp_d[ii],
    "Wind speed × TSP":              lambda ii, _: ws_d[ii]   * tsp_d[ii],
    "Wind proj × TSP":               lambda ii, key: sp_d[ii][key]  * tsp_d[ii],
    "Wind proj (norm) × TSP":        lambda ii, key: snp_d[ii][key] * tsp_d[ii],
    "RH":                            lambda ii, _: rh_d[ii],
    "Wind speed × RH":               lambda ii, _: ws_d[ii]   * rh_d[ii],
    "Wind proj × RH":                lambda ii, key: sp_d[ii][key]  * rh_d[ii],
    "Wind proj (norm) × RH":         lambda ii, key: snp_d[ii][key] * rh_d[ii],
    "TSP × RH":                      lambda ii, _: tsp_d[ii]  * rh_d[ii],
}

for label, x_func in all_correlations_d.items():
    fig, axes = plt.subplots(len(tilts_sorted), max_cols,
                             figsize=(4 * max_cols, 3 * len(tilts_sorted)),
                             sharey='row', squeeze=False)

    for row, tilt in enumerate(tilts_sorted):
        mirrors_tilt = mirrors_by_tilt[tilt]

        for col, (key, m) in enumerate(mirrors_tilt):
            ax = axes[row, col]
            ax.set_title(m, fontsize=10)

            for ii, f in enumerate(files):
                if m not in reflect_data_d.mirror_names[ii]:
                    continue
                jj_local = list(reflect_data_d.mirror_names[ii]).index(m)
                sr = soiling_rate_d[ii][:, jj_local]
                x  = x_func(ii, key)

                ax.scatter(x, sr, color=f"C{ii}", label=campaign_title(f), s=40, zorder=3)

                mask = np.isfinite(x) & np.isfinite(sr)
                if mask.sum() >= 2:
                    z = np.polyfit(x[mask], sr[mask], 1)
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
                    ax.plot(x_line, np.polyval(z, x_line), color=f"C{ii}", ls='-', linewidth=1.2)

                if mask.sum() >= 3:
                    r, p = pearsonr(x[mask], sr[mask])
                    ax.annotate(f"r={r:+.2f}\np={p:.2f}",
                                xy=(0.05, 0.95), xycoords='axes fraction',
                                fontsize=8, va='top')

            ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
            ax.axvline(0, color='k', linewidth=0.6, linestyle='--')
            ax.set_xlabel(label, fontsize=8)
            ax.grid(True)

        axes[row, 0].set_ylabel("Soiling rate (%/day)", fontsize=9)

        for col in range(len(mirrors_tilt), max_cols):
            axes[row, col].set_visible(False)

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', ncol=len(files),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{label} — Daily average", fontsize=12)
    plt.tight_layout()
    plt.show()
# %%
