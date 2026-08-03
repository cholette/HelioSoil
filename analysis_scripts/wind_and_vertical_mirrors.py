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
import pathlib

# %% functions

def cardinal_to_uv(cardinal: str) -> tuple[float, float]:
    """
    Convert a cardinal direction string to (U, V) unit vector components.
    U: positive = eastward, V: positive = northward.
    Direction is where the wind is GOING (not coming from). # HOW IS THIS APPLIED COHERENTLY BETWEEN MIRRORS NORMALS AND WIND DIRECTION?
    """
    # angles = {
    #     'N':   90, 'NNE': 67.5, 'NE':  45, 'ENE': 22.5,
    #     'E':    0, 'ESE': -22.5, 'SE': -45, 'SSE': -67.5,
    #     'S':  -90, 'SSW': -112.5, 'SW': -135, 'WSW': -157.5,
    #     'W':  180, 'WNW': 157.5, 'NW': 135, 'NNW': 112.5,
    # }
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
    """Extract orientation and tilt from e.g. 'ONW_M1_T90' -> ('NW', 90)"""
    parts = name.split('_')
    orientation = parts[0][1:]      # 'ONW' -> 'NW'
    tilt = int(parts[2][1:])        # 'T90' -> 90
    return orientation, tilt

def campaign_title(filepath):
    """Extract month name from filename e.g. '_experiment_20250206_20250212_...' -> 'February 2025'"""
    basename = os.path.basename(filepath)
    match = re.search(r'_(\d{8})_', basename)
    if match:
        date = datetime.strptime(match.group(1), '%Y%m%d')
        return date.strftime('%B %Y')   # e.g. 'February 2025'
    return basename  # fallback to full name if pattern not found

def circular_linear_corr(theta_deg, y):
    mask = np.isfinite(theta_deg) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    theta = np.deg2rad(theta_deg[mask])
    y_ = y[mask]
    r_cs, _ = pearsonr(np.sin(theta), y_)
    r_cc, _ = pearsonr(np.cos(theta), y_)
    r_sc, _ = pearsonr(np.sin(theta), np.cos(theta))
    r2 = (r_cs**2 + r_cc**2 - 2*r_cs*r_cc*r_sc) / (1 - r_sc**2)
    r_cl = np.sqrt(max(r2, 0))
    p = 1 - sps.chi2.cdf(len(y_) * r2, df=2)
    return r_cl, p

def print_correlation(label, x, y, mirror):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        print(f"  {mirror:<15} | {label:<45} | not enough data")
        return
    r_p, p_p = pearsonr(x[mask],  y[mask])
    r_s, p_s = spearmanr(x[mask], y[mask])
    print(f"  {mirror:<15} | {label:<45} | Pearson r={r_p:+.3f} (p={p_p:.3f})  Spearman r={r_s:+.3f} (p={p_s:.3f})")

# %% Analysis of wind

tilts_of_interest = [90, 60, 30, 0]
time_to_remove_at_end = [0, 0, 0, 0, 0, 0]  # hours to remove at end of each experiment
k_factor = "import"
dust_type = "PM10"

# For yadnarie
# main_directory = pathlib.Path('../data/')
# d = main_directory/"yadnarie/"
d = os.path.join(main_directory, "data", "yadnarie") + "\\"
save_file = os.path.join(main_directory,"results","wind_results_yadnarie")
# compass = 2
all_intervals = np.array([
    [np.datetime64('2024-11-11T20:30:00'), np.datetime64('2024-11-16T07:30:00')],
    [np.datetime64('2024-11-12T11:40:00'), np.datetime64('2024-11-16T07:30:00')]],
    dtype='datetime64[m]')
reflectometer_incidence_angle = 15 # [deg]
reflectometer_acceptance_angle = 12.5e-3 # [rad] half acceptance angle

files, all_intervals, exp_mirrors, all_mirrors = smu.get_training_data(d, "_experiment_", time_to_remove_at_end=time_to_remove_at_end)
sim_data = smb.simulation_inputs(files, k_factors=k_factor, dust_type=dust_type)
wind_mask = {ii: sim_data.wind_speed[ii] > 0.001 for ii in range(len(sim_data.file_name))}   # %% Wind mask (exclude zero wind, 0.001 required for approximation issue [it never reads 0])

# %%
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

# %% Get soiling rates between measurements
soiling_rate = {}
for ii,f in enumerate(reflect_data.file_name):
    dr = np.diff(reflect_data.average[ii],axis=0)
    dt = np.diff(reflect_data.times[ii]).astype('timedelta64[s]').astype(float)/3600/24
    soiling_rate[ii] = -dr/dt[:,np.newaxis]         # soiling rate is >0 when rho(t+1) < rho(t)

# %% Mirror normals (positive N,E) -- IS THIS ACTUALLY NORMALS OR SIMPLY WHERE THEY POINT TO? normals should be in 3D
mirror_names = []
for ii,_ in enumerate(reflect_data.file_name):
    mirror_names.extend(reflect_data.mirror_names[ii])
mirror_names = np.unique(mirror_names)

mirror_i = []
mirror_j = []
mirror_keys = []
for m in mirror_names:
    orientation, tilt = parse_mirror_name(m)
    u,v=cardinal_to_uv(orientation)
    mirror_i.append(u)
    mirror_j.append(v)
    mirror_keys.append((orientation, tilt))

mirrors_to_exclude = {'ONE_M1', 'OSW_M2'}
mirrors_of_interest = [(key, m) for key, m in zip(mirror_keys, mirror_names)
                       if key[1] in tilts_of_interest
                       and '_'.join(m.split('_')[:2]) not in mirrors_to_exclude]

print("Mirrors of interest:")
for key, m in mirrors_of_interest:
    print(f"  {m} -> {key}")

# %% Wind direction
wind_nu = {}
wind_nv = {}
wind_u = {}
wind_v = {}
for ii,_ in enumerate(sim_data.file_name):
    wind_nu[ii] = -np.sin(np.deg2rad(sim_data.wind_direction[ii]))  # normals of wind (coming from)
    wind_nv[ii] = -np.cos(np.deg2rad(sim_data.wind_direction[ii]))  # normals of wind (coming from)
    wind_u[ii] = sim_data.wind_speed[ii]*wind_nu[ii]                # wind speed vector (coming from)
    wind_v[ii] = sim_data.wind_speed[ii]*wind_nv[ii]                # wind speed vector (coming from)

# %% Projection of wind vector onto normal of mirror
n_mirrors = len(mirror_names)
s_proj = {}
sn_proj = {}
for ii,_ in enumerate(sim_data.file_name):
    s_proj[ii] = {}
    sn_proj[ii] = {}
    for jj, key in enumerate(mirror_keys):
        sn_proj[ii][key] = wind_nu[ii] * mirror_i[jj] + wind_nv[ii] * mirror_j[jj]
        s_proj[ii][key] = wind_u[ii] * mirror_i[jj] + wind_v[ii] * mirror_j[jj]


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
    fig.suptitle(f"sn_proj — {campaign_title(f)}")
    plt.tight_layout()
    plt.show()

# %% Plots

for ii,f in enumerate(files):
    fig,ax = smu.wind_rose(sim_data,ii,wind_mask[ii])
    ax.set_title(campaign_title(f))
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.plot(sim_data.time[ii],sim_data.wind_direction[ii])
    ax1.set_title(campaign_title(f))
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Wind direction (°)")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=25, ha='right')
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(sim_data.time[ii],sim_data.wind_speed[ii])
    ax2.set_title(campaign_title(f))
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Wind speed (m/s)")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=25, ha='right')
    plt.tight_layout()
    plt.show()

# %% Aggregate wind and dust over each soiling interval
wind_speed_mean  = {}
wind_dir_mean    = {}
pm25_mean        = {}
pm10_mean        = {}
rh_mean          = {}
crosses_midnight = {}
s_proj_mean      = {}
sn_proj_mean     = {}

for ii, _ in enumerate(sim_data.file_name):
    n_intervals = len(reflect_data.times[ii]) - 1
    wind_speed_mean[ii]  = np.zeros(n_intervals)
    wind_dir_mean[ii]    = np.zeros(n_intervals)
    pm25_mean[ii]        = np.zeros(n_intervals)
    pm10_mean[ii]        = np.zeros(n_intervals)
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

        # Exclude timesteps where wind speed is zero (other variables are unreliable)
        mask = time_mask & wind_mask[ii]

        wind_speed_mean[ii][kk] = np.nanmean(sim_data.wind_speed[ii][mask])
        pm25_mean[ii][kk]       = np.nanmean(sim_data.pm2p5[ii][mask])
        pm10_mean[ii][kk]       = np.nanmean(sim_data.pm10[ii][mask])
        if len(sim_data.relative_humidity) > 0:
            rh_mean[ii][kk]     = np.nanmean(sim_data.relative_humidity[ii][mask])

        # Circular mean for wind direction
        wd_rad = np.deg2rad(sim_data.wind_direction[ii][mask])
        wind_dir_mean[ii][kk] = np.rad2deg(np.arctan2(
            np.nanmean(np.sin(wd_rad)),
            np.nanmean(np.cos(wd_rad))
        )) % 360

        for key in mirror_keys:
            s_proj_mean[ii][key][kk]  = np.nanmean(s_proj[ii][key][mask])
            sn_proj_mean[ii][key][kk] = np.nanmean(sn_proj[ii][key][mask])

# %% Plot wind projection for 90 degree mirrors
for tilt in tilts_of_interest:
    mirrors_tilt = [(key, m) for key, m in mirrors_of_interest if key[1] == tilt]
    n_mirrors_tilt = len(mirrors_tilt)
    n_campaigns = len(files)

    fig, axes = plt.subplots(n_mirrors_tilt, n_campaigns,
                             figsize=(6*n_campaigns, 3*n_mirrors_tilt),
                             sharey='row', sharex='col')

    # Ensure 2D array for indexing
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

    fig.suptitle(f"Wind projection onto mirror normal — {tilt}° tilt mirrors",
                 fontsize=13)
    plt.tight_layout()
    plt.show()

# %% Correlations

for ii, f in enumerate(files):
    print(f"\n{'='*100}")
    print(f"Campaign: {campaign_title(f)}")
    print(f"{'='*100}")

    ws  = wind_speed_mean[ii]
    wd  = wind_dir_mean[ii]
    p25 = pm25_mean[ii]
    p10 = pm10_mean[ii]
    rh  = rh_mean[ii]

    for jj, (key, m) in enumerate(zip(mirror_keys, mirror_names)):
        # Skip if mirror not in this campaign
        if m not in reflect_data.mirror_names[ii]:
            continue

        # Local index for soiling_rate columns
        jj_local = list(reflect_data.mirror_names[ii]).index(m)

        sr    = soiling_rate[ii][:, jj_local]
        proj  = s_proj_mean[ii][key]
        nproj = sn_proj_mean[ii][key]

        print(f"\n  --- {m} ---")

        # 1) Wind speed alone
        print_correlation("Wind speed",                          ws,        sr, m)

        # 2) Wind direction alone
        print_correlation("Wind direction",                      wd,        sr, m)

        # 3) Wind projection (speed + direction)
        print_correlation("Wind projection",                     proj,      sr, m)

        # 4) Wind projection normalised (direction only)
        print_correlation("Wind projection (norm)",              nproj,     sr, m)

        # 5) PM2.5 alone
        print_correlation("PM2.5",                               p25,       sr, m)

        # 6) PM10 alone
        print_correlation("PM10",                                p10,       sr, m)

        # 7) Wind speed × PM2.5
        print_correlation("Wind speed × PM2.5",                  ws*p25,    sr, m)

        # 8) Wind speed × PM10
        print_correlation("Wind speed × PM10",                   ws*p10,    sr, m)

        # 9) Wind projection × PM2.5
        print_correlation("Wind projection × PM2.5",             proj*p25,  sr, m)

        # 10) Wind projection × PM10
        print_correlation("Wind projection × PM10",              proj*p10,  sr, m)

        # 11) Wind projection (normalised) × PM2.5
        print_correlation("Wind projection (norm) × PM2.5",      nproj*p25, sr, m)

        # 12) Wind projection (normalised) × PM10
        print_correlation("Wind projection (norm) × PM10",       nproj*p10, sr, m)

        # 13) RH alone
        print_correlation("RH",                                   rh,        sr, m)

        # 14) Wind speed × RH
        print_correlation("Wind speed × RH",                      ws*rh,     sr, m)

        # 15) Wind projection × RH
        print_correlation("Wind projection × RH",                 proj*rh,   sr, m)

        # 16) Wind projection (normalised) × RH
        print_correlation("Wind projection (norm) × RH",          nproj*rh,  sr, m)

        # 17) PM2.5 × RH
        print_correlation("PM2.5 × RH",                           p25*rh,    sr, m)

        # 18) PM10 × RH
        print_correlation("PM10 × RH",                            p10*rh,    sr, m)



# %% Compute all correlations for mirrors of interest and rank them
all_correlations = {
    "Wind speed":                    lambda ii, key: wind_speed_mean[ii],
    "Wind direction":                lambda ii, key: wind_dir_mean[ii],
    "Wind projection":               lambda ii, key: s_proj_mean[ii][key],
    "Wind proj (norm)":              lambda ii, key: sn_proj_mean[ii][key],
    "PM2.5":                         lambda ii, key: pm25_mean[ii],
    "PM10":                          lambda ii, key: pm10_mean[ii],
    "Wind speed × PM2.5":            lambda ii, key: wind_speed_mean[ii]  * pm25_mean[ii],
    "Wind speed × PM10":             lambda ii, key: wind_speed_mean[ii]  * pm10_mean[ii],
    "Wind proj × PM2.5":             lambda ii, key: s_proj_mean[ii][key] * pm25_mean[ii],
    "Wind proj × PM10":              lambda ii, key: s_proj_mean[ii][key] * pm10_mean[ii],
    "Wind proj (norm) × PM2.5":      lambda ii, key: sn_proj_mean[ii][key]* pm25_mean[ii],
    "Wind proj (norm) × PM10":       lambda ii, key: sn_proj_mean[ii][key]* pm10_mean[ii],
    "RH":                            lambda ii, _: rh_mean[ii],
    "Wind speed × RH":               lambda ii, _: wind_speed_mean[ii]  * rh_mean[ii],
    "Wind proj × RH":                lambda ii, key: s_proj_mean[ii][key] * rh_mean[ii],
    "Wind proj (norm) × RH":         lambda ii, key: sn_proj_mean[ii][key]* rh_mean[ii],
    "PM2.5 × RH":                    lambda ii, _: pm25_mean[ii] * rh_mean[ii],
    "PM10 × RH":                     lambda ii, _: pm10_mean[ii] * rh_mean[ii],
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

    # Average |r| across all mirrors and campaigns
    mean_scores = {
        label: np.mean(vals) if vals else 0.0
        for label, vals in corr_scores.items()
    }

    # Sort and return top n
    ranked = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Correlation ranking (mean |Pearson r|):")
    for label, score in ranked:
        print(f"    {label:<35} {score:.3f}")

    return [(label, all_correlations[label]) for label, _ in ranked[:n_top]]


# %% Plot top 4 correlations for each tilt
for tilt in tilts_of_interest:
    mirrors_tilt = [(key, m) for key, m in mirrors_of_interest if key[1] == tilt]
    n_mirrors_tilt = len(mirrors_tilt)

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
        # Only plot mirrors of interest (tilt 90 or 60)
        _, tilt = parse_mirror_name(m)
        if tilt not in tilts_of_interest:
            continue
        ax[0].errorbar(t, ave[:, jj], yerr=1.96*std[:, jj],
                       label=lgd_label[jj], marker='o', capsize=4.0)

    ax[0].set_ylabel("Measured Reflectance")
    ax[0].legend(fontsize=lgd_size, loc='center right', bbox_to_anchor=(1.15, 0.5))
    ax[0].grid(True)

    # Wind speed (non-zero only)
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
    ax_ref.set_title("Reflectance, Wind & Dust/RH – {month_str}")

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

ws_d  = {}
wd_d  = {}
p25_d = {}
p10_d = {}
rh_d  = {}
sp_d  = {}
snp_d = {}

for ii, _ in enumerate(sim_data.file_name):
    n_d = len(reflect_data_d.times[ii]) - 1
    ws_d[ii]  = np.zeros(n_d)
    wd_d[ii]  = np.zeros(n_d)
    p25_d[ii] = np.zeros(n_d)
    p10_d[ii] = np.zeros(n_d)
    rh_d[ii]  = np.full(n_d, np.nan)
    sp_d[ii]  = {key: np.zeros(n_d) for key in mirror_keys}
    snp_d[ii] = {key: np.zeros(n_d) for key in mirror_keys}

    for kk in range(n_d):
        t0 = reflect_data_d.times[ii][kk]
        t1 = reflect_data_d.times[ii][kk + 1]
        tmask = (sim_data.time[ii] >= t0) & (sim_data.time[ii] < t1)
        mask  = tmask & wind_mask[ii]

        ws_d[ii][kk]  = np.nanmean(sim_data.wind_speed[ii][mask])
        p25_d[ii][kk] = np.nanmean(sim_data.pm2p5[ii][mask])
        p10_d[ii][kk] = np.nanmean(sim_data.pm10[ii][mask])
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
    "PM2.5":                         lambda ii, _: p25_d[ii],
    "PM10":                          lambda ii, _: p10_d[ii],
    "Wind speed × PM2.5":            lambda ii, _: ws_d[ii]  * p25_d[ii],
    "Wind speed × PM10":             lambda ii, _: ws_d[ii]  * p10_d[ii],
    "Wind proj × PM2.5":             lambda ii, key: sp_d[ii][key] * p25_d[ii],
    "Wind proj × PM10":              lambda ii, key: sp_d[ii][key] * p10_d[ii],
    "Wind proj (norm) × PM2.5":      lambda ii, key: snp_d[ii][key] * p25_d[ii],
    "Wind proj (norm) × PM10":       lambda ii, key: snp_d[ii][key] * p10_d[ii],
    "RH":                            lambda ii, _: rh_d[ii],
    "Wind speed × RH":               lambda ii, _: ws_d[ii]  * rh_d[ii],
    "Wind proj × RH":                lambda ii, key: sp_d[ii][key] * rh_d[ii],
    "Wind proj (norm) × RH":         lambda ii, key: snp_d[ii][key] * rh_d[ii],
    "PM2.5 × RH":                    lambda ii, _: p25_d[ii] * rh_d[ii],
    "PM10 × RH":                     lambda ii, _: p10_d[ii] * rh_d[ii],
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

# %% Focused analysis: SE1 (OSE_M1) and NW1 (ONW_M1) — 7 Feb evening → 11 Feb morning 2025

t_focus_start  = np.datetime64('2025-02-07T18:00:00', 's')
t_focus_end    = np.datetime64('2025-02-11T09:00:00', 's')
focus_prefixes = {'OSE_M1', 'ONW_M1'}
_ts_str = pd.Timestamp('2025-02-07 18:00').strftime('%d %b %H:%M')
_te_str = pd.Timestamp('2025-02-11 09:00').strftime('%d %b %H:%M')

# Identify the campaign that covers the focus window
ii_focus = None
for ii in range(len(files)):
    t_sim_s = sim_data.time[ii].astype('datetime64[s]')
    if np.any((t_sim_s >= t_focus_start) & (t_sim_s <= t_focus_end)):
        ii_focus = ii
        break
if ii_focus is None:
    raise RuntimeError("No campaign covers the focus window 2025-02-07 → 2025-02-11")
f_focus = files[ii_focus]
print(f"Focus campaign: {campaign_title(f_focus)} (index {ii_focus})")

# Focus mirrors: all tilts of OSE_M1 and ONW_M1 present in the focus campaign
focus_mirrors = [
    (key, m) for key, m in mirrors_of_interest
    if '_'.join(m.split('_')[:2]) in focus_prefixes
    and m in reflect_data.mirror_names[ii_focus]
]
if not focus_mirrors:
    raise RuntimeError("No SE1/NW1 mirrors found in the focus campaign.")
print("Focus mirrors:")
for key, m in focus_mirrors:
    print(f"  {m} -> {key}")

# Sim-time mask for the focus window
sim_mask_focus = (
    (sim_data.time[ii_focus].astype('datetime64[s]') >= t_focus_start) &
    (sim_data.time[ii_focus].astype('datetime64[s]') <= t_focus_end)
)

# Ref-time mask (for plotting reflectance)
ref_times_f    = reflect_data.times[ii_focus].astype('datetime64[s]')
ref_mask_focus = (ref_times_f >= t_focus_start) & (ref_times_f <= t_focus_end)

# Soiling intervals fully contained within the focus window
n_int_f  = len(ref_times_f) - 1
focus_kk = np.array([
    kk for kk in range(n_int_f)
    if ref_times_f[kk] >= t_focus_start and ref_times_f[kk + 1] <= t_focus_end
], dtype=int)
print(f"Focus soiling intervals: {len(focus_kk)} of {n_int_f} total")
for kk in focus_kk:
    print(f"  kk={kk}: {ref_times_f[kk]} → {ref_times_f[kk + 1]}")

n_fm = len(focus_mirrors)

# --- Figure 1: 3-panel time series restricted to the focus window ---
sim_times_focus    = pd.to_datetime(sim_data.time[ii_focus][sim_mask_focus].astype('datetime64[ns]'))
ref_times_focus_pd = pd.to_datetime(ref_times_f[ref_mask_focus].astype('datetime64[ns]'))

fig_ts, (ax_ref, ax_wind, ax_dust) = plt.subplots(
    3, 1, figsize=(12, 10), sharex=True, constrained_layout=True
)

for key, m in focus_mirrors:
    jj_local = list(reflect_data.mirror_names[ii_focus]).index(m)
    lbl = re.sub(r"_T\d{2,3}", "", m.replace("O", "").replace("_M", ""))
    ax_ref.errorbar(
        ref_times_focus_pd,
        reflect_data.average[ii_focus][ref_mask_focus, jj_local],
        yerr=1.96 * reflect_data.sigma[ii_focus][ref_mask_focus, jj_local],
        marker='o', capsize=4, label=lbl
    )
ax_ref.set_ylabel("Measured Reflectance")
ax_ref.legend(fontsize=9, loc='lower left')
ax_ref.grid(True)
ax_ref.set_title(f"SE1 & NW1 — {campaign_title(f_focus)}   [{_ts_str} → {_te_str}]")

ws_f = sim_data.wind_speed[ii_focus][sim_mask_focus]
wd_f = sim_data.wind_direction[ii_focus][sim_mask_focus]
sc = ax_wind.scatter(sim_times_focus, ws_f, c=wd_f, cmap='hsv', vmin=0, vmax=360,
                     s=4, alpha=0.7, zorder=3)
cbar = fig_ts.colorbar(sc, ax=ax_wind, pad=0.01)
cbar.set_label("Wind Direction [°]", fontsize=9)
ax_wind.axhline(np.nanmean(ws_f), color='green', ls='--', lw=1,
                label=f"Mean = {np.nanmean(ws_f):.2f} m/s")
ax_wind.set_ylabel("Wind Speed [m/s]", color='green')
ax_wind.tick_params(axis='y', labelcolor='green')
ax_wind.legend(fontsize=9)
ax_wind.grid(True)

pm25_f = sim_data.pm2p5[ii_focus][sim_mask_focus]
pm10_f = sim_data.pm10[ii_focus][sim_mask_focus]
ax_dust.plot(sim_times_focus, pm25_f, color='brown', lw=0.8,
             label=f"PM2.5 (mean={np.nanmean(pm25_f):.1f} µg/m³)")
ax_dust.plot(sim_times_focus, pm10_f, color='orange', lw=0.8,
             label=f"PM10 (mean={np.nanmean(pm10_f):.1f} µg/m³)")
ax_dust.set_ylabel(r"Dust concentration [$\mu$g/m³]")
ax_dust.grid(True)

ax_rh_ts = ax_dust.twinx()
if len(sim_data.relative_humidity) > 0:
    rh_f = sim_data.relative_humidity[ii_focus][sim_mask_focus]
    ax_rh_ts.plot(sim_times_focus, rh_f, color='blue', lw=0.8,
                  label=f"RH (mean={np.nanmean(rh_f):.1f} %)")
    ax_rh_ts.axhline(np.nanmean(rh_f), color='blue', ls='--', lw=1)
ax_rh_ts.set_ylabel("Relative Humidity [%]", color='blue')
ax_rh_ts.tick_params(axis='y', labelcolor='blue')
ax_rh_ts.set_ylim(0, 100)

lines_d, labels_d = ax_dust.get_legend_handles_labels()
lines_r, labels_r = ax_rh_ts.get_legend_handles_labels()
ax_dust.legend(lines_d + lines_r, labels_d + labels_r, fontsize=9)
ax_dust.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
ax_dust.tick_params(axis='x', rotation=15)
plt.show()

# --- Figure 2: Wind projection time series for focus mirrors, focus window ---
fig_wp, axes_wp = plt.subplots(n_fm, 2, figsize=(14, 3 * n_fm),
                                sharex=True, squeeze=False)
t_proj_pd = pd.to_datetime(sim_data.time[ii_focus][sim_mask_focus].astype('datetime64[ns]'))

for row, (key, m) in enumerate(focus_mirrors):
    axes_wp[row, 0].plot(t_proj_pd, s_proj[ii_focus][key][sim_mask_focus],
                          color='steelblue', lw=0.6)
    axes_wp[row, 0].axhline(0, color='k', lw=0.8, ls='--')
    axes_wp[row, 0].set_ylabel(f"{m}\ns_proj (m/s)", fontsize=8)
    axes_wp[row, 0].grid(True)

    axes_wp[row, 1].plot(t_proj_pd, sn_proj[ii_focus][key][sim_mask_focus],
                          color='darkorange', lw=0.6)
    axes_wp[row, 1].axhline(0, color='k', lw=0.8, ls='--')
    axes_wp[row, 1].set_ylabel(f"{m}\nsn_proj", fontsize=8)
    axes_wp[row, 1].grid(True)

axes_wp[0, 0].set_title("Wind projection s_proj [m/s]", fontsize=10)
axes_wp[0, 1].set_title("Wind projection (normalised) sn_proj", fontsize=10)
for col in range(2):
    axes_wp[-1, col].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%H:%M'))
    plt.setp(axes_wp[-1, col].xaxis.get_majorticklabels(), rotation=15, ha='right')

fig_wp.suptitle(f"Wind projections — SE1 & NW1 — {campaign_title(f_focus)}", fontsize=12)
plt.tight_layout()
plt.show()

# --- Figures 3+: Correlation scatter plots, focus mirrors × focus intervals ---
if len(focus_kk) < 2:
    print(f"Only {len(focus_kk)} soiling interval(s) in focus window — skipping correlation scatter plots.")
else:
    for label, x_func in all_correlations.items():
        fig_c, axes_c = plt.subplots(1, n_fm, figsize=(5 * n_fm, 4),
                                      sharey=True, squeeze=False)

        for col, (key, m) in enumerate(focus_mirrors):
            ax = axes_c[0, col]
            ax.set_title(m, fontsize=10)

            if m not in reflect_data.mirror_names[ii_focus]:
                ax.set_visible(False)
                continue

            jj_local = list(reflect_data.mirror_names[ii_focus]).index(m)
            sr = soiling_rate[ii_focus][focus_kk, jj_local]
            x  = x_func(ii_focus, key)[focus_kk]

            ax.scatter(x, sr, color='C0', s=50, zorder=3, label=campaign_title(f_focus))

            mask = np.isfinite(x) & np.isfinite(sr)
            if mask.sum() >= 2:
                z = np.polyfit(x[mask], sr[mask], 1)
                x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
                ax.plot(x_line, np.polyval(z, x_line), color='C0', ls='-', lw=1.2)

            if mask.sum() >= 3:
                r, p = pearsonr(x[mask], sr[mask])
                ax.annotate(f"r={r:+.2f}\np={p:.2f}",
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            fontsize=9, va='top')

            ax.axhline(0, color='k', lw=0.6, ls='--')
            ax.axvline(0, color='k', lw=0.6, ls='--')
            ax.set_xlabel(label, fontsize=9)
            ax.grid(True)

        axes_c[0, 0].set_ylabel("Soiling rate (%/day)", fontsize=9)

        handles_c, labels_leg_c = axes_c[0, 0].get_legend_handles_labels()
        fig_c.legend(handles_c, labels_leg_c,
                     loc='lower center', ncol=n_fm,
                     fontsize=9, bbox_to_anchor=(0.5, -0.06))

        fig_c.suptitle(f"{label} — SE1 & NW1 focus [{_ts_str} – {_te_str}]", fontsize=12)
        plt.tight_layout()
        plt.show()

# %% Circular-linear correlation: wind direction vs PM10

print(f"\n{'='*80}")
print("Circular-linear correlation: Wind Direction vs PM10")
print(f"{'='*80}")

n_camp = len(files)
fig_wdp, axes_wdp = plt.subplots(1, n_camp, figsize=(5 * n_camp, 4),
                                   sharey=True, squeeze=False)

for ii, f in enumerate(files):
    wd  = wind_dir_mean[ii]
    p10 = pm10_mean[ii]
    r_cl, p = circular_linear_corr(wd, p10)
    print(f"  {campaign_title(f):<25}  r={r_cl:.3f}  p={p:.3f}")

    ax = axes_wdp[0, ii]
    mask = np.isfinite(wd) & np.isfinite(p10)
    ax.scatter(wd[mask], p10[mask], color=f"C{ii}", s=40, zorder=3)

    # sin+cos fit: PM10 = a + b*sin(θ) + c*cos(θ)
    if mask.sum() >= 3:
        theta = np.deg2rad(wd[mask])
        A = np.column_stack([np.ones(mask.sum()), np.sin(theta), np.cos(theta)])
        coeffs, _, _, _ = np.linalg.lstsq(A, p10[mask], rcond=None)
        theta_plot = np.linspace(0, 2 * np.pi, 360)
        fit = coeffs[0] + coeffs[1] * np.sin(theta_plot) + coeffs[2] * np.cos(theta_plot)
        ax.plot(np.rad2deg(theta_plot), fit, color=f"C{ii}", lw=1.5)

    ax.set_title(f"{campaign_title(f)}\nr={r_cl:.3f}  p={p:.3f}", fontsize=10)
    ax.set_xlabel("Wind Direction [°]")
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(['N', 'E', 'S', 'W', 'N'])
    ax.grid(True)

axes_wdp[0, 0].set_ylabel(r"PM10 [$\mu$g/m³]")
fig_wdp.suptitle("Wind Direction vs PM10", fontsize=12)
plt.tight_layout()
plt.show()
# %%
