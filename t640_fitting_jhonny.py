# %% 
import numpy as np
import matplotlib.pyplot as plt
import heliosoil.dust_distributions as dd
import pandas as pd
from dust_dynesty import (ModelConfig, Priors, MVNPriors, make_prior_transform,
                                  expected_bin_counts, simulate_counts,run_dynesty,
                                  unpack_full)


model_file = "fitting_simulated_data_flow_noise.stan"
modes = 3 # max 3 usually
automatic_prior = False     # need to set physical prior below
multivariate_priors = True  # need to set samples from literature for each mode below
ρ = 2.0
v = 36.0 * 60 # 36.0cm^3/min


# %% Load in T640 data
df = pd.read_csv('data/Jhonny/ALL_hourly_average_bin_concentration.csv')
df.set_index('Date_Time',inplace=True)
window = 3  # number of neighboring columns to include, including the target itself
df_filled = df.apply(lambda row: row.fillna(
    row.rolling(window=window, center=True, min_periods=1).mean()
), axis=1)
df = df_filled

col_names = df.columns
diameters = col_names[1::].to_numpy(np.float64)
new_col_names = [col_names[0]]
new_col_names.extend([f"bin_{ii:d}" for ii in range(len(col_names)-1)])
df.columns = new_col_names

de0 = diameters[1]-diameters[0]
bin_edges = np.log10(np.r_[diameters[0]-de0,diameters])
bin_centers = np.log10(diameters)

# import T640
# dat = T640.T640()
# dat.import_data('data/Jhonny/DUST_T640_ALL.xlsx')
# dat_hourly = dat.downsample('60min')

# %% Select rows
row_indices = [0,1,2,3]
all_counts = []
for ii in row_indices:
    all_counts.append(df.iloc[ii,1::].to_numpy(np.float64)*v)

all_counts = np.array(all_counts)

# %% Plotting samples and true distribution
for ii,counts in enumerate(all_counts):
    fig,ax = plt.subplots()
    ax.stairs(counts,edges=bin_edges,label='Data')
    # ax.set_xlim((x[0],x[-1]))
    ax.set_xticklabels([f'$10^{{{s:.0f}}}$' for s in ax.get_xticks()])
    ax.set_xlabel('Diameter [$\mu$m]')
    ax.set_ylabel(r'Number density $\left( \frac{\#}{dlog(D)\cdot cm^3} \right)$')
    

# %% Prior function
margin = 1.0
# exp_counts = (all_counts/v).sum(axis=1).mean()
exp_counts = np.nanmean(np.nansum(all_counts,axis=1)) # expected number of particles per cubic cm
def set_priors(m=None):
    if automatic_prior:
        pri = Priors.set(k,bin_edges,margin=margin,n_expected=(exp_counts))

    elif multivariate_priors:
        mode1_samples = np.array([  [7100,0.0117,0.232],
                                    [133,0.008,0.657],
                                    [6650,0.015,0.225],
                                    [3200,0.02,0.161],
                                    [726,0.002,0.247]])

        mode2_samples = np.array([  [6320,0.0373,0.250],
                                    [66.6,0.266,0.210],
                                    [147,0.054,0.557],
                                    [2900,0.116,0.217],
                                    [114,0.038,0.770]])

        mode3_samples = np.array([  [960,0.151,0.204],
                                    [3.1,0.58,0.396],
                                    [1990,0.084,0.266],
                                    [0.3,1.8,0.380],
                                    [0.178,21.6,0.438]])
        
        pri = MVNPriors.from_samples([mode1_samples, 
                                      mode2_samples, 
                                      mode3_samples])

    else:
        if m == 1:
            pri = Priors(mu_loc = (-1.0,), mu_scale= (0.5,),
                    sigma_median = (0.5,), sigma_lnsd = (0.5,),
                    A_median=(exp_counts,), A_lnsd = (1.0,)
                    )
        elif m == 2:
            pri = Priors(mu_loc = (-1.93, -1.29), mu_scale= (0.3, 0.3),
                        sigma_median = (0.25, 0.35), sigma_lnsd = (0.25, 0.25),
                        A_median=(0.75*exp_counts, 0.25*exp_counts), 
                        A_lnsd = (1.0, 1.0))
        elif m == 3:
            pri = Priors(mu_loc = (-1.887, -1.052, -0.470), mu_scale= (0.146, 0.327, 0.517),
                        sigma_median = (0.319, 0.309, 0.3115), sigma_lnsd = (0.528, 0.39, 0.27),
                        A_median=(4271, 2358, 738), 
                        A_lnsd = (1.628, 1.93, 1.62))
    return pri

 
# %% Fit trimodal model
cfg = ModelConfig(n_modes=modes, fit_r=True, fit_delta=False) 
pri = set_priors(modes)
res = run_dynesty(bin_edges, all_counts, cfg, pri, nlive=500)
print(f'Evidence log Z = {res.logz[-1]:.2f} +/- {res.logzerr[-1]:.2f}')

# %%
eq = res.samples_equal()    

K = cfg.n_modes
A = eq[:,0:K]
mu = eq[:,K:2 * K]
sigma = eq[:,2 * K:3 * K]

# pull r by name, fall back to the fixed value if it wasn't sampled
if "r" in cfg.names:
    r = eq[:, cfg.names.index("r")]
else:
    r = np.full(len(eq), cfg.r_fixed)


fig,ax = plt.subplots(figsize=(10,10),nrows=3,ncols=K)
if K==1:
    ax = np.atleast_2d(np.asarray(ax)).T

for k in range(cfg.n_modes):
    ax[0,k].hist(A[:,k],1000)
    # ax[0,k].set_xscale('log')
    ax[0,k].set_xlim((np.percentile(A[:,k],0),np.percentile(A[:,k],99.0)))

    ax[1,k].hist(mu[:,k],100)
    ax[2,k].hist(sigma[:,k],100)

    # ax[0,k].axvline(N[k],color='red',ls=':')
    # ax[1,k].axvline(log10_μ[k],color='red',ls=':')
    # ax[2,k].axvline(σ_log10[k],color='red',ls=':')
    
# fig.suptitle(f'Posteriors (fit k={K:d}, true k={len(σ_log10):d})',fontsize=18)
fig.tight_layout()


# r
if "r" in cfg.names:
    figr, axr = plt.subplots()
    axr.hist(r, 100)
    # axr.axvline(r_true, color="red", ls=":")
    axr.set_xscale("log")
    axr.set_title("Overdispersion parameter (r)")   # also fixes the ax→axr bug from #1

# %% posterior predictive 
def predictive(theta_samples, cfg, edges, rng=None):
    """Parameter draws -> (y_rep, lam): replicated bin counts and 
    expected counts."""
    rng = rng or np.random.default_rng()
    S, Nb = len(theta_samples), len(edges) - 1
    y_rep, lam = np.empty((S, Nb)), np.empty((S, Nb))
    for k, th in enumerate(theta_samples):
        A, mu, sigma, r, sigma_m, delta = unpack_full(th, edges, cfg) # cfg.unpack(th)
        lam[k]   = expected_bin_counts(edges, A, mu, sigma, sigma_m, delta)
        y_rep[k] = simulate_counts(edges, A, mu, sigma, r=r,
                                   sigma_m=sigma_m, delta=delta, rng=rng)
    return y_rep, lam

def plot_predictive(y_samples,cfg,edges,rng=None,kind='Posterior'):

    centers = edges[0:-1] + np.diff(edges)
    lo,med_lo, med, med_hi, hi = np.percentile(y_samples, [5, 25, 50, 75, 95], axis=0)
    fig,ax = plt.subplots()
    # actual_counts = np.diff(number_dist.cumulative(edges))
    for ii,c in enumerate(all_counts):
        if ii == 0:
            label = "Observed"
        else:
            label = '_'
        ax.stairs(c,edges=edges,ls='-',label=label,color='black',alpha=0.2)

    ax.stairs(med,edges=edges,ls='--',label=f'{kind} Predictive (Median)',color='blue')

    # ax.plot(centers,actual_counts,color='red',ls=':',label='Actual')
    ax.fill_between(centers,y1=lo,y2=hi,label=f'[{kind}] Predictive (CI)',alpha=0.2,color='blue')
    ax.fill_between(centers,y1=med_lo,y2=med_hi,alpha=0.2,color='blue')

    ax.set_xlim((edges[0]-margin,edges[-1]+margin))
    ax.set_xticklabels([f'$10^{{{s:.0f}}}$' for s in ax.get_xticks()])
    ax.set_xlabel('Diameter [$\mu$m]')
    ax.set_ylabel(f'Bin counts [#]')
    ax.legend()
    plt.show()
    return fig,ax

# cfg, pri = ModelConfig(n_modes=3), Priors()
rng = np.random.default_rng(0)

# --- prior predictive: draw theta from the prior via the transform ---
ptform = make_prior_transform(cfg, pri)
theta_prior = np.array([ptform(rng.random(cfg.ndim)) for _ in range(2000)])
y_prior, lam_prior = predictive(theta_prior, cfg, bin_edges, rng)
fig,ax = plot_predictive(y_prior,cfg,bin_edges,kind='Prior')

# --- posterior predictive: draw theta from the fitted posterior ---
theta_post = res.samples_equal()                      # res = run_dynesty(...)
y_post, lam_post = predictive(theta_post, cfg, bin_edges, rng)
fig,ax = plot_predictive(y_post,cfg,bin_edges,kind='Posterior')


# %% plot predictive over a wider range
def density(log_d, A, mu, sigma):
    z = (np.asarray(log_d)[:, None] - np.asarray(mu)[None, :]) / np.asarray(sigma)[None, :]
    return (np.exp(-0.5 * z**2) / (np.asarray(sigma) * np.sqrt(2*np.pi))) @ np.asarray(A)

def extended_grid(bin_edges, ext=1.2, n_margin=80):
    """Real bin edges over the observed interval, plus n_margin points across
    each margin of width `ext` (decades)."""
    be = np.asarray(bin_edges, float)
    lo = np.linspace(be[0] - ext, be[0],  n_margin + 1)[:-1]
    hi = np.linspace(be[-1], be[-1] + ext, n_margin + 1)[1:]
    return np.concatenate([lo, be, hi])

def predictive_density(theta_samples, cfg, obs_edges, grid):
    """Density on `grid` per draw; full amplitudes recovered from obs_edges."""
    dens = np.empty((len(theta_samples), len(grid)))
    for k, th in enumerate(theta_samples):
        A, mu, sigma, *_ = unpack_full(th, obs_edges, cfg)
        dens[k] = density(grid, A, mu, sigma)
    return dens

def plot_density_band(ax, grid, dens, q=(5, 95), color="C0", label="", floor=0.0):
    # lo, med, hi = np.percentile(dens, [q[0], 50, q[1]], axis=0)
    lo,med_lo, med, med_hi, hi = np.percentile(dens, [q[0], 25, 50, 75, q[1]], axis=0)
    clip = lambda a: np.clip(a, floor, None) if floor > 0 else a
    ax.fill_between(grid, clip(lo), clip(hi), alpha=0.25, color=color, label=f"{label} {q[0]}\u2013{q[1]}%")
    ax.fill_between(grid, clip(med_lo), clip(med_hi), alpha=0.4, color=color, label=f"{label} {25}\u2013{75}%")
    ax.plot(grid, clip(med), color=color, lw=1.8, label=f"{label} median")

def plot_predictive_density(theta_prior, eq, cfg, bin_edges, counts, v, truth=None,
                            ext=1.2, n_margin=80, q=(5, 95), logy=True, floor=1e-3,
                            figsize=(13, 5),sharex=True,sharey=True):
    grid   = extended_grid(bin_edges, ext, n_margin)
    d_pri  = predictive_density(theta_prior, cfg, bin_edges, grid)
    d_post = predictive_density(eq,          cfg, bin_edges, grid)
    obs    = counts / v / np.diff(bin_edges)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=sharex, sharey=sharey)
    fl = floor if logy else 0.0
    for ax, (d, col, name) in zip(axes, [(d_pri, "C1", "Prior"), (d_post, "C0", "Posterior")]):
        ax.axvspan(bin_edges[0], bin_edges[-1], color="0.9", zorder=0, label="observed window")
        plot_density_band(ax, grid, d, q=q, color=col, label=name, floor=fl)
        if truth is not None:                                              # <-- true distribution
            td = np.asarray(truth(grid), float)
            ax.plot(grid, np.clip(td, fl, None) if logy else td,
                    color="red", ls="--", lw=1.5, label="true")
        ax.stairs(np.clip(obs, fl, None) if logy else obs, bin_edges, color="k", lw=1.3, label="observed")
        if logy:
            ax.set_yscale("log"); ax.set_ylim(bottom=floor)
        ax.set_xlim(grid[0], grid[-1]); ax.set_xlabel(r"$\log_{10}(D/\mu m)$")
        ax.set_title(f"{name} predictive"); ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$n_N(\log D)$ [cm$^{-3}$ per $d\log D$]")
    fig.tight_layout()
    return fig, axes

fig,axes = plot_predictive_density(theta_prior, eq, cfg, bin_edges, counts, 1, 
                        truth=None, logy=False,ext=1.2,sharey=True)   # or logy=False
# xl = [a.get_xlim()[0] for a in axes]
# axes[0].set_xlim((-1,0.5))
# axes[1].set_ylim((0,20))
plt.show()

# %% Compare mass
def convert_samples(theta_samples, cfg, edges, rho, source="number"):
    """Convert (S, ndim) theta samples to the OTHER kind (number <-> mass) using
    the dust_distributions conversions. `source` is the kind the theta represents
    ("number" or "mass"). Returns (amp, mu, sigma), each (S, K). r / sigma_m /
    delta are measurement parameters and are left untouched. rho is a scalar
    float (the class methods assert that)."""
    Cls     = dd.NumberDistribution if source == "number" else dd.MassDistribution
    convert = (lambda d: d.to_mass(rho)) if source == "number" else (lambda d: d.to_number(rho))
    S, K = len(theta_samples), cfg.n_modes
    amp = np.empty((S, K)); mu = np.empty((S, K)); sig = np.empty((S, K))
    for i, th in enumerate(theta_samples):
        A, m, s, *_ = unpack_full(th, edges, cfg)
        gmm = convert(Cls.from_params(np.concatenate([A, m, s]))).distribution
        amp[i], mu[i], sig[i] = gmm.weights, gmm.mus, gmm.sigmas
    return amp, mu, sig

LD = np.linspace(-3,3,100)
# mass_dist = number_dist.to_mass(ρ)
# maxMass = sum(mass_dist.distribution.weights)

# Prior mass distribution
Am_prior, mu_m_prior, sig_m_prior = convert_samples(theta_prior, cfg, bin_edges, rho=ρ, source="number")   # number -> mass
dens_mass_prior = np.array([density(LD, Am_prior[i], mu_m_prior[i], sig_m_prior[i]) for i in range(len(Am_prior))])
lo_prior,med_lo_prior, med_prior, med_hi_prior, hi_prior = np.percentile(dens_mass_prior, [5, 25, 50, 75, 95], axis=0)

fig,ax = plt.subplots(ncols=2,sharey=True,figsize=(10,5))
ax[0].set_xlim((LD[0],LD[1]))
# ax[0] = mass_dist.plot(ax=ax[0],x_grid=LD,legend=False,
                    #    mplkwds={'color':"red","ls":'--','label':'Truth'})
ax[0].plot(LD,med_prior,color='black',label='Median')
ax[0].fill_between(LD, lo_prior, hi_prior, alpha=0.25, color='blue', label=f"Posterior {5}\u2013{95}%")
ax[0].fill_between(LD, med_lo_prior, med_hi_prior, alpha=0.4, color='blue', label=f"Posterior {25}\u2013{75}%")
ax[0].set_title('Prior Mass')
ax[0].legend()
ax[0].set_ylim((0,5000))

# Posterior mass distribution
Am_post, mu_m_post, sig_m_post = convert_samples(eq, cfg, bin_edges, rho=ρ, source="number")   # number -> mass
dens_mass_post = np.array([density(LD, Am_post[i], mu_m_post[i], sig_m_post[i]) for i in range(len(Am_post))])
lo_post,med_lo_post, med_post, med_hi_post, hi_post = np.percentile(dens_mass_post, [5, 25, 50, 75, 95], axis=0)

# ax[1] = mass_dist.plot(ax=ax[1],x_grid=LD,legend=False,
                    #    mplkwds={'color':"red","ls":'--','label':'Truth'})
ax[1].plot(LD,med_post,color='black',label='Median')
ax[1].fill_between(LD, lo_post, hi_post, alpha=0.25, color='blue', label=f"Posterior {5}\u2013{95}%")
ax[1].fill_between(LD, med_lo_post, med_hi_post, alpha=0.4, color='blue', label=f"Posterior {25}\u2013{75}%")
ax[1].legend()
ax[1].set_title('Posterior Mass')
ax[1].set_ylim((0,100))
fig.tight_layout()


# %%
