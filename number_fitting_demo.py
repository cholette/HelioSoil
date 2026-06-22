# %% 
import numpy as np
import matplotlib.pyplot as plt
import heliosoil.dust_distributions as dd
import pandas as pd
from dust_dynesty import (ModelConfig, Priors, make_prior_transform,
                                  expected_bin_counts, simulate_counts,run_dynesty,
                                  unpack_full)


model_file = "fitting_simulated_data_flow_noise.stan"
modes = 3 # max 3 usually
automatic_prior = False # need to set physical prior below


# %% Generate binned samples from a known distribution

# %%%%%%%%%%%%%% Actual %%%%%%%%%%%%%%%%%%%%%

# Urban (Borghesani)
# N = np.array([3000,999.875,0.125])
# log10_μ = np.log10([0.0117,0.051231,0.8226])
# σ_log10 = np.log10([1.71061,2.239,2.512])

# Urban
N = np.array([7100,6320,960])
log10_μ = np.log10([0.0117,0.0373,0.151])
σ_log10 = np.array([0.232,0.250,0.204])

# Rural
# N = np.array([6650,147,1990])
# log10_μ = np.log10([0.015,0.054,0.084])
# σ_log10 = np.log10([1.6788,3.6058,1.8450])

θ = 0.01 # variance in pump flow rate
r_true = 1.0/θ
num_hists = 3
ρ = 2.0 # g/cm3

params = np.r_[N,log10_μ ,σ_log10]
number_dist = dd.NumberDistribution.from_params(params)

# sample
v = 36.0 # sampling volume [cm3]

# =============== Simulate non-truncated  ==================
# bin_edges = np.linspace(-3,2,100)
# bin_centers = 0.5*(bin_edges[0:-1] + bin_edges[1::])


# ================ Simulate with truncation (more Realistic) =================
bin_edges = pd.read_excel("data/device_bins.xlsx")['Edges'].values
bin_edges = np.log10(bin_edges)
# bin_edges = np.log10([0.1, 1.0, 2.5, 4.0, 10.0, 17.0])
margin = 1.0


all_counts= number_dist.sample_histograms(num_hists,v,bin_edges,θ=θ)
exp_counts = (all_counts/v).sum(axis=1).mean()
                                   
# %% Plotting samples and true distribution
for ii,counts in enumerate(all_counts):
    fig,ax = plt.subplots()

    count_density_per_cm3 = counts/np.diff(bin_edges) / v # scale to 1cm3
    ax.stairs(count_density_per_cm3,edges=bin_edges,label='Simulated')
    # ax.bar(bin_centers,count_density_per_cm3,width=np.diff(bin_edges))

    x = np.linspace(-3,3,1000)
    ax.plot(x,number_dist.density(x),color='red',label='Actual')
    ax.set_xlim((x[0],x[-1]))
    ax.set_xticklabels([f'$10^{{{s:.0f}}}$' for s in ax.get_xticks()])
    ax.set_xlabel('Diameter [$\mu$m]')
    ax.set_ylabel(f'Number density [{number_dist.units}]')
    ax.legend()
    ax.set_title(f'Histogram for sample {ii:d}')

# %% Prior function
def set_priors(m):
    if automatic_prior:
        pri = Priors.set(k,bin_edges,margin=margin,n_expected=(exp_counts))
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

# %% Fit model (note: selection assumes equal prior model probability)
if isinstance(modes,list):
    logZ = np.zeros(len(modes))
    print('Selecting number of modes')
    for ii,k in enumerate(modes):
        print(f'  fitting k={k}')
        cfg = ModelConfig(n_modes=k, fit_r=True, fit_delta=False)

        pri = set_priors(k)
        
        res = run_dynesty(bin_edges, all_counts/v, cfg, pri, nlive=500)
        print(f'Evidence log Z (k={k:d})= {res.logz[-1]:.2f} +/- {res.logzerr[-1]:.2f}')
        logZ[ii] = res.logz[-1]
    
    # select best mode
    b = int(np.argmax(logZ))
    best_mode = modes[b]
    gaps = np.sort(logZ[b] - logZ)                  # ascending; gaps[0] == 0 (best vs itself)
    ΔlogZ = gaps[1] if len(gaps) > 1 else np.inf    # gap to the closest competitor
    print(f"Best mode: k={best_mode}" +
        ("" if not np.isfinite(ΔlogZ) else f" (gap ΔlogZ = {ΔlogZ:.2f} to closest model)"))

elif isinstance(modes,int):
    print('Number of modes specified. Fitting model directly.')
    best_mode = modes
else:
    raise ValueError("modes needs to be a list of ints (with k selection) or an int (no k selection)")

# %% Fit selected model
cfg = ModelConfig(n_modes=best_mode, fit_r=True, fit_delta=False) 
pri = set_priors(best_mode)
res = run_dynesty(bin_edges, all_counts/v, cfg, pri, nlive=500)
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

    ax[0,k].axvline(N[k],color='red',ls=':')
    ax[1,k].axvline(log10_μ[k],color='red',ls=':')
    ax[2,k].axvline(σ_log10[k],color='red',ls=':')
    
fig.suptitle(f'Posteriors (fit k={K:d}, true k={len(σ_log10):d})',fontsize=18)
fig.tight_layout()


# r
if "r" in cfg.names:
    figr, axr = plt.subplots()
    axr.hist(r, 100)
    axr.axvline(r_true, color="red", ls=":")
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
    actual_counts = np.diff(number_dist.cumulative(edges))
    for ii,c in enumerate(all_counts):
        if ii == 0:
            label = "Observed"
        else:
            label = '_'
        ax.stairs(c/v,edges=edges,ls='-',label=label,color='black',alpha=0.2)

    ax.stairs(med,edges=edges,ls='--',label=f'{kind} Predictive (Median)',color='blue')

    ax.plot(centers,actual_counts,color='red',ls=':',label='Actual')
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

fig,axes = plot_predictive_density(theta_prior, eq, cfg, bin_edges, counts, v, 
                        truth=number_dist.density, logy=False,ext=1.2,sharey=True)   # or logy=False
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
Am, mu_m, sig_m = convert_samples(eq, cfg, bin_edges, rho=ρ, source="number")   # number -> mass
dens_mass = np.array([density(LD, Am[i], mu_m[i], sig_m[i]) for i in range(len(Am))])
lo,med_lo, med, med_hi, hi = np.percentile(dens_mass, [5, 25, 50, 75, 95], axis=0)

fig,ax = plt.subplots()
ax.plot(LD,med,color='black',label='Median')
ax.fill_between(LD, lo, hi, alpha=0.25, color='blue', label=f"Posterior {5}\u2013{95}%")
ax.fill_between(LD, med_lo, med_hi, alpha=0.4, color='blue', label=f"Posterior {25}\u2013{75}%")
ax = number_dist.to_mass(ρ).plot(ax=ax,mplkwds={'color':"red","ls":'--','label':'Truth'})

ax.legend()