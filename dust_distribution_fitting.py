# %%
import numpy as np
import matplotlib.pyplot as plt
import heliosoil.dust_distributions as dd

# %% Create fake distribution

# %%%%%%%%%%%%%% Urban %%%%%%%%%%%%%%%%%%%%%
# N = np.array([3000,999.875,0.125])
# log10_μ = np.log10([0.0117,0.051231,0.8226])
# log10_σ = np.log10([1.71061,2.239,2.512])

N = np.array([6650, 147, 1990])
log10_μ = np.log10([0.015, 0.054, 0.084])
log10_σ = np.log10([1.6788, 3.6058, 1.8450])
params = np.r_[N, log10_μ, log10_σ]

# ax = plot_old(dist_old)
number_dist = dd.NumberDistribution.from_params(params)

# %% Sample
v = 1  # cm3
samples = number_dist.sample(v)

# %% plotting
fig, ax = plt.subplots()
ax.hist(samples, 10, density=True)
x = np.linspace(-3, 3, 1000)
ax.plot(x, number_dist.density(x))
ax.set_xticklabels([f"$10^{{{s:.0f}}}$" for s in ax.get_xticks()])
ax.set_xlabel("Diameter [$\mu$m]")
ax.set_ylabel(f"Number density [{number_dist.units}]")
plt.show()

# %%
