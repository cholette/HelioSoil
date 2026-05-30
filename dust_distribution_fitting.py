# %% 
import numpy as np
import matplotlib.pyplot as plt
import heliosoil.dust_distributions_OLD as dd_OLD
import heliosoil.dust_distributions as dd
import cmdstanpy

# %% Create fake distribution

# %%%%%%%%%%%%%% Urban %%%%%%%%%%%%%%%%%%%%%
# N = np.array([3000,999.875,0.125])
# log10_μ = np.log10([0.0117,0.051231,0.8226])
# log10_σ = np.log10([1.71061,2.239,2.512])

N = np.array([6650,147,1990])
log10_μ = np.log10([0.015,0.054,0.084])
log10_σ = np.log10([1.6788,3.6058,1.8450])
params = np.r_[N,log10_μ ,log10_σ]

# dist_old = dd_OLD.DustDistribution(params,'number')

# def plot_old(do):
#     fig,ax = plt.subplots()
#     x = np.linspace(-3,3,1000)
#     ax.semilogx(10**x,do.pdf(x),label='Old')
#     return ax

# ax = plot_old(dist_old)
number_dist = dd.NumberDistribution.from_params(params)
# number_dist.plot(ax=ax,mplkwds={'label':'New','ls':'--'})
# ax.legend()


# dist_old.convert_to_mass(rho=2000.0)
# ax = plot_old(dist_old)
# mass_dist = number_dist.to_mass(rho=2.0)
# mass_dist.plot(ax=ax)
# ax.legend()

# %% Sample

v = 36 # cm3
samples = number_dist.sample(v)

# %% plotting
fig,ax = plt.subplots()
ax.hist(samples,100,density=True)
x = np.linspace(-3,3,1000)
ax.plot(x,number_dist.density(x))
plt.show()

# %%
