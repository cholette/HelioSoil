# %% Creating "reasonable" number per area distribution on a mirror
import numpy as np
import heliosoil.base_models as smb
import heliosoil.fitting as smf
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams["figure.figsize"] = (5, 5)
import scipy.stats as sps
import mirror_soiling_data as msd
from tqdm import tqdm

d = msd.get_dataset_path("qut")
parameter_file = d.joinpath("qut_parameters.xlsx")
files = msd.list_dataset_files("qut")

# %%
imodel = smf.SemiPhysical(parameter_file)
imodel.hrz0 = 5.6


dust = smb.Dust()
dust.D = [np.logspace(-3.0, 2, 100)]
dust.rho = [2000]  # Silica
dust.m = [1.52]
dust.Nd = [np.array([3000, 999.875, 0.125])]
dust.log10_mu = [np.log10([0.0117, 0.051231, 0.8226])]
dust.log10_sig = [np.log10([1.71061, 2.239, 2.512])]
dust.hamaker = None
dust.poisson = None
dust.youngs_modulus = None

dust._compute_distributions(f=0)
dust.plot_distributions()

# %% Deposition
_, _, _, _, vz = imodel.deposition_velocity(dust, f=0, wind_speed=np.array([5.0]), air_temp=np.array([25.0]))
vz = vz[:, 0]  # make 1D since there is only one time step here
pdfqN = dust.pdfN[0] * vz
pdfA = pdfqN * np.pi * dust.D[0] ** 2

# %%
fig, ax = plt.subplots()
ax.semilogx(dust.D[0], pdfqN)
ax.set_xlabel(r"Diameter [$\mu$m]")
ax.set_ylabel(r"$\frac{\#}{m^{2} \cdot s}$")

# %%
Δt = 10 * 3600.0
area = 0.01e-4  # m2
nm_per_pixel = 6500 / 60  # camera pitch 6.5 µm
side_length = np.sqrt(area)
pdfN_in_area = pdfqN * area * Δt
fig, ax = plt.subplots()
ax.semilogx(dust.D[0], pdfN_in_area)
ax.set_xlabel(r"Diameter [$\mu$m]")
ax.set_ylabel(rf"# in {area * 1e4:.0f} $cm^2$ of surface")

# %%
log_D = np.log10(dust.D[0])
d_log_D = np.diff(log_D, axis=0)[0]
# bin_edges = np.r_[log_D[0]-d_logD,log_D+d_logD]

pdf_avg = (pdfN_in_area[:-1] + pdfN_in_area[1:]) / 2.0
cumulative = np.concatenate([[0], np.cumsum(d_log_D * pdf_avg)])

# %%
fig, ax = plt.subplots()
ax.semilogx(dust.D[0], pdfqN)
ax2 = ax.twinx()
ax2.semilogx(dust.D[0], cumulative, "red")
ax2.tick_params(axis="y", colors="red")
ax2.yaxis.label.set_color("red")
ax2.spines["right"].set_color("red")

# %% Sample
rng = np.random.default_rng(seed=3)
centers = log_D[0:-1] + d_log_D
centers = centers.flatten()
λ = np.diff(cumulative)
N = sps.poisson.rvs(λ)
D_min = 0.1
mask = centers > np.log10(D_min)
N_above = N[mask]
centers_above = centers[mask]

fig, ax = plt.subplots()
ax.bar(centers.flatten(), height=N, width=d_log_D * 0.9)
ax.set_xticklabels([f"$10^{{{a:.0f}}}$" for a in ax.get_xticks()])
ax.set_xlabel("D[$\mu$m]")
ax.set_ylabel("# deposited")
ax.bar(centers_above, height=N_above, width=d_log_D * 0.9, color="red")

# Place particles
particle_list = {"D": [], "x": [], "y": []}
for ii, n in tqdm(enumerate(N_above)):
    pos = rng.uniform(-side_length / 2.0, side_length / 2.0, size=(n, 2))  # arbitrary bounds
    for jj in range(pos.shape[0]):
        particle_list["D"].append(float(10 ** centers_above[ii]))
        particle_list["x"].append(float(pos[jj, 0]))
        particle_list["y"].append(float(pos[jj, 1]))


# %%
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

diameters = particle_list["D"]
xs, ys = particle_list["x"], particle_list["y"]

fig, ax = plt.subplots(figsize=(10, 10))
patches = [mpatches.Circle((x, y), d / 2.0 * 1e-6) for x, y, d in zip(xs, ys, diameters)]
col = PatchCollection(patches, facecolors="black", edgecolors="none")
ax.add_collection(col)
ax.set_aspect("equal")
ax.autoscale()
fig.savefig("../results/fake_microsope.tiff", dpi=1000)

# %%
