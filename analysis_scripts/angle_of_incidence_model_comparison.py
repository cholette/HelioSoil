# %% Analysis of geometry reflectance model and comparison with models from [1] and [2]
# 
# References
# [1] A. Heimsath and P. Nitz, “The effect of soiling on the reflectance of solar reflector materials - Model for prediction of incidence angle dependent reflectance and attenuation due to dust deposition,” Solar Energy Materials and Solar Cells, vol. 195, pp. 258–268, Jun. 2019, doi: 10.1016/j.solmat.2019.03.015.
# [2] A. Heimsath et al., “Specular reflectance of soiled glass mirrors – Study on the impact of incidence angles,” AIP Conference Proceedings, vol. 1734, no. 1, p. 130009, May 2016, doi: 10.1063/1.4949219.


# %%
import numpy as np
import matplotlib.pyplot as plt

normal_cleanliness = [0.95,0.90,0.85]
d = [1.0, 0.9, 0.5]
a = [] # [16.78] # reverse engineering from lab soiling results in [2]
θ =  np.arange(5.0,50,0.1)
cosd = lambda x: np.cos(np.deg2rad(x))

fig,ax = plt.subplots(nrows=len(normal_cleanliness),
                      figsize=(5,8/3.0*len(normal_cleanliness))
                        )
for ii,nc in enumerate(normal_cleanliness):
    for jj,dd in enumerate(d):
        μ = 1.0 * (-np.log(nc) )**(1/dd) / 2
        heimsath = np.exp(-(2.0*μ/cosd(θ))**dd) # from [1]
        ax[ii].plot(θ,heimsath,label=f"Heimsath et al. 2019, d={dd}")
    
    b = (1 - np.sqrt(nc)) * cosd(0)
    heimsath_2 = (1-b/cosd(θ))**2           # Eq. (2) in [2]
    # ax[ii].plot(θ,heimsath_2,label=f"Heimsath  et al. 2016, Eq. (2)")


    for _,aa in enumerate(a):
        heimsath_3 = heimsath_2 + (nc-heimsath_2)/aa/cosd(θ) # Eq. (3) in [2]
        ax[ii].plot(θ,heimsath_3,label=f"Heimsath  et al. 2016, Eq. (3) in [2] , a ={a}")
    
    AR = (1 - nc) / 2
    ax[ii].plot(θ,(1-2*AR/cosd(θ)),label="Eq. (1)")

    AR = (1 - np.sqrt(nc))
    # ax[ii].plot(θ,(1-AR/cosd(θ))**2,label="Blocking and shading")
    ax[ii].set_title(f"Normal incidence cleanliness = {nc*100:.0f}%")
    ax[ii].legend()


ax[-1].set_xlabel(r'$\theta$ [deg]')
plt.tight_layout()                
plt.show()



# %%
