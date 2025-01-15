# HelioSoil

A library for soiling analysis and mirror washing for Concentrating Solar Power (CSP) heliostats.

## Summary

This library provides tools developed for predicting soling reflectance losses for Solar Tower CSP plants using weather and plant design data. Two models are provided (and are detailed in [9]):

* A physics-based deposition. This model includes both full-field simulations (via the `base_models.field_model` class) and methods for fitting parameters based on experimental data via the `fitting.semi_physical` class.
* A constant-mean deposition velocity model. Currently, this model can be fit to experimental data via the `fitting.constant_mean_deposition_velocity` class , but full field simulation is not yet supported.

This version includes the following new features:

* A Mie scattering loss model to better represent the reflectance losses due to small particles
* Stochastic soiling model for the inclusion of uncertainty in the predictions

The details of the physics-based soiling model (including the sectorization and fitting procedure) can be found in [1-3] and a demo of soiling loss predictions can be found in `demo_hrz0_fitting.ipynb`. The fitting of hrz0 using experimental data is demonstrated in `demo_hrz0_fitting.ipynb` using experimental data collected at the Queensland University of Technology (QUT), which are discussed in [1]. The data from these experiments (and others) are provided in a separate [mirror_soiling_data](https://github.com/cholette/mirror_soiling_data) repository and should be placed in the a `data/` folder for the fitting demo to work.

In addition to a soiling model, this library provides a basic economic and cleaning schedule modules[^1] to 1) understand the economic losses due to soiling given a certain number of cleaning crews, and 2) enable optimization of the cleaning trucks and washing frequency. A demonstration  of this capability is available in `demo_cleaning_optimization.ipynb` and discussion on the economic and cleaning models can be found in [3,4].

[^1]: The optimization requires a full field simulation, so only the semi-physical model is suppored at this time. Later releases are expected to support the use of the constant-mean model and more sophisticated cleaning schedules.

## Input Files

The model takes as inputs three `.xlsx` files: one for the basic model parameters, one for the input data, and for one for solar field layout. One input document in `.epw` format is also required to define the climate and the geographical coordinates of the simulated plant. Examples of these sheets are provided in the `woomera_demo/` subfolder.

The input data workbook has the following sheets:

* *Dust* which has at least two columns with the Parameter name in the first column and the parameter value in the second column. The following parameters must be defined:
  * `D`, three element vector (separated  by semi-colons) describing the diameter grid for the dust size distribution
  * `N_size`, integer, number of modes in the airborne dust distribution
  * `N_d`, `N_size` element vector (separated  by semi-colons) describing the number parameters (heights) for each of the lognormal modes of the airborne dust distribution
  * `mu`, `N_size` element vector (separated  by semi-colons) describing the mean parameters for each of the lognormal modes of the airborne dust distribution
  * `sigma`, `N_size` element vector (separated  by semi-colons) describing the sigma parameters for each of the lognormal modes of the airborne dust distribution
  * `rho`, integer, dust density in kg/m^3
  * `hamaker_dust`, Hamaker constant of dust in J
  * `poisson`, Poisson ration of dust
  * `youngs_modulus_dust`, Young's modulus of dust N/m^2
  * `refractive_index_real_part`, Real part of complex refractive index
  * `refractive_index_imaginary_part`, Imaginary part of complex refractive index
* *Weather* which has columns with the following headers:
  * `Time` (required), a datetime in `dd/mm/yyyy HH:MM` format.
  * `AirTemp` (required), a float of the air temperature. Units are C.
  * `WindSpeed` (required), a float of the wind speed. Units are m/s.
  * `TSP/PMX` (required), a float of dust concentration in air as TSP or PMX. Units are µg/m^3.
  * `DNI` (optional), a float representing the Direct Normal Irradiation. Units are W/m^2.
  * `RainIntensity` (optional), a float representing the rain intensity. Units are mm/hr.

The `fitting.semi_physical` and `fitting.constant_mean_deposition_velocity` classes do not require the field layout file, but the following sheets must be present in addition to *Dust* and *Weather*:

* *Tilts* which has n+1 columns for n mirrors. The first column is `Time` in the same datetime format as the weather and columns 2 to n+1 are with headers `Mirror_x`, with `x=1,2,...,n` which contain the tilts in degrees.
* *Reflectance_Average*. Columns are Time in datetime format (see above) followed by the average of reflectance measurements for each mirror at each time. This will be used in a later release to enable fitting of the hrz0 parameter via experiments.
* *Reflectance_Sigma*. Columns are Time in datetime format (see above) followed by the standard deviation of the reflectance measurements for each mirror at each time. This will be used in a later release to enable fitting of the hrz0 parameter via experiments.

Examples of the format of these sheets can be found in the [mirror_soiling_data](https://github.com/cholette/mirror_soiling_data) repository.

For the fitting classes, if the `loss_model = 'mie'`, the *Source_Intensity* sheet needs to be defined, with two columns: the Wavelength in nm, and Source Intensity which can be unitless or in W/m^2/nm (it will be normalized to integrate to one internally when imported).

## External dependencies

The details of the required python packages can be found in the environment.yml file. Aside from these requirements, the average optical efficiencies of each sector are computed using [CoPylot](https://www.nrel.gov/docs/fy21osti/78774.pdf). To use CoPylot, install [SolarPILOT](https://www2.nrel.gov/csp/solarpilot) following the instructions in Section 2.1 of [5] and place the files `copylot.py` and `solarpilot.dll` into the main directory.

The data available from the [mirror_soiling_data](https://github.com/cholette/mirror_soiling_data) should be placed in the `data/` subfolder for the fitting scripts and notebooks to work.

## Assumptions and Notes

The field simulation class `base_models.field_model`  assumes a Solar Tower plant. For the semi-physical deposition model, the dust size distribution is known (e.g. from literature [6]). This distribution is scaled according to the airborne dust concentration measurements in the input data. See [1] for details.

Either a first- or second-surface geometry model is used to evaluate reflectance losses from the deposited dust. The second surface model is default and is likely closer to reality for most heliostats.

The current cleaning optimization is quite simple: a number of trucks clean the field a certain number of times annually. More sophisticated policies may be added in future releases (e.g. the mixed-integer linear program from [3] or the condition-based policies from [7,8]).

## Acknowledgements

<img style="float: left;background-color: white;margin-bottom:10px;margin-right:10px" src="docs/astri_logo.png" width="88" height="113">

This library was developed with the support of the Australian Government, through the Australian Renewable Energy Agency (ARENA) and within the framework of the Australian Solar Thermal Research Institute (ASTRI).

--------------------

## Suggested citation

If you find this library useful, consider citing the following papers which form the intellectual basis for the methods in this library.

For the soiling model, please cite
### Plain text

  G. Picotti, P. Borghesani, G. Manzolini, M. E. Cholette, and R. Wang, “Development and experimental validation of a physical model for the soiling of mirrors for CSP industry applications,” Sol. Energy, vol. 173, pp. 1287–1305, 2018.

### BibTeX

~~~bibtex
@article{picotti_2020_physical_model,
  title = {Development and experimental validation of a physical model for the soiling of mirrors for CSP industry applications},
  journal = {Solar Energy},
  volume = {173},
  pages = {1287-1305},
  year = {2018},
  issn = {0038-092X},
  doi = {https://doi.org/10.1016/j.solener.2018.08.066},
  author = {G. Picotti and P. Borghesani and G. Manzolini and M.E. Cholette and R. Wang}
  }
~~~

For the cleaning costs and optimization, please cite

### Plain text

  G. Picotti, L. Moretti, M. E. Cholette, M. Binotti, R. Simonetti, E. Martelli, T. A. Steinberg, G. Manzolini,“Optimization of cleaning strategies for heliostat fields in solar tower plants,” Sol. Energy, vol. 204, pp. 501–514, 2020.

### BibTeX

  ~~~bibtex
  @article{picotti_2020_cleaning,
  title = {Optimization of cleaning strategies for heliostat fields in solar tower plants},
  journal = {Solar Energy},
  volume = {204},
  pages = {501-514},
  year = {2020},
  issn = {0038-092X},
  doi = {https://doi.org/10.1016/j.solener.2020.04.032},
  author = {Giovanni Picotti and Luca Moretti and Michael E. Cholette and Marco Binotti and Riccardo Simonetti and Emanuele Martelli and Theodore A. Steinberg and Giampaolo Manzolini}
  }
  ~~~

## References

[1] G. Picotti, P. Borghesani, G. Manzolini, M. E. Cholette, and R. Wang, “Development and experimental validation of a physical model for the soiling of mirrors for CSP industry applications,” Sol. Energy, vol. 173, pp. 1287–1305, 2018. [link](https://eprints.qut.edu.au/123160/)

[2] G. Picotti, P. Borghesani, M. E. Cholette, and G. Manzolini, “Soiling of solar collectors – Modelling approaches for airborne dust and its interactions with surfaces,” Renew. Sustain. Energy Rev., vol. 81, pp. 2343–2357, Jan. 2018. [link](https://eprints.qut.edu.au/223121/)

[3] G. Picotti, L. Moretti, M. E. Cholette, M. Binotti, R. Simonetti, E. Martelli, T. A. Steinberg, G. Manzolini, “Optimization of cleaning strategies for heliostat fields in solar tower plants,” Sol. Energy, vol. 204, pp. 501–514, 2020 [link](https://eprints.qut.edu.au/201838/)

[4] G. Picotti, M. Binotti, M. E. Cholette, P. Borghesani, G. Manzolini, and T. Steinberg, “Modelling the soiling of heliostats: Assessment of the optical efficiency and impact of cleaning operations,” AIP Conf. Proc., vol. 2126, no. 1, p. 30043. [link](https://aip.scitation.org/doi/pdf/10.1063/1.5117555?class=pdf)

[5] W. T. Hamilton, M. J. Wagner, and A. J. Zolan, “Demonstrating solarpilot’s Python Application Programmable Interface Through Heliostat Optimal Aimpoint Strategy Use Case,” Journal of Solar Energy Engineering, vol. 144, no. 3, Mar. 2022, doi: 10.1115/1.4053973.

[6] J. H. Seinfeld and S. N. Pandis, Atmospheric chemistry and physics: from air pollution to climate change. New York: Wiley-Interscience, 1998.

[7] H. Truong-Ba, M. E. Cholette, G. Picotti, T. A. Steinberg, and G. Manzolini, “Sectorial reflectance-based cleaning policy of heliostats for Solar Tower power plants,” Renew. Energy, vol. 166, pp. 176–189, 2020. [link](https://eprints.qut.edu.au/207053/)

[8] H. Truong Ba, M. E. Cholette, R. Wang, P. Borghesani, L. Ma, and T. A. Steinberg, “Optimal condition-based cleaning of solar power collectors,” Solar Energy, vol. 157, pp. 762–777, Nov. 2017. [link](https://eprints.qut.edu.au/111078/)

[9] G. Picotti, M. E. Cholette, C. B. Anderson, T. A. Steinberg, and G. Manzolini, “Stochastic Soiling Loss Models for Heliostats in Concentrating Solar Power Plants.” arXiv, Apr. 24, 2023. doi: 10.48550/arXiv.2304.11814. [link](https://arxiv.org/abs/2304.11814)
