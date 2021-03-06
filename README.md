# HelioSoil
A library for soiling analysis and mirror washing for Concentrating Solar Power (CSP) heliostats.

## Summary
This library provides tools developed for predicting soling reflectance losses for Solar Tower CSP plants using weather and plant design data. The deposition model has one free parameter (hrz0>1) which is the ratio of a reference height to the roughness length of the site. The value can either be assumed (e.g. expertise, literature) or (better) may estimated via some experimental procedure via the `fitting_experiment` class. In order to account for the effects of tracking on soiling, the solar field is divided up into a number of sectors and a single "representative" heliostat is used to represent the soiling status of the entire sector.

The details of the soiling model (including the sectorization and fitting procedure) can be found in [1-3] and a demo of soiling loss predictions can be found in `demo.ipynb`. The fitting of hrz0 using experimental data is demonstrated in `hrz0_fitting_demo.ipynb` using experimental data collected at the Queensland University of Technology (QUT), which are discussed in [1]. The data from these experiments are provided in the `data/qut_experiments`.

In addition to a soiling model, this library provides a basic economic and cleaning schedule modules to 1) understand the economic losses due to soiling given a certain number of cleaning crews, and 2) enable optimization of the cleaning trucks and washing frequency. A demonsration of this capability is available in `heuristic_optimization.ipynb` [^1] and discussion on the economic and cleaning models can be found in [3,4].

[^1]: In a later release, more sophisticated cleaning schedules are likely to be included.

## Input Files
The model takes as inputs three `.xlsx` files: one for the basic model parameters, one for the input data, and for one for solar field layout. One input document in `.epw` format is also required to define the climate and the geographical coordinates of the simulated plant.

The input data workbook can have the following sheets: 

* *Weather* which has columns with the following headers: 
    - Time (required), a datetime in `dd/mm/yyyy HH:MM` format.
    - AirTemp (required), a float of the air temperature. Units are C.
    - WindSpeed (required), a float of the wind speed. Units are m/s.
    - TSP/PMX (required), a float of dust concentration in air as TSP or PMX. Units are ??g/m^3.
    - DNI (optional), a float representing the Direct Normal Irradiation. Units are W/m^2.
    - RainIntensity (optional), a float representing the rain intensity. Units are mm/hr.

The following sheets are required for `fitting_experiment` class only:

* *Tilts* which has n+1 columns for n mirrors. The first column is `Time` in the same datetime format as the weather and columns 2 to n+1 are with headers `Mirror_x`, with `x=1,2,...,n` which contain the tilts in degrees. This sheet is required for the `fitting_experiment` class, since  the `field_model` class computes the tilts via the tracking requirement in the `helios_angles` method.

* *Reflectance_Average*. Columns are Time in datetime format (see above) followed by the average of reflectance measurements for each mirror at each time. This will be used in a later release to enable fitting of the hrz0 parameter via experiments.

* *Reflectance_Sigma*. Columns are Time in datetime format (see above) followed by the standard deviation of the reflectance measurements for each mirror at each time. This will be used in a later release to enable fitting of the hrz0 parameter via experiments.

Examples of the format of these sheets can be found in the `data` folder.

## External dependencies
The details of the required python packages can be found in the environment.yml file. Aside from these requirements, the average optical efficiencies of each sector are computed using [CoPylot](https://www.nrel.gov/docs/fy21osti/78774.pdf). To use CoPylot, install SolarPILOT following the instructions in Section 2.1 of [5] and place the files `copylot.py` and `solarpilot.dll` into the main directory.

## Assumptions and Notes
The module assumes a Solar Tower plant and that the dust size distribution is known (e.g. from literature [6]). This distribution is scaled according to the airborne dust concentration measurements in the input data. See [1] for details. 
	
Either a first- or second-surface geometry model is used to evaluate reflectance losses from the deposited dust. The second surface model is default and is likely closer to reality for most heliostats.

The current cleaning optimization is quite simple: a number of trucks clean the field a certain number of times annually. More sophisticated policies may be added in future releases (e.g. the mixed-integer linear program from [3] or the condition-based policies from [7,8]).

## Acknowledgements
<img style="float: left;background-color: white;margin-bottom:10px;margin-right:10px" src="doc/astri_logo.png" width="88" height="113">

This library was developed with the support of the Australian Government, through the Australian Renewable Energy Agency (ARENA) and within the framework of the Australian Solar Thermal Research Institute (ASTRI).

--------------------
## Suggested citation
If you find this library useful, consider citing the following papers which form the intellectual basis for the methods in this library. 

For the soiling model, please cite
### Plain text
  G. Picotti, P. Borghesani, G. Manzolini, M. E. Cholette, and R. Wang, ???Development and experimental validation of a physical model for the soiling of mirrors for CSP industry applications,??? Sol. Energy, vol. 173, pp. 1287???1305, 2018.

### BibTeX
  ~~~
  @article{PICOTTI20181287,
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
  G. Picotti, L. Moretti, M. E. Cholette, M. Binotti, R. Simonetti, E. Martelli, T. A. Steinberg, G. Manzolini,???Optimization of cleaning strategies for heliostat fields in solar tower plants,??? Sol. Energy, vol. 204, pp. 501???514, 2020.

### BibTeX
  ~~~
@article{PICOTTI2020501,
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
[1] G. Picotti, P. Borghesani, G. Manzolini, M. E. Cholette, and R. Wang, ???Development and experimental validation of a physical model for the soiling of mirrors for CSP industry applications,??? Sol. Energy, vol. 173, pp. 1287???1305, 2018. [link](https://eprints.qut.edu.au/123160/)

[2] G. Picotti, P. Borghesani, M. E. Cholette, and G. Manzolini, ???Soiling of solar collectors ??? Modelling approaches for airborne dust and its interactions with surfaces,??? Renew. Sustain. Energy Rev., vol. 81, pp. 2343???2357, Jan. 2018. [link](https://eprints.qut.edu.au/223121/)

[3] G. Picotti, L. Moretti, M. E. Cholette, M. Binotti, R. Simonetti, E. Martelli, T. A. Steinberg, G. Manzolini, ???Optimization of cleaning strategies for heliostat fields in solar tower plants,??? Sol. Energy, vol. 204, pp. 501???514, 2020 [link](https://eprints.qut.edu.au/201838/)

[4] G. Picotti, M. Binotti, M. E. Cholette, P. Borghesani, G. Manzolini, and T. Steinberg, ???Modelling the soiling of heliostats: Assessment of the optical efficiency and impact of cleaning operations,??? AIP Conf. Proc., vol. 2126, no. 1, p. 30043. [link](https://aip.scitation.org/doi/pdf/10.1063/1.5117555?class=pdf)

[5] W. T. Hamilton, M. J. Wagner, and A. J. Zolan, ???Demonstrating solarpilot???s Python Application Programmable Interface Through Heliostat Optimal Aimpoint Strategy Use Case,??? Journal of Solar Energy Engineering, vol. 144, no. 3, Mar. 2022, doi: 10.1115/1.4053973.

[6] J. H. Seinfeld and S. N. Pandis, Atmospheric chemistry and physics: from air pollution to climate change. New York: Wiley-Interscience, 1998.

[7] H. Truong-Ba, M. E. Cholette, G. Picotti, T. A. Steinberg, and G. Manzolini, ???Sectorial reflectance-based cleaning policy of heliostats for Solar Tower power plants,??? Renew. Energy, vol. 166, pp. 176???189, 2020. [link](https://eprints.qut.edu.au/207053/)

[8] H. Truong Ba, M. E. Cholette, R. Wang, P. Borghesani, L. Ma, and T. A. Steinberg, ???Optimal condition-based cleaning of solar power collectors,??? Solar Energy, vol. 157, pp. 762???777, Nov. 2017. [link](https://eprints.qut.edu.au/111078/)


