"""
HelioSoil - Soiling Model and Cleaning Optimization for Solar Power Plants
"""

from importlib import metadata

try:
    # Read the version from the installed package's metadata
    __version__ = metadata.version("heliosoil")
except metadata.PackageNotFoundError:
    # Fallback for when the package is not installed, e.g., during development
    __version__ = "0.3"

# Import key classes and functions to make them available at the package level
# From base_models
from .base_models import (
    SimulationInputs,
    Dust,
    Truck,
    Sun,
    Heliostats,
    ReflectanceMeasurements,
    Constants,
    TruckParameters,
    SoilingBase,
    PhysicalBase,
    ConstantMeanBase,
)

# From field_models
from .field_models import CentralTowerPlant, FieldModel, SimplifiedFieldModel, ReceiverParameters, PlantParameters

# From fitting
from .fitting import SemiPhysical, ConstantMeanDeposition

# From horizontal_impaction
from .horizontal_impaction import (
    ConstantMeanWindBase,
    ConstantMeanWindDeposition,
    wind_projection_factors,
    wind_tangential_factor,
    parse_orientation_names,
)

# From cleaning_optimization
from .cleaning_optimization import (
    OptimizationProblem,
    optimize_periodic_schedule,
    periodic_schedule_tcc,
    optimize_rollout_schedule,
    rollout_heuristic_tcc,
    plot_optimization_results,
    plot_soiling_factor,
    plot_cleaning_schedule,
    plot_soiled_optical_efficiency,
)

# From utilities
from .utilities import (
    get_project_root,
    simple_annual_cleaning_schedule,
    plot_experiment_data,
    trim_experiment_data,
    daily_average,
    sample_simulation_inputs,
    get_training_data,
    default_training_mirrors,
    wind_rose,
    soiling_rates_summary,
    loss_table_from_sim,
    loss_hel_table_from_sim,
    logger,
    configure_logging,
    _print_if,
    _std_errors_from_cov,
    _ensure_list,
    _check_keys,
    _import_option_helper,
    cardinal_to_azimuth,
)

from .paper_specific_utilities import (
    plot_for_paper,
    plot_for_heliostats,
    soiling_rate,
    daily_soiling_rate,
    fit_quality_panels,
    fit_quality_plots,
    save_fit_quality_figures,
    summarize_fit_quality,
    regression_performance_stats,
    daily_rate_residuals,
    daily_soiling_tilt_all_data,
    plot_experiment_PA,
)

from .dust_distributions import DustDistribution

__all__ = [
    # base_models
    "SimulationInputs",
    "Dust",
    "Truck",
    "Sun",
    "Heliostats",
    "ReflectanceMeasurements",
    "Constants",
    "TruckParameters",
    "SoilingBase",
    "PhysicalBase",
    "ConstantMeanBase",
    # field_models
    "CentralTowerPlant",
    "FieldModel",
    "SimplifiedFieldModel",
    "ReceiverParameters",
    "PlantParameters",
    # fitting
    "SemiPhysical",
    "ConstantMeanDeposition",
    # horizontal_impaction
    "ConstantMeanWindBase",
    "ConstantMeanWindDeposition",
    "wind_projection_factors",
    "wind_tangential_factor",
    "parse_orientation_names",
    # cleaning_optimization
    "OptimizationProblem",
    "optimize_periodic_schedule",
    "periodic_schedule_tcc",
    "optimize_rollout_schedule",
    "rollout_heuristic_tcc",
    "plot_optimization_results",
    "plot_soiling_factor",
    "plot_cleaning_schedule",
    "plot_soiled_optical_efficiency",
    # utilities
    "get_project_root",
    "simple_annual_cleaning_schedule",
    "plot_experiment_data",
    "trim_experiment_data",
    "daily_average",
    "sample_simulation_inputs",
    "get_training_data",
    "default_training_mirrors",
    "wind_rose",
    "soiling_rates_summary",
    "loss_table_from_sim",
    "loss_hel_table_from_sim",
    "DustDistribution",
    "logger",
    "configure_logging",
    "_print_if",
    "_std_errors_from_cov",
    "_ensure_list",
    "_check_keys",
    "_import_option_helper",
    "cardinal_to_azimuth",
    # Version
    "__version__",
    # paper_specific_utilities
    "plot_for_paper",
    "plot_for_heliostats",
    "soiling_rate",
    "daily_soiling_rate",
    "fit_quality_panels",
    "fit_quality_plots",
    "save_fit_quality_figures",
    "summarize_fit_quality",
    "regression_performance_stats",
    "daily_rate_residuals",
    "daily_soiling_tilt_all_data",
    "plot_experiment_PA",
]
