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
from .field_models import (
    CentralTowerPlant,
    FieldModel,
    SimplifiedFieldModel,
    ReceiverParameters,
    PlantParameters,
)

# From fitting
from .fitting import SemiPhysical, ConstantMeanDeposition

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
    wind_rose,
    soiling_rates_summary,
    loss_table_from_sim,
    loss_hel_table_from_sim,
    DustDistribution,
    _print_if,
    _ensure_list,
    _check_keys,
    _import_option_helper,
)

from .paper_specific_utilities import (
    plot_for_paper,
    plot_for_heliostats,
    soiling_rate,
    daily_soiling_rate,
    fit_quality_plots,
    summarize_fit_quality,
    daily_soiling_tilt_all_data,
    plot_experiment_PA,
)

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
    "wind_rose",
    "soiling_rates_summary",
    "loss_table_from_sim",
    "loss_hel_table_from_sim",
    "DustDistribution",
    "_print_if",
    "_ensure_list",
    "_check_keys",
    "_import_option_helper",
    # Version
    "__version__",
    # paper_specific_utilities
    "plot_for_paper",
    "plot_for_heliostats",
    "soiling_rate",
    "daily_soiling_rate",
    "fit_quality_plots",
    "summarize_fit_quality",
    "daily_soiling_tilt_all_data",
    "plot_experiment_PA",
]
