"""
HelioSoil - Soiling Model and Cleaning Optimization for Solar Power Plants
"""
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
    ConstantMeanBase
)

# From field_models
from .field_models import (
    CentralTowerPlant,
    FieldModel,
    SimplifiedFieldModel,
    ReceiverParameters,
    PlantParameters
)

# From fitting
from .fitting import (
    SemiPhysical,
    ConstantMeanDeposition
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
    plot_soiled_optical_efficiency
)

# From utilities
from .utilities import (
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
    _import_option_helper
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
    "__version__"
]