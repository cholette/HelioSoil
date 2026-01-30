import os
import sys
from pathlib import Path
import numpy as np
from numpy import radians as rad
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
import copy
import contextlib
from tqdm.auto import tqdm
from scipy.interpolate import RectBivariateSpline
from heliosoil.utilities import _print_if, get_project_root
from heliosoil.base_models import PhysicalBase, ConstantMeanBase, Sun, SimulationInputs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

FILE_COPYLOT = "copylot.py"
FILE_SOLARPILOTDLL = "solarpilot.dll"


@contextlib.contextmanager
def working_directory(path):
    """A context manager to temporarily change the working directory."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def _import_copylot():
    """
    Dynamically imports copylot from the project root and checks for dependencies.
    """
    project_root = get_project_root()
    copylot_path = project_root / FILE_COPYLOT
    dll_path = project_root / FILE_SOLARPILOTDLL

    # Check if both required files exist in the project root
    if not copylot_path.is_file() or not dll_path.is_file():
        raise FileNotFoundError(
            f"Could not find {FILE_COPYLOT} and/or {FILE_SOLARPILOTDLL} in the project root directory: {project_root}\n"
            "Please download these required files from the SolarPILOT GitHub repository "
            "(https://github.com/NREL/SolarPILOT) and place them in the HELIOSOIL/lib folder."
        )

    # Add project root to path to allow for dynamic import
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        import copylot as copylot

        return copylot, project_root
    except ImportError as e:
        # This might catch other import errors within copylot, so it's good to be specific
        raise ImportError(
            f"Successfully found 'copylot.py' but failed to import it. "
            f"Please ensure it and its dependencies are correctly set up. Original error: {e}"
        )


@dataclass
class ReceiverParameters:
    """Default parameters for central receiver configuration."""

    receiver_type: str = field(
        default="External cylindrical",
        metadata={"description": "Type of receiver (External cylindrical or Flat plate)"},
    )
    tower_height: float = field(
        default=120.0, metadata={"units": "m", "description": "Height of receiver tower"}
    )
    panel_height: float = field(
        default=30.5, metadata={"units": "m", "description": "Height of receiver panel"}
    )
    width_diameter: float = field(
        default=15.8, metadata={"units": "m", "description": "Width/diameter of receiver"}
    )
    orientation_elevation: Optional[float] = field(
        default=None,
        metadata={"units": "degrees", "description": "Elevation angle for flat plate receiver"},
    )
    thermal_losses: float = field(
        default=105,
        metadata={"units": "MW", "description": "Constant thermal losses from receiver"},
    )
    thermal_max: float = field(
        default=1000.0, metadata={"units": "MW", "description": "Maximum thermal power"}
    )
    thermal_min: float = field(
        default=210, metadata={"units": "MW", "description": "Minimum thermal power"}
    )


@dataclass
class PlantParameters:
    """Parameters for power plant configuration."""

    power_block_efficiency: float = field(
        default=0.42,
        metadata={"units": "fraction", "description": "Power block conversion efficiency"},
    )
    heliostat_aim_point_strategy: str = field(
        default="Image size priority", metadata={"description": "Heliostat aim point strategy"}
    )
    electricity_price: float = field(
        default=100.0, metadata={"units": "$/MWh", "description": "Price of electricity"}
    )
    plant_other_maintenance: float = field(
        default=0.0, metadata={"units": "$/MWh", "description": "Non-cleaning maintenance costs"}
    )


class CentralTowerPlant:
    """Central tower plant with parameter management."""

    def __init__(self):
        self._receiver = ReceiverParameters()
        self._plant = PlantParameters()

    @property
    def plant(self) -> dict:
        """Get plant parameters as dictionary for backward compatibility."""
        return {
            "power_block_efficiency": self._plant.power_block_efficiency,
            "aim_point_strategy": self._plant.heliostat_aim_point_strategy,
            "electricity_price": self._plant.electricity_price,
            "plant_other_maintenance": self._plant.plant_other_maintenance,
        }

    @property
    def receiver(self) -> dict:
        """Get receiver parameters as dictionary for backward compatibility."""
        return {
            "receiver_type": self._receiver.receiver_type,
            "tower_height": self._receiver.tower_height,
            "panel_height": self._receiver.panel_height,
            "width_diameter": self._receiver.width_diameter,
            "orientation_elevation": self._receiver.orientation_elevation,
            "thermal_losses": self._receiver.thermal_losses,
            "thermal_max": self._receiver.thermal_max,
            "thermal_min": self._receiver.thermal_min,
        }

    def import_plant(self, file_params: Union[str, Path]) -> None:
        """Load parameters from Excel file with validation."""
        table = pd.read_excel(file_params, index_col="Parameter")

        # Check required parameters
        required_params = [
            "receiver_type",
            "receiver_tower_height",
            "receiver_height",
            "receiver_thermal_losses",
            "minimum_receiver_power",
            "maximum_receiver_power",
            "power_block_efficiency",
            "heliostat_aim_point_strategy",
            "electricity_price",
            "plant_other_maintenance",
        ]

        # Handle width/diameter parameters
        width_params = ["receiver_width_diameter", "receiver_width", "receiver_diameter"]
        if not any(param in table.index for param in width_params):
            required_params.append("receiver_width_diameter")

        # Validate required parameters
        missing_params = [param for param in required_params if param not in table.index]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Update receiver parameters
        self._receiver.receiver_type = table.loc["receiver_type"].Value

        # Handle width/diameter options
        if "receiver_width_diameter" in table.index:
            self._receiver.width_diameter = float(table.loc["receiver_width_diameter"].Value)
        elif "receiver_width" in table.index:
            self._receiver.width_diameter = float(table.loc["receiver_width"].Value)
        elif "receiver_diameter" in table.index:
            self._receiver.width_diameter = float(table.loc["receiver_diameter"].Value)
        else:
            raise ValueError("Missing receiver width/diameter parameter")

        # Handle flat plate specific parameters
        if self._receiver.receiver_type == "Flat plate":
            if "receiver_orientation_elevation" not in table.index:
                raise ValueError("Missing receiver_orientation_elevation for Flat plate")
            self._receiver.orientation_elevation = float(
                table.loc["receiver_orientation_elevation"].Value
            )

        # Update remaining parameters
        self._receiver.tower_height = float(table.loc["receiver_tower_height"].Value)
        self._receiver.panel_height = float(table.loc["receiver_height"].Value)
        self._receiver.thermal_losses = float(table.loc["receiver_thermal_losses"].Value)
        self._receiver.thermal_min = float(table.loc["minimum_receiver_power"].Value)
        self._receiver.thermal_max = float(table.loc["maximum_receiver_power"].Value)

        # Update plant parameters
        self._plant.power_block_efficiency = float(table.loc["power_block_efficiency"].Value)
        self._plant.heliostat_aim_point_strategy = table.loc["heliostat_aim_point_strategy"].Value
        self._plant.electricity_price = float(table.loc["electricity_price"].Value)
        self._plant.plant_other_maintenance = float(table.loc["plant_other_maintenance"].Value)


class FieldCommonMethods:
    sun: Sun
    latitude: float
    longitude: float
    timezone_offset: float

    def sun_angles(self, simulation_inputs: SimulationInputs, verbose=True):
        sim_in = simulation_inputs

        _print_if(
            "Calculating sun apparent movement and angles for "
            + str(sim_in.N_simulations)
            + " simulations",
            verbose,
        )

        files = list(sim_in.time.keys())
        for f in files:
            self.sun.angles_and_clearsky_dni(
                self.latitude, self.longitude, sim_in.time[f], tz_offset=self.timezone_offset
            )

        # Check if DNI data exists in the simulation inputs
        if (
            f not in sim_in.dni
            or not isinstance(sim_in.dni[f], (list, np.ndarray))
            or len(sim_in.dni[f]) == 0
        ):
            print("No DNI in weather data file using clear sky DNI")
            sim_in.dni[f] = self.sun.DNI[f]

    def helios_angles(
        self, plant, verbose: bool = True, aoi_model: str = "second_surface", d: float = None
    ):
        sun = self.sun
        aoi_models = ["first_surface", "second_surface", "heimsath"]
        assert len(sun.elevation) > 0, "You need to call sun_angles() before helios_angles()"
        assert aoi_model in aoi_models, f"aoi_model model must be one of {aoi_models}"

        helios = self.helios
        files = list(sun.elevation.keys())
        N_sims = len(files)
        _print_if(
            "Calculating heliostat movement and angles for " + str(N_sims) + " simulations",
            verbose,
        )

        for f in files:
            stowangle = sun.stow_angle
            h_tower = plant.receiver["tower_height"]
            helios.dist = np.sqrt(
                helios.x**2 + helios.y**2
            )  # horizontal distance between mirror and tower
            helios.elevation_angle_to_tower = np.degrees(
                np.arctan((h_tower / helios.dist))
            )  # elevation angle from heliostats to tower

            T_m = np.array(
                [-helios.y, -helios.x, np.ones((len(helios.x))) * h_tower]
            )  # relative position of tower from mirror (left-handed ref.sys.)
            L_m = np.sqrt(np.sum(T_m**2, axis=0))  # distance mirror-tower [m]
            t_m = (
                T_m / L_m
            )  # unit vector in the direction of the tower (from mirror, left-handed ref.sys.)
            s_m = np.array(
                [
                    np.cos(rad(sun.elevation[f])) * np.cos(rad(sun.azimuth[f])),
                    np.cos(rad(sun.elevation[f])) * np.sin(rad(sun.azimuth[f])),
                    np.sin(rad(sun.elevation[f])),
                ]
            )  # unit vector of direction of the sun from mirror (left-handed)
            s_m = np.transpose(s_m)
            THETA_m = 0.5 * np.arccos(s_m.dot(t_m))
            THETA_m = np.transpose(
                THETA_m
            )  # incident angle (the angle a ray of sun makes with the normal to the surface of the mirrors) in radians
            helios.incidence_angle[f] = np.degrees(THETA_m)  # incident angle in degrees
            helios.incidence_angle[f][
                :, sun.elevation[f] <= stowangle
            ] = np.nan  # heliostats are stored vertically at night facing north

            # apply the formula (Guo et al.) to obtain the components of the normal for each mirror
            A_norm = np.zeros((len(helios.x), max(s_m.shape), min(s_m.shape)))
            B_norm = np.sin(THETA_m) / np.sin(2 * THETA_m)
            for ii in range(len(helios.x)):
                A_norm[ii, :, :] = s_m + t_m[:, ii]

            N = A_norm[:, :, 0] * B_norm  # north vector
            E = A_norm[:, :, 1] * B_norm  # east vector
            H = A_norm[:, :, 2] * B_norm  # height vector
            N[:, sun.elevation[f] <= stowangle] = 1  # heliostats are stored at night facing north
            E[:, sun.elevation[f] <= stowangle] = 0  # heliostats are stored at night facing north
            H[:, sun.elevation[f] <= stowangle] = 0  # heliostats are stored at night facing north
            # Nd = np.degrees(np.arctan2(E,N))
            # Ed = np.degrees(np.arctan2(N,E))
            # Hd = np.degrees(np.arctan2(H,np.sqrt(E**2+N**2)))

            helios.elevation[f] = np.degrees(
                np.arctan(H / (np.sqrt(N**2 + E**2)))
            )  # [deg] elevation angle of the heliostats
            helios.elevation[f][:, sun.elevation[f] <= stowangle] = (
                90 - helios.stow_tilt
            )  # heliostats are stored at stow_tilt at night facing north
            helios.tilt[f] = 90 - helios.elevation[f]  # [deg] tilt angle of the heliostats
            helios.azimuth[f] = np.degrees(
                np.arctan2(E, N)
            )  # [deg] azimuth angle of the heliostat

            if aoi_model.lower() == "first_surface":
                helios.inc_ref_factor[f] = (1 + np.sin(rad(helios.incidence_angle[f]))) / np.cos(
                    rad(helios.incidence_angle[f])
                )  # first surface
                helios.aoi_model = "first_surface"
                _print_if("First surface loss model", verbose)
            elif aoi_model.lower() == "second_surface":
                helios.inc_ref_factor[f] = 2 / np.cos(
                    rad(helios.incidence_angle[f])
                )  # second surface model
                helios.aoi_model = "second_surface"
                _print_if("Second surface loss model", verbose)
            elif aoi_model.lower() == "heimsath":
                assert d is not None, "For aoi_model==heimsath, you must provide a value for d."
                _print_if("Heimsath loss model", verbose)
                helios.aoi_model = "heimsath"
                helios.inc_ref_factor[f] = None
                helios.aoi_parameter[f] = d

        self.helios = helios

    def reflectance_loss(self, simulation_inputs, cleans, verbose=True):
        """
        Calculates the reflectance losses with cleaning for the given simulation inputs and cleaning schedule.

        This method computes the hourly soiling factor for each heliostat in the solar field, taking into account the cleaning schedule provided. It accumulates the soiling between cleaning events and applies the appropriate reflectance loss factor based on the incidence angle.

        Args:
            simulation_inputs (object): An object containing the simulation inputs, including the time series data.
            cleans (numpy.ndarray): A 2D array indicating the cleaning schedule for each heliostat, where each row represents a heliostat and each column represents a day.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            None
        """
        sim_in = simulation_inputs
        N_sims = sim_in.N_simulations
        _print_if(
            "Calculating reflectance losses with cleaning for " + str(N_sims) + " simulations",
            verbose,
        )

        helios = self.helios
        n_helios = helios.x.shape[0]

        files = list(sim_in.time.keys())
        for fi in range(len(files)):
            f = files[fi]
            n_hours = int(helios.delta_soiled_area[f].shape[1])

            # accumulate soiling between cleans
            temp_soil = np.zeros((n_helios, n_hours))
            temp_soil2 = np.zeros((n_helios, n_hours))
            for hh in range(n_helios):
                sra = copy.deepcopy(
                    helios.delta_soiled_area[f][hh, :]
                )  # use copy.deepcopy otherwise when modifying sra to compute temp_soil2, also helios.delta_soiled_area is modified
                clean_idx = np.where(cleans[fi][hh, :])[0]
                # clean_at_0 = True  # kept true if sector hh-th is cleaned on day 0
                if len(clean_idx) > 0 and clean_idx[0] != 0:
                    clean_idx = np.insert(
                        clean_idx, 0, 0
                    )  # insert clean_idx = 0 to compute soiling since the beginning
                    # clean_at_0 = False  # true only when sector hh-th is cleaned on day 0
                if len(clean_idx) == 0 or clean_idx[-1] != (sra.shape[0]):
                    clean_idx = np.append(
                        clean_idx, sra.shape[0]
                    )  # append clean_idx = 8760 to compute soiling until the end

                clean_idx_n = np.arange(len(clean_idx))
                for cc in clean_idx_n[:-1]:
                    temp_soil[hh, clean_idx[cc] : clean_idx[cc + 1]] = np.cumsum(
                        sra[clean_idx[cc] : clean_idx[cc + 1]]
                    )  # Note: clean_idx = 8760 would be outside sra, but Python interprets it as "till the end"

                # Run again with initial condition equal to final soiling to obtain an approximation of "steady-state" soiling
                # if clean_at_0:
                #     temp_soil2[hh,:] = temp_soil[hh,:]
                # else:
                sra[0] = temp_soil[hh, -1]
                for cc in clean_idx_n[:-1]:
                    temp_soil2[hh, clean_idx[cc] : clean_idx[cc + 1]] = np.cumsum(
                        sra[clean_idx[cc] : clean_idx[cc + 1]]
                    )

            self.apply_aoi_model(temp_soil2, f)

        self.helios = helios
        area_loss = temp_soil2

        return area_loss

    def apply_aoi_model(self,nn_area_loss,f:int,inplace:bool=True):
        helios = self.helios
        if helios.aoi_model == "heimsath":
            ξ = 1.0 - 2.0 * nn_area_loss
            d = self.helios.aoi_parameter[f]
            φ = np.deg2rad(helios.incidence_angle[f])
            sf = ξ ** (np.cos(φ) ** (-d))
        elif helios.aoi_model in ["first_surface", "second_surface"]:
            sf = (
                1 - nn_area_loss * helios.inc_ref_factor[f]
            )  # hourly soiling factor for each sector of the solar field
        else:
            raise ValueError("helios.aoi_model not recognized. ")
        
        if inplace:
            helios.soiling_factor[f] = sf
        else:
            return helios.soiling_factor[f]

    def optical_efficiency(
        self, plant, simulation_inputs, climate_file, verbose=True, n_az=10, n_el=10
    ):
        """
        Computes the optical efficiency of a heliostat field for a given set of simulation inputs and climate data.

        This method sets up the simulation parameters in the CoPylot library, including the heliostat field layout,
        receiver properties, and simulation settings. It then computes the optical efficiency of each heliostat in
        the field for a grid of solar azimuth and elevation angles, and uses this lookup table to compute the time
        series of optical efficiency for each heliostat during the simulation.

        Args:
            plant (object): A plant object containing information about the receiver and aim point strategy.
            simulation_inputs (object): An object containing the simulation inputs, including the time series data.
            climate_file (str): The path to the climate data file (in .epw format).
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            n_az (int, optional): The number of azimuth angles to use in the lookup table. Defaults to 10.
            n_el (int, optional): The number of elevation angles to use in the lookup table. Defaults to 10.

        Returns:
            None
        """
        helios = self.helios
        sim_in = simulation_inputs
        sun = self.sun

        # check that lat/lon and timezone correspond to the same location
        if climate_file.split(".")[-1] == "epw":
            with open(climate_file) as f:
                line = f.readline()
                line = line.split(",")
                lat = line[6]
                lon = line[7]
                tz = line[-2]
        elif climate_file.split(".")[-1] == "csv":
            print('Processing weather data as a TMY .csv file...')
            with open(climate_file) as f:
                headers = f.readline()
                headers = [h.lower() for h in headers.split(',')]
                line = f.readline()
                line = line.split(",")

                lat = line[headers.index('latitude')]
                lon = line[headers.index('longitude')]
                tz = line[headers.index('time zone')]
        else:
            raise ValueError("Climate file type must be .epw or a TMY .csv")

        # check that parameter file and climate file have same location and timezone
        if self.latitude != float(lat) or self.longitude != float(lon):
            raise ValueError("Location of field_model and climate file do not match")
        if self.timezone_offset != float(tz):
            raise ValueError("Timezone offset of field_model and climate file do not match")
        copylot, project_root = _import_copylot()

        with working_directory(project_root):
            cp = copylot.CoPylot()
        r = cp.data_create()
        assert cp.data_set_string(
            r,
            "ambient.0.weather_file",
            climate_file,
        )

        # layout setup
        assert cp.data_set_number(r, "heliostat.0.height", helios.height)
        assert cp.data_set_number(r, "heliostat.0.width", helios.width)
        assert cp.data_set_number(r, "heliostat.0.soiling", 1)  # Sets cleanliness to 100%
        assert cp.data_set_number(r, "heliostat.0.reflectivity", 1)
        try:
            assert cp.data_set_string(r, "receiver.0.rec_type", plant.receiver["receiver_type"])
        except Exception:
            assert cp.data_set_string(r, "receiver.0.rec_type", "External cylindrical")
        if plant.receiver["receiver_type"] == "External cylindrical":
            assert cp.data_set_number(
                r, "receiver.0.rec_diameter", plant.receiver["width_diameter"]
            )
        elif plant.receiver["receiver_type"] == "Flat plate":
            assert cp.data_set_number(r, "receiver.0.rec_width", plant.receiver["width_diameter"])
            try:
                assert cp.data_set_number(
                    r, "receiver.0.rec_elevation", plant.receiver["orientation_elevation"]
                )
            except Exception:
                assert cp.data_set_number(r, "receiver.0.rec_elevation", -35)
        assert cp.data_set_number(r, "receiver.0.rec_diameter", plant.receiver["width_diameter"])
        assert cp.data_set_number(r, "receiver.0.rec_height", plant.receiver["panel_height"])
        assert cp.data_set_number(r, "receiver.0.optical_height", plant.receiver["tower_height"])

        # field setup
        ff = helios.full_field
        N_helios = ff["id"].shape[0]
        zz = [0 for ii in range(N_helios)]
        layout = [
            [0, ff["x"][ii], ff["y"][ii], zz[ii]] for ii in range(N_helios)
        ]  # [list(id),list(ff['x']),list(ff['y']),list(zz)]
        assert cp.assign_layout(r, layout)
        field = cp.get_layout_info(r)

        # simulation parameters
        assert cp.data_set_number(
            r, "fluxsim.0.flux_time_type", 0
        )  # 1 for time simulation, 0 for solar angles
        assert cp.data_set_number(
            r, "fluxsim.0.flux_dni", 1000.0
        )  # set the simulation DNI to 1000 W/m2. Only used to display indicative receiver power.
        assert cp.data_set_string(r, "fluxsim.0.aim_method", plant.plant["aim_point_strategy"])

        files = list(sim_in.time.keys())
        Ns = len(helios.x)
        helios.optical_efficiency = {f: [] for f in files}
        az_grid = np.linspace(0, 360, num=n_az)
        el_grid = np.linspace(sun.stow_angle, 90, num=n_el)
        eff_grid = np.zeros((Ns, n_az, n_el))
        # rec_power_grid = np.zeros((n_az, n_el))
        sec_ids = ff["sector_id"][field["id"].values.astype(int)]  # CoPylot re-orders sectors
        # fmt = "Getting efficiencies for az={0:.3f}, el={1:.3f}"

        # buliding the lookup table for grid of solar angles
        total_iterations = len(az_grid) * len(el_grid)
        progress_bar = tqdm(
            total=total_iterations, desc="Computing optical efficiency grid", leave=True
        )

        for ii in range(len(az_grid)):
            for jj in range(len(el_grid)):
                assert cp.data_set_number(r, "fluxsim.0.flux_solar_az_in", az_grid[ii])
                assert cp.data_set_number(r, "fluxsim.0.flux_solar_el_in", el_grid[jj])
                assert cp.simulate(r)
                dat = cp.detail_results(r)
                dat_summary = cp.summary_results(r)

                effs = dat["efficiency"]
                if dat_summary["Power absorbed by the receiver"] == " -nan(ind)":
                    raise ValueError(
                        "SolarPILOT unable to simulate with current parameter configuration"
                    )
                else:
                    # Update progress bar description with current power
                    current_power = dat_summary["Power absorbed by the receiver"]
                    progress_bar.set_description(
                        f"Computing grid - Power: {current_power:.2e} kW (az={az_grid[ii]:.1f}°, el={el_grid[jj]:.1f}°)"
                    )

                for kk in range(Ns):
                    idx = np.where(sec_ids == kk)[0]
                    eff_grid[kk, ii, jj] = effs[idx].mean()

                progress_bar.update(1)

        progress_bar.close()

        # Apply lookup table to simulation
        for f in files:
            T = len(sim_in.time[f])
            # Use above lookup to compute helios.optical_efficiency[f]
            _print_if("Computing optical efficiency time series for file " + str(f), verbose)
            helios.optical_efficiency[f] = np.zeros((Ns, T))
            for ll in range(Ns):
                opt_fun = RectBivariateSpline(el_grid, az_grid, eff_grid[ll, :, :].T, kx=1, ky=1)
                helios.optical_efficiency[f][ll, :] = np.array(
                    [opt_fun(sun.elevation[f][tt], sun.azimuth[f][tt])[0, 0] for tt in range(T)]
                )
            _print_if("Done!", verbose)
        self.helios = helios

    def plot_soiling_factor(self,file:int=None,hour:int=None):
        Ns = self.helios.x.shape[0]  # Number of sectors
        sid = self.helios.full_field["sector_id"]
        v = self.helios.soiling_factor[file][:,hour]
        fig,ax = plt.subplots()
        norm = Normalize(vmin=np.nanmin(v), vmax=1.0)
        sf = np.ones_like(self.helios.full_field["x"])
        for ii in range(Ns):
            mask = (sid == ii)
            xx,yy = self.helios.full_field["x"][mask],self.helios.full_field["y"][mask]
            sf[mask] = v[ii]
            sc = ax.scatter( xx,yy,c=v[ii]*np.ones_like(xx),cmap=plt.get_cmap('viridis'), norm=norm)

        fig.colorbar(sc, ax=ax, label='Soiling Factor')

        # Add plot styling
        ax.set_xlabel("Distance from receiver - X [m]")
        ax.set_ylabel("Distance from receiver - Y [m]")
        ax.set_title("Solar Field Soiling Factor Snapshot")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_aspect("equal")
        plt.tight_layout()

        dat = self.helios.full_field.copy()
        dat['soiling_factor'] = sf
        return fig,ax,dat

class FieldModel(PhysicalBase, FieldCommonMethods):
    def __init__(
        self,
        file_params,
        file_SF,
        cleaning_rate: Optional[float] = None,
        num_sectors: Optional[Union[int, Tuple[int, int], str]] = None,
    ):
        super().__init__()
        super().import_site_data_and_constants(file_params)

        self.sun = Sun()
        self.sun.import_sun(file_params)

        self.helios.import_helios(
            file_params, file_SF, cleaning_rate=cleaning_rate, num_sectors=num_sectors
        )
        if not (isinstance(self.helios.stow_tilt, float)) and not (
            isinstance(self.helios.stow_tilt, int)
        ):
            self.helios.stow_tilt = None

    def compute_acceptance_angles(self, plant, verbose=True):
        # Approximate acceptance half-angles using simple geometric approximation from:
        #    Sutter, Montecchi, von Dahlen, Fernández-García, and M. Röger,
        #    “The effect of incidence angle on the reflectance of solar mirrors,”
        #     Solar Energy Materials and Solar Cells, vol. 176, pp. 119–133,
        #     Mar. 2018, doi: 10.1016/j.solmat.2017.11.029.
        # This approach neglects blocking, shading, and the use of a single acceptance angle assumes that
        # the acceptance zone is conical. Tower height is presumed to be to the center.

        tower_height = plant.receiver["tower_height"]
        panel_height = plant.receiver["panel_height"]
        files = list(self.helios.tilt.keys())
        n_helios, n_times = self.helios.tilt[0].shape
        self.helios.acceptance_angles = {f: np.zeros((n_helios,)) for f in files}
        for f in files:
            for ii, h in enumerate(zip(self.helios.x, self.helios.y)):
                d = np.sqrt(h[0] ** 2 + h[1] ** 2)
                h1 = tower_height  # middle of panel
                h2 = tower_height + panel_height / 2.0  # top of panel
                a1 = np.arctan(h1 / d)
                a2 = np.arctan(h2 / d)
                self.helios.acceptance_angles[f][ii] = a2 - a1  # [rad]

        max_accept = max([self.helios.acceptance_angles[f].max() for f in files])
        min_accept = min([self.helios.acceptance_angles[f].min() for f in files])
        _print_if(
            f"Acceptance angle range: ({min_accept * 1e3:.1f}, {max_accept * 1e3:.1f}) [mrad]",
            verbose,
        )


class SimplifiedFieldModel(ConstantMeanBase, FieldCommonMethods):
    def __init__(
        self,
        file_params,
        file_SF,
        cleaning_rate: float = None,
        num_sectors: Optional[Union[int, Tuple[int, int], str]] = None,
    ):
        super().__init__()
        super().import_site_data_and_constants(file_params)

        self.sun = Sun()
        self.sun.import_sun(file_params)

        self.helios.import_helios(
            file_params, file_SF, cleaning_rate=cleaning_rate, num_sectors=num_sectors
        )
        if not (isinstance(self.helios.stow_tilt, float)) and not (
            isinstance(self.helios.stow_tilt, int)
        ):
            self.helios.stow_tilt = None
