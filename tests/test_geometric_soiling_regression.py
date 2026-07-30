r"""
Geometric-model regression test for the Woomera demo soiling factor.

Runs the full ``FieldModel`` pipeline on ``examples/woomera_demo`` with the
*geometric* loss model (extinction weights = 1). This skips the slow Mie
extinction computation and is deterministic to machine precision (numpy +
pysolar + scipy only -- no numba, no miepython), so the result is reproducible
across platforms and can be pinned tightly with no per-environment caveat.

A downsampled, human-readable slice of the resulting hourly soiling factor is
compared against a committed CSV baseline, so unintended changes to the model
output are caught. Only the soiling factor is checked; ``optical_efficiency``
(SolarPILOT) is not exercised.

------------------------------------------------------------------------------
Generating / updating the baseline
------------------------------------------------------------------------------
The baseline is not written on a normal run. To create or deliberately
regenerate it (after an *intended* model change), set the
HELIOSOIL_UPDATE_REFERENCES environment variable, then run pytest:

    PowerShell:  $env:HELIOSOIL_UPDATE_REFERENCES=1
                 pytest tests/test_geometric_soiling_regression.py
                 Remove-Item Env:\HELIOSOIL_UPDATE_REFERENCES
    cmd.exe:     set HELIOSOIL_UPDATE_REFERENCES=1 && pytest tests/test_geometric_soiling_regression.py
    bash:        HELIOSOIL_UPDATE_REFERENCES=1 pytest tests/test_geometric_soiling_regression.py

This writes ``tests/reference_data/woomera_geometric_soiling_factor.csv`` and a
``.meta.json`` provenance sidecar, then skips. Inspect the diff, commit both
files, and re-run without the flag to verify.
"""

import os
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import heliosoil
import heliosoil.base_models as smb
import heliosoil.field_models as smf
import heliosoil.utilities as smu
from heliosoil.utilities import get_project_root


# --- configuration that defines the reference run (matches the demo) ---------
N_TRUCKS = 4
N_CLEANS = 10
CLEANING_RATE = 330
DUST_TYPE = "PM10"
FILE = 0

# Time axis is kept at full hourly resolution; only the sectors are downsampled
# to keep the CSV narrow. N_SECTOR_SAMPLES (and the CSV float precision in
# _write_reference) are the knobs for baseline size.
N_SECTOR_SAMPLES = 6

REF_DIR = Path(__file__).parent / "reference_data"
REF_CSV = REF_DIR / "woomera_geometric_soiling_factor.csv"
REF_META = REF_DIR / "woomera_geometric_soiling_factor.meta.json"

_TRUE = {"1", "true", "True", "yes", "on"}
UPDATE = os.environ.get("HELIOSOIL_UPDATE_REFERENCES", "") in _TRUE

RTOL = 1e-10
ATOL = 1e-12


def _demo_paths():
    d = get_project_root() / "examples" / "woomera_demo"
    return (str(d / "parameters.xlsx"), str(d / "SF_woomera_SolarPILOT.csv"), str(d / "woomera_data.xlsx"))


def _run_geometric_soiling_factor():
    """Full FieldModel pipeline with the geometric loss model; returns soiling factor."""
    file_params, file_SF, file_weather = _demo_paths()

    model = smf.FieldModel(file_params, file_SF, cleaning_rate=CLEANING_RATE)
    sim = smb.SimulationInputs(files=[file_weather], dust_type=[DUST_TYPE])
    plant = smf.CentralTowerPlant()
    plant.import_plant(file_params)

    model.sun_angles(sim, verbose=False)
    model.helios_angles(plant, aoi_model="second_surface", verbose=False)
    # "geometry" -> unity extinction weights: no Mie, no acceptance angles needed.
    model.helios.compute_extinction_weights(sim, "geometry", verbose=False)
    model.deposition_flux(sim, verbose=False)
    model.adhesion_removal(sim, verbose=False)
    model.calculate_delta_soiled_area(sim, verbose=False)

    cleans = smu.simple_annual_cleaning_schedule(model.helios.tilt[FILE].shape[0], N_TRUCKS, N_CLEANS, dt=sim.dt[FILE] / 3600.0)
    model.reflectance_loss(sim, {FILE: cleans}, verbose=False)
    return np.asarray(model.helios.soiling_factor[FILE], dtype=np.float64)


def _sample_indices(n_sectors, n_hours):
    """Full hourly time axis; evenly spaced sector subsample."""
    t_idx = np.arange(n_hours)
    s_idx = np.unique(np.linspace(0, n_sectors - 1, num=min(N_SECTOR_SAMPLES, n_sectors), dtype=int))
    return s_idx, t_idx


def _to_frame(sf):
    """Downsample the (n_sectors, n_hours) soiling factor to a (hour x sector) frame."""
    n_sectors, n_hours = sf.shape
    s_idx, t_idx = _sample_indices(n_sectors, n_hours)
    data = sf[np.ix_(s_idx, t_idx)].T  # rows = hours, cols = sectors
    return pd.DataFrame(data, index=pd.Index(t_idx, name="hour"), columns=[f"sector_{s}" for s in s_idx])


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(get_project_root()), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _write_reference(sf):
    REF_DIR.mkdir(parents=True, exist_ok=True)
    # %.12g keeps ~12 significant figures (well inside RTOL) while limiting size.
    _to_frame(sf).to_csv(REF_CSV, float_format="%.12g", na_rep="nan")
    REF_META.write_text(
        json.dumps(
            {
                "heliosoil_version": getattr(heliosoil, "__version__", None),
                "numpy_version": np.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "git_sha": _git_sha(),
                "loss_model": "geometry",
                "config": {"n_trucks": N_TRUCKS, "n_cleans": N_CLEANS, "cleaning_rate": CLEANING_RATE, "dust_type": DUST_TYPE, "file": FILE},
                "soiling_factor_shape": list(sf.shape),
                "time_sampling": "hourly (all)",
                "n_sector_samples": N_SECTOR_SAMPLES,
            },
            indent=2,
        )
    )


def test_woomera_geometric_soiling_factor():
    """Downsampled geometric soiling factor matches the committed CSV baseline."""
    sf = _run_geometric_soiling_factor()

    if UPDATE:
        _write_reference(sf)
        pytest.skip(f"Baseline written to {REF_CSV}. Commit it (and its .meta.json), then re-run without HELIOSOIL_UPDATE_REFERENCES to verify.")

    if not REF_CSV.exists():
        pytest.fail(
            f"Missing baseline {REF_CSV}. Generate it (PowerShell):\n"
            "    $env:HELIOSOIL_UPDATE_REFERENCES=1\n"
            f"    pytest tests/{Path(__file__).name}\n"
            "    Remove-Item Env:\\HELIOSOIL_UPDATE_REFERENCES\n"
            "then commit it and its .meta.json sidecar."
        )

    expected = pd.read_csv(REF_CSV, index_col="hour")
    computed = _to_frame(sf)

    # A changed sampling grid means the field shape changed -- itself a regression.
    assert list(computed.columns) == list(expected.columns), f"sampled sectors changed: {list(computed.columns)} vs {list(expected.columns)}"
    assert list(computed.index) == list(expected.index), "sampled hours changed vs baseline"

    np.testing.assert_allclose(
        computed.to_numpy(),
        expected.to_numpy(),
        rtol=RTOL,
        atol=ATOL,
        equal_nan=True,
        err_msg="geometric soiling factor drifted from the committed baseline",
    )
