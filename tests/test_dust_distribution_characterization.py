r"""
Characterization test for the Dust size-distribution outputs.

Pins the full number/mass/area distributions (on a fine log-diameter grid) and
the PM2.5 / PM10 / TSP integrals produced by ``Dust._compute_distributions``,
so the planned refactor onto the ``dust_distributions`` classes can be proven to
preserve the science. Baselines are built from known mixture parameters on a
self-contained grid (no data files), so they are fully reproducible.

Several mixtures are pinned (see CASES): two trimodal distributions and a
single-mode distribution (the latter exercises the N_components == 1 path).
Each case has its own baseline CSV + .meta.json.

------------------------------------------------------------------------------
Units
------------------------------------------------------------------------------
Baselines are stored in the target *standard* units:

    number  pdfN : d#/dlog10(D) per cm^3
    mass    pdfM : d(ug)/dlog10(D) per m^3
    area    pdfA : d(um^2)/dlog10(D) per cm^3      (surface-area convention)
    PM2.5/PM10/TSP : ug/m^3

The current Dust class does not yet use these units everywhere, so each computed
array is multiplied by a fixed conversion constant below to reach the standard.
These constants are exact powers of ten and document the unit migration:

    * Today they convert current Dust output -> standard units.
    * After Dust is reworked to emit standard units natively, set each constant
      to 1.0; the committed baselines (already in standard units) then verify
      that the standardization preserved the values.

------------------------------------------------------------------------------
Generating / updating the baselines
------------------------------------------------------------------------------
PowerShell:
    $env:HELIOSOIL_UPDATE_REFERENCES=1
    pytest tests/test_dust_distribution_characterization.py
    Remove-Item Env:\HELIOSOIL_UPDATE_REFERENCES

Inspect tests/reference_data/dust_distribution_pdfs__*.csv (+ .meta.json),
commit them, then re-run without the flag to verify.
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
from heliosoil.utilities import get_project_root


FILE = 0

# Fine, self-contained log-diameter grid [um], shared by all cases.
GRID_D = np.logspace(-3.0, 4.0, 2000)

# Mixture cases: number conc. Nd [1/cm^3], mode diameters mu [um],
# geometric widths sigma [-], particle density rho [kg/m^3].
CASES = {
    "trimodal_silica": {
        "Nd": [3000.0, 999.875, 0.125],
        "mu": [0.0117, 0.051231, 0.8226],
        "sigma": [1.71061, 2.239, 2.512],
        "rho": 2000.0,
    },
    "trimodal_mineral": {
        "Nd": [1200.0, 480.0, 6.0],
        "mu": [0.02, 0.15, 2.0],
        "sigma": [1.6, 1.8, 2.2],
        "rho": 2650.0,
    },
    "single_mode": {
        "Nd": [800.0],
        "mu": [0.3],
        "sigma": [2.0],
        "rho": 2650.0,
    },
}

# --- current Dust output units -> target standard units (exact) --------------
# pdfN: current [1/m^3]/dlog10  ->  standard [1/cm^3]/dlog10
PDFN_CURRENT_TO_STD = 1e-6
# pdfM: current [ug/m^3]/dlog10 (already standard)
PDFM_CURRENT_TO_STD = 1.0
# pdfA: current [m^2/m^3]/dlog10 -> standard [um^2/cm^3]/dlog10
PDFA_CURRENT_TO_STD = 1e6
# PM*/TSP: current [ug/m^3] (already standard)
PM_CURRENT_TO_STD = 1.0

REF_DIR = Path(__file__).parent / "reference_data"

_TRUE = {"1", "true", "True", "yes", "on"}
UPDATE = os.environ.get("HELIOSOIL_UPDATE_REFERENCES", "") in _TRUE

RTOL = 1e-9
ATOL = 1e-12


def _pdf_csv(case):
    return REF_DIR / f"dust_distribution_pdfs__{case}.csv"


def _meta_json(case):
    return REF_DIR / f"dust_distribution__{case}.meta.json"


def _build_dust(case):
    p = CASES[case]
    dust = smb.Dust()
    dust.D = {FILE: GRID_D.copy()}
    dust.rho = {FILE: float(p["rho"])}
    dust.Nd = {FILE: np.array(p["Nd"], dtype=float)}
    dust.log10_mu = {FILE: np.log10(np.array(p["mu"], dtype=float))}
    dust.log10_sig = {FILE: np.log10(np.array(p["sigma"], dtype=float))}
    dust._compute_distributions(f=FILE)
    return dust


def _standard_pdfs(dust):
    """Distribution arrays in target standard units, as a tidy frame."""
    return pd.DataFrame(
        {
            "D_um": dust.D[FILE],
            "pdfN_per_cm3_per_dlog10D": dust.pdfN[FILE] * PDFN_CURRENT_TO_STD,
            "pdfM_ug_per_m3_per_dlog10D": dust.pdfM[FILE] * PDFM_CURRENT_TO_STD,
            "pdfA_um2_per_cm3_per_dlog10D": dust.pdfA[FILE] * PDFA_CURRENT_TO_STD,
        }
    )


def _standard_integrals(dust):
    """PM2.5 / PM10 / TSP in ug/m^3, from the analytic mass CDF."""
    return {
        "PM2.5": float(dust.pm_concentration(FILE, 2.5)) * PM_CURRENT_TO_STD,
        "PM10": float(dust.PM10[FILE]) * PM_CURRENT_TO_STD,
        "TSP": float(dust.TSP[FILE]) * PM_CURRENT_TO_STD,
    }


def _git_sha():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(get_project_root()),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _write_reference(case, dust):
    REF_DIR.mkdir(parents=True, exist_ok=True)
    _standard_pdfs(dust).to_csv(_pdf_csv(case), index=False, float_format="%.12g")
    _meta_json(case).write_text(
        json.dumps(
            {
                "case": case,
                "heliosoil_version": getattr(heliosoil, "__version__", None),
                "numpy_version": np.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "git_sha": _git_sha(),
                "units": {
                    "pdfN": "d#/dlog10(D) per cm^3",
                    "pdfM": "d(ug)/dlog10(D) per m^3",
                    "pdfA": "d(um^2)/dlog10(D) per cm^3",
                    "PM2.5/PM10/TSP": "ug/m^3",
                },
                "conversions_current_to_std": {
                    "pdfN": PDFN_CURRENT_TO_STD,
                    "pdfM": PDFM_CURRENT_TO_STD,
                    "pdfA": PDFA_CURRENT_TO_STD,
                    "PM": PM_CURRENT_TO_STD,
                },
                "mixture": CASES[case],
                "grid": {"n": GRID_D.size, "D_min_um": GRID_D[0], "D_max_um": GRID_D[-1]},
                "expected_integrals_ug_per_m3": _standard_integrals(dust),
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("case", list(CASES))
def test_dust_distribution_pdfs(case):
    """pdfN / pdfM / pdfA (standard units, fine grid) match the committed baseline."""
    dust = _build_dust(case)

    if UPDATE:
        _write_reference(case, dust)
        pytest.skip(
            f"Baseline for '{case}' written to {_pdf_csv(case)} (+ .meta.json). Commit "
            "both, then re-run without HELIOSOIL_UPDATE_REFERENCES to verify."
        )

    ref_csv = _pdf_csv(case)
    if not ref_csv.exists():
        pytest.fail(
            f"Missing baseline {ref_csv}. Generate it (PowerShell):\n"
            "    $env:HELIOSOIL_UPDATE_REFERENCES=1\n"
            f"    pytest tests/{Path(__file__).name}\n"
            "    Remove-Item Env:\\HELIOSOIL_UPDATE_REFERENCES"
        )

    expected = pd.read_csv(ref_csv)
    computed = _standard_pdfs(dust)

    assert list(computed.columns) == list(expected.columns), "pdf columns changed vs baseline"
    np.testing.assert_allclose(
        computed["D_um"].to_numpy(), expected["D_um"].to_numpy(), rtol=RTOL, atol=0.0,
        err_msg="diameter grid changed vs baseline",
    )
    pdf_cols = (
        "pdfN_per_cm3_per_dlog10D",
        "pdfM_ug_per_m3_per_dlog10D",
        "pdfA_um2_per_cm3_per_dlog10D",
    )
    for col in pdf_cols:
        np.testing.assert_allclose(
            computed[col].to_numpy(), expected[col].to_numpy(), rtol=RTOL, atol=ATOL,
            err_msg=f"{col} drifted from the committed baseline for '{case}'",
        )


@pytest.mark.parametrize("case", list(CASES))
def test_dust_distribution_integrals(case):
    """PM2.5 / PM10 / TSP (ug/m^3) match the committed baseline."""
    dust = _build_dust(case)

    if UPDATE:
        pytest.skip("Integrals are written with the pdf baseline in test_dust_distribution_pdfs.")

    ref_meta = _meta_json(case)
    if not ref_meta.exists():
        pytest.fail(
            f"Missing baseline {ref_meta}. Generate it with HELIOSOIL_UPDATE_REFERENCES=1 "
            "(see test_dust_distribution_pdfs)."
        )

    expected = json.loads(ref_meta.read_text())["expected_integrals_ug_per_m3"]
    computed = _standard_integrals(dust)

    assert set(computed) == set(expected), "integral set changed vs baseline"
    for name in expected:
        np.testing.assert_allclose(
            computed[name], expected[name], rtol=RTOL, atol=ATOL,
            err_msg=f"{name} drifted from the committed baseline for '{case}'",
        )
