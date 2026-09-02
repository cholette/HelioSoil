"""
Tests for building ReflectanceMeasurements without Excel files.

``ReflectanceMeasurements`` was only constructible by reading workbooks, which forced
synthetic data and tests to fabricate stand-ins. ``from_arrays`` builds one in memory.

Both constructors fill the per-experiment dictionaries through the same
``_populate_experiment``, so the point of these tests is to confirm that the shared path
really is shared: a workbook is written to a temporary directory, imported, and compared
field by field against ``from_arrays`` given the same numbers. If the two ever diverge --
on the ``rho0`` convention, the ``delta_ref`` sign, the ``sigma_of_the_mean`` scaling, or
the prediction-index lookup -- this fails.

Note the unit difference the comparison has to bridge: the Excel sheets hold percentages
and the importer divides by 100, whereas ``from_arrays`` takes fractions directly.
"""

import numpy as np
import pandas as pd
import pytest

from heliosoil.base_models import ReflectanceMeasurements


RTOL = 1e-12
ATOL = 0.0

TIMES = np.array([0.0, 3.0, 7.0, 12.0])
TIME_GRID = np.arange(0.0, 15.0, 0.5)
MIRROR_NAMES = ["north", "south", "east"]

# Percentages, as they appear in the workbook.
AVERAGE_PERCENT = np.array([[95.1, 94.4, 96.0], [93.8, 93.0, 94.7], [92.2, 91.1, 93.4], [90.6, 89.7, 91.8]])
SIGMA_PERCENT = np.array([[0.21, 0.18, 0.25], [0.23, 0.19, 0.24], [0.20, 0.22, 0.26], [0.24, 0.21, 0.23]])
TILTS_PERCOLUMN = np.array(  # workbook layout: one column per mirror
    [[5.0, 10.0, 15.0], [7.0, 12.0, 17.0], [9.0, 14.0, 19.0], [11.0, 16.0, 21.0]]
)
NUMBER_OF_MEASUREMENTS = 9.0
INCIDENCE_ANGLE = 15.0
ACCEPTANCE_ANGLE = 0.0159


def _write_workbook(path, average_percent=AVERAGE_PERCENT):
    """A minimal workbook with the sheets the importer expects."""
    average = pd.DataFrame(average_percent, columns=MIRROR_NAMES)
    average.insert(0, "Time", TIMES)
    sigma = pd.DataFrame(SIGMA_PERCENT, columns=MIRROR_NAMES)
    sigma.insert(0, "Time", TIMES)
    tilts = pd.DataFrame(TILTS_PERCOLUMN, columns=MIRROR_NAMES)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        average.to_excel(writer, sheet_name="Reflectance_Average", index=False)
        sigma.to_excel(writer, sheet_name="Reflectance_Sigma", index=False)
        tilts.to_excel(writer, sheet_name="Tilts", index=False)
    return path


def _imported(tmp_path, average_percent=AVERAGE_PERCENT, import_tilts=True):
    path = _write_workbook(tmp_path / "experiment.xlsx", average_percent)
    return ReflectanceMeasurements(
        files=[str(path)],
        time_grids=[TIME_GRID],
        number_of_measurements=[NUMBER_OF_MEASUREMENTS],
        reflectometer_incidence_angle=[INCIDENCE_ANGLE],
        reflectometer_acceptance_angle=[ACCEPTANCE_ANGLE],
        import_tilts=import_tilts,
    )


def _from_arrays(average_percent=AVERAGE_PERCENT, tilts=TILTS_PERCOLUMN.transpose()):
    return ReflectanceMeasurements.from_arrays(
        times=[TIMES],
        average=[average_percent / 100.0],
        sigma=[SIGMA_PERCENT / 100.0],
        time_grids=[TIME_GRID],
        tilts=None if tilts is None else [tilts],
        mirror_names=[MIRROR_NAMES],
        number_of_measurements=[NUMBER_OF_MEASUREMENTS],
        reflectometer_incidence_angle=[INCIDENCE_ANGLE],
        reflectometer_acceptance_angle=[ACCEPTANCE_ANGLE],
    )


def _assert_experiments_match(imported, built, keys=(0,)):
    for f in keys:
        for name in ("times", "average", "sigma", "sigma_of_the_mean", "delta_ref", "rho0"):
            np.testing.assert_allclose(getattr(built, name)[f], getattr(imported, name)[f], rtol=RTOL, atol=ATOL, err_msg=f"mismatch in {name}")
        assert built.mirror_names[f] == imported.mirror_names[f]
        assert list(built.prediction_indices[f]) == list(imported.prediction_indices[f])
        np.testing.assert_allclose(built.prediction_times[f][0], imported.prediction_times[f][0], rtol=RTOL)
        assert built.reflectometer_incidence_angle[f] == imported.reflectometer_incidence_angle[f]
        assert built.reflectometer_acceptance_angle[f] == imported.reflectometer_acceptance_angle[f]
        assert built.number_of_measurements[f] == imported.number_of_measurements[f]


# ---------------------------------------------------------------------------
# Equivalence with the Excel importer
# ---------------------------------------------------------------------------


def test_from_arrays_matches_the_excel_importer(tmp_path):
    """Every populated field agrees with the workbook path, given the same numbers."""
    _assert_experiments_match(_imported(tmp_path), _from_arrays())


def test_from_arrays_matches_the_excel_importer_with_tilts(tmp_path):
    """Tilts are stored as (n_mirrors, n_times) by both paths."""
    imported, built = _imported(tmp_path), _from_arrays()

    np.testing.assert_allclose(built.tilts[0], imported.tilts[0], rtol=RTOL, atol=ATOL)
    assert built.tilts[0].shape == (len(MIRROR_NAMES), len(TIMES))


def test_from_arrays_matches_the_excel_importer_with_missing_values(tmp_path):
    """A NaN measurement flows through both paths identically.

    rho0 is a nanmax, so it must ignore the gap rather than propagate it.
    """
    average = AVERAGE_PERCENT.copy()
    average[1, 2] = np.nan

    imported, built = _imported(tmp_path, average), _from_arrays(average)
    _assert_experiments_match(imported, built)
    assert np.isfinite(built.rho0[0]).all()
    assert np.isnan(built.average[0][1, 2])


# ---------------------------------------------------------------------------
# Behaviour of from_arrays on its own
# ---------------------------------------------------------------------------


def test_from_arrays_derives_the_documented_conventions():
    """rho0, delta_ref and sigma_of_the_mean follow their definitions."""
    built = _from_arrays()
    average = AVERAGE_PERCENT / 100.0

    np.testing.assert_allclose(built.rho0[0], np.nanmax(average, axis=0), rtol=RTOL)
    np.testing.assert_allclose(built.delta_ref[0][0], np.zeros(len(MIRROR_NAMES)), atol=0.0)
    np.testing.assert_allclose(built.delta_ref[0][1:], -np.diff(average, axis=0), rtol=RTOL)
    np.testing.assert_allclose(built.sigma_of_the_mean[0], SIGMA_PERCENT / 100.0 / np.sqrt(NUMBER_OF_MEASUREMENTS), rtol=RTOL)


def test_from_arrays_defaults_mirror_names_and_angles():
    built = ReflectanceMeasurements.from_arrays(
        times=[TIMES], average=[AVERAGE_PERCENT / 100.0], sigma=[SIGMA_PERCENT / 100.0], time_grids=[TIME_GRID]
    )
    assert built.mirror_names[0] == ["mirror_0", "mirror_1", "mirror_2"]
    assert built.number_of_measurements[0] == 1.0
    assert built.reflectometer_incidence_angle[0] == 0.0
    assert built.tilts == {}


def test_from_arrays_handles_several_experiments():
    """Experiments may differ in mirror count and measurement times."""
    second_times = np.array([0.0, 5.0, 11.0])
    second_average = np.array([[0.94, 0.95], [0.92, 0.93], [0.90, 0.91]])

    built = ReflectanceMeasurements.from_arrays(
        times=[TIMES, second_times],
        average=[AVERAGE_PERCENT / 100.0, second_average],
        sigma=[SIGMA_PERCENT / 100.0, np.full_like(second_average, 2e-3)],
        time_grids=[TIME_GRID, TIME_GRID],
        number_of_measurements=4.0,  # a scalar applies to every experiment
    )

    assert built.average[0].shape == (4, 3)
    assert built.average[1].shape == (3, 2)
    assert built.mirror_names[1] == ["mirror_0", "mirror_1"]
    assert built.number_of_measurements == [4.0, 4.0]
    assert list(built.prediction_indices[1]) == [0, 10, 22]


def test_importer_optional_arguments_may_be_omitted(tmp_path):
    """The optional constructor arguments fall back to their documented defaults.

    number_of_measurements, the two reflectometer angles and imported_column_names are
    all tested against None, but previously defaulted to an empty list. Omitting the
    angles or the count then tripped the length check in _import_option_helper, and
    omitting imported_column_names silently selected zero mirror columns. Every caller
    in the repository passed all four explicitly, which is why it went unnoticed.
    """
    path = _write_workbook(tmp_path / "experiment.xlsx")
    data = ReflectanceMeasurements(files=[str(path)], time_grids=[TIME_GRID])

    assert data.average[0].shape == (len(TIMES), len(MIRROR_NAMES))
    assert data.mirror_names[0] == MIRROR_NAMES
    assert data.number_of_measurements == [1.0]
    assert data.reflectometer_incidence_angle == [0.0]
    assert data.reflectometer_acceptance_angle == [0.0]

    # sigma_of_the_mean divides by sqrt(1) in this case, i.e. it equals sigma.
    np.testing.assert_allclose(data.sigma_of_the_mean[0], SIGMA_PERCENT / 100.0, rtol=RTOL, atol=ATOL)


def test_from_arrays_rejects_ragged_inputs():
    with pytest.raises(ValueError, match="one entry per experiment"):
        ReflectanceMeasurements.from_arrays(
            times=[TIMES], average=[AVERAGE_PERCENT / 100.0], sigma=[SIGMA_PERCENT / 100.0, SIGMA_PERCENT / 100.0], time_grids=[TIME_GRID]
        )
