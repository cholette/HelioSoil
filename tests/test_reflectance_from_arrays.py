"""
Tests for building ReflectanceMeasurements without Excel files.

``ReflectanceMeasurements`` was only constructible by reading workbooks, which forced
synthetic data and tests to fabricate stand-ins. ``from_arrays`` builds one in memory.

Both constructors fill the per-experiment dictionaries through the same
``_populate_experiment``, so the point of these tests is to confirm that the shared path
really is shared: a workbook is written to a temporary directory, imported, and compared
field by field against ``from_arrays`` given the same numbers. If the two ever diverge --
on the ``rho0`` convention, the ``delta_ref`` sign, the ``sigma_of_the_mean`` scaling, the
prediction-index lookup, or the reference-mirror drift correction -- this fails.

Note the unit difference the comparison has to bridge: the Excel sheets hold percentages
and the importer divides by 100, whereas ``from_arrays`` takes fractions directly.
"""

import numpy as np
import pandas as pd
import pytest

from heliosoil.base_models import ReflectanceMeasurements
from heliosoil.utilities import _nominal_reflectance_series, _nominal_reflectance_anchor


RTOL = 1e-12
ATOL = 0.0

TIMES = np.array([0.0, 3.0, 7.0, 12.0])
TIME_GRID = np.arange(0.0, 15.0, 0.5)
MIRROR_NAMES = ["north", "south", "east"]

# Percentages, as they appear in the workbook.
AVERAGE_PERCENT = np.array(
    [
        [95.1, 94.4, 96.0],
        [93.8, 93.0, 94.7],
        [92.2, 91.1, 93.4],
        [90.6, 89.7, 91.8],
    ]
)
SIGMA_PERCENT = np.array(
    [
        [0.21, 0.18, 0.25],
        [0.23, 0.19, 0.24],
        [0.20, 0.22, 0.26],
        [0.24, 0.21, 0.23],
    ]
)
TILTS_PERCOLUMN = np.array(  # workbook layout: one column per mirror
    [
        [5.0, 10.0, 15.0],
        [7.0, 12.0, 17.0],
        [9.0, 14.0, 19.0],
        [11.0, 16.0, 21.0],
    ]
)
# A reference mirror, re-cleaned each round: its reading tracks instrument drift.
REFERENCE_PERCENT = np.array([96.4, 96.1, 95.6, 95.9])
NUMBER_OF_MEASUREMENTS = 9.0
INCIDENCE_ANGLE = 15.0
ACCEPTANCE_ANGLE = 0.0159


def _write_workbook(path, average_percent=AVERAGE_PERCENT, reference_percent=None):
    """A minimal workbook with the sheets the importer expects.

    ``reference_percent`` adds a column whose name ends in "_Ref", which is what
    _is_reference_mirror_column looks for.
    """
    columns = list(MIRROR_NAMES)
    average_block, sigma_block = average_percent, SIGMA_PERCENT
    if reference_percent is not None:
        columns = columns + ["monitor_Ref"]
        average_block = np.column_stack([average_percent, reference_percent])
        sigma_block = np.column_stack([SIGMA_PERCENT, np.full(len(TIMES), 0.15)])

    average = pd.DataFrame(average_block, columns=columns)
    average.insert(0, "Time", TIMES)
    sigma = pd.DataFrame(sigma_block, columns=columns)
    sigma.insert(0, "Time", TIMES)
    tilts = pd.DataFrame(TILTS_PERCOLUMN, columns=MIRROR_NAMES)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        average.to_excel(writer, sheet_name="Reflectance_Average", index=False)
        sigma.to_excel(writer, sheet_name="Reflectance_Sigma", index=False)
        tilts.to_excel(writer, sheet_name="Tilts", index=False)
    return path


def _imported(tmp_path, average_percent=AVERAGE_PERCENT, import_tilts=True, reference_percent=None):
    path = _write_workbook(tmp_path / "experiment.xlsx", average_percent, reference_percent)
    return ReflectanceMeasurements(
        files=[str(path)],
        time_grids=[TIME_GRID],
        number_of_measurements=[NUMBER_OF_MEASUREMENTS],
        reflectometer_incidence_angle=[INCIDENCE_ANGLE],
        reflectometer_acceptance_angle=[ACCEPTANCE_ANGLE],
        # Passed explicitly, as every caller in the repository does. The dataclass default
        # is an empty list rather than None, and the importer tests `is not None`, so
        # omitting this selects ZERO mirror columns instead of all of them. Not this
        # change's business to fix, but the reason it cannot be left to default here.
        imported_column_names=list(MIRROR_NAMES),
        import_tilts=import_tilts,
    )


def _from_arrays(average_percent=AVERAGE_PERCENT, tilts=TILTS_PERCOLUMN.transpose(), reference_percent=None):
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
        reference_values=None if reference_percent is None else [reference_percent / 100.0],
    )


def _assert_experiments_match(imported, built, keys=(0,)):
    for f in keys:
        for name in ("times", "average", "sigma", "sigma_of_the_mean", "delta_ref", "rho0", "rho0_index", "drift_factor"):
            np.testing.assert_allclose(
                getattr(built, name)[f], getattr(imported, name)[f], rtol=RTOL, atol=ATOL, err_msg=f"mismatch in {name}"
            )
        assert built.mirror_names[f] == imported.mirror_names[f]
        assert list(built.prediction_indices[f]) == list(imported.prediction_indices[f])
        np.testing.assert_allclose(built.prediction_times[f][0], imported.prediction_times[f][0], rtol=RTOL)
        assert built.reflectometer_incidence_angle[f] == imported.reflectometer_incidence_angle[f]
        assert built.reflectometer_acceptance_angle[f] == imported.reflectometer_acceptance_angle[f]
        assert built.number_of_measurements[f] == imported.number_of_measurements[f]
        # nominal_reflectance is present only when a reference mirror was supplied, and its
        # absence is what makes the fitting code fall back to the fixed constant.
        assert (f in built.nominal_reflectance) == (f in imported.nominal_reflectance)
        if f in imported.nominal_reflectance:
            np.testing.assert_allclose(built.nominal_reflectance[f], imported.nominal_reflectance[f], rtol=RTOL, atol=ATOL)


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

    rho0 is a nanmax, so it must ignore the gap rather than propagate it. Real campaigns
    have these: mirrors added mid-campaign carry leading NaNs.
    """
    average = AVERAGE_PERCENT.copy()
    average[1, 2] = np.nan

    imported, built = _imported(tmp_path, average), _from_arrays(average)
    _assert_experiments_match(imported, built)
    assert np.isfinite(built.rho0[0]).all()
    assert np.isnan(built.average[0][1, 2])


def test_from_arrays_matches_the_excel_importer_with_a_reference_mirror(tmp_path):
    """The drift correction agrees between the two paths.

    The workbook carries a "_Ref" column that the importer detects, strips from the mirror
    set and turns into drift_factor / nominal_reflectance; from_arrays is handed the same
    series through reference_values.
    """
    imported = _imported(tmp_path, reference_percent=REFERENCE_PERCENT)
    built = _from_arrays(reference_percent=REFERENCE_PERCENT)

    assert imported.reference_mirror_columns[0] == ["monitor_Ref"]
    assert imported.mirror_names[0] == MIRROR_NAMES  # the reference is not a soiling mirror
    assert 0 in imported.nominal_reflectance and 0 in built.nominal_reflectance
    _assert_experiments_match(imported, built)

    # ... and the drift really is the reference's own movement, not something else.
    expected_drift = REFERENCE_PERCENT / REFERENCE_PERCENT[0]
    np.testing.assert_allclose(built.drift_factor[0], expected_drift, rtol=RTOL)


# ---------------------------------------------------------------------------
# The no-reference default
# ---------------------------------------------------------------------------


def test_no_reference_leaves_the_scalar_nominal_path_in_charge():
    """Without a reference mirror the fitting code must see the fixed constant.

    _nominal_reflectance_series / _nominal_reflectance_anchor gate on nominal_reflectance
    being populated for the file; from_arrays' default must therefore leave it unset, or a
    synthetic dataset would silently switch on the drift correction.
    """
    built = _from_arrays()
    fallback = 0.95

    assert built.nominal_reflectance == {}
    np.testing.assert_array_equal(built.drift_factor[0], np.ones(len(TIMES)))
    assert _nominal_reflectance_series(built, 0, fallback) == fallback
    assert _nominal_reflectance_anchor(built, 0, fallback) == fallback


def test_all_nan_reference_falls_back_to_no_drift():
    """A reference column that is entirely NaN is ignored rather than propagated."""
    built = _from_arrays(reference_percent=np.full(len(TIMES), np.nan))

    assert built.nominal_reflectance == {}
    np.testing.assert_array_equal(built.drift_factor[0], np.ones(len(TIMES)))


# ---------------------------------------------------------------------------
# Behaviour of from_arrays on its own
# ---------------------------------------------------------------------------


def test_from_arrays_derives_the_documented_conventions():
    """rho0, rho0_index, delta_ref and sigma_of_the_mean follow their definitions."""
    built = _from_arrays()
    average = AVERAGE_PERCENT / 100.0

    np.testing.assert_allclose(built.rho0[0], np.nanmax(average, axis=0), rtol=RTOL)
    np.testing.assert_array_equal(built.rho0_index[0], np.nanargmax(average, axis=0))
    np.testing.assert_allclose(built.delta_ref[0][0], np.zeros(len(MIRROR_NAMES)), atol=0.0)
    np.testing.assert_allclose(built.delta_ref[0][1:], -np.diff(average, axis=0), rtol=RTOL)
    np.testing.assert_allclose(built.sigma_of_the_mean[0], SIGMA_PERCENT / 100.0 / np.sqrt(NUMBER_OF_MEASUREMENTS), rtol=RTOL)


def test_from_arrays_defaults_mirror_names_and_angles():
    built = ReflectanceMeasurements.from_arrays(
        times=[TIMES],
        average=[AVERAGE_PERCENT / 100.0],
        sigma=[SIGMA_PERCENT / 100.0],
        time_grids=[TIME_GRID],
    )
    assert built.mirror_names[0] == ["mirror_0", "mirror_1", "mirror_2"]
    assert built.number_of_measurements[0] == 1.0
    assert built.reflectometer_incidence_angle[0] == 0.0
    assert built.tilts == {}
    assert built.reference_mirror_columns[0] == []


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


def test_from_arrays_rejects_ragged_inputs():
    with pytest.raises(ValueError, match="one entry per experiment"):
        ReflectanceMeasurements.from_arrays(
            times=[TIMES],
            average=[AVERAGE_PERCENT / 100.0],
            sigma=[SIGMA_PERCENT / 100.0, SIGMA_PERCENT / 100.0],
            time_grids=[TIME_GRID],
        )


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------


def test_importer_optional_arguments_may_be_omitted(tmp_path):
    """The optional constructor arguments fall back to their documented defaults.

    number_of_measurements, the two reflectometer angles and imported_column_names are all
    documented as Optional and handled as None, but previously defaulted to an empty list.
    Omitting the angles or the count then tripped the length check in _import_option_helper,
    and omitting imported_column_names silently selected zero mirror columns. Every caller
    in the repository passes all four explicitly, which is why it went unnoticed.
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


def test_importer_rejects_an_explicitly_empty_column_list(tmp_path):
    """An empty list is a caller error, not a synonym for "every column"."""
    path = _write_workbook(tmp_path / "experiment.xlsx")
    with pytest.raises(ValueError, match="would import no mirrors"):
        ReflectanceMeasurements(files=[str(path)], time_grids=[TIME_GRID], imported_column_names=[])


def test_importer_reports_which_option_has_the_wrong_length(tmp_path):
    """A mismatched per-experiment list names itself and both lengths.

    The old message was a plain string containing a literal "{option}" placeholder, so it
    identified neither the option nor the sizes.
    """
    path = _write_workbook(tmp_path / "experiment.xlsx")
    with pytest.raises(ValueError, match=r"number_of_measurements has 2 entries but there are 1 experiment"):
        ReflectanceMeasurements(files=[str(path)], time_grids=[TIME_GRID], number_of_measurements=[9.0, 9.0])
