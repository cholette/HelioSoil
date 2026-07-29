"""
Tests for reference-mirror ("_ref"/"_Ref") column handling: a reference mirror is
re-cleaned before each day's measurement round, so its reading tracks day-to-day
instrument/environmental drift rather than soiling. Covers detection and exclusion
from the trainable mirror set (heliosoil.base_models.ReflectanceMeasurements,
heliosoil.utilities.get_training_data), the drift-corrected nominal_reflectance
derived from it and its interaction with trim_experiment_data/daily_average, and
its use as the fitting anchor in heliosoil.fitting.CommonFittingMethods.
"""

import logging
import types

import numpy as np
import pandas as pd
import pytest

from heliosoil.base_models import ReflectanceMeasurements
from heliosoil.fitting import CommonFittingMethods
from heliosoil.utilities import _nominal_reflectance_anchor, _nominal_reflectance_series, daily_average, get_training_data, trim_experiment_data

TIMES = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"])


def _write_reflectance_file(path, times, average_cols, sigma_value=0.3, tilts=None):
    """Write a minimal Reflectance_Average/Reflectance_Sigma(/Tilts) workbook.

    average_cols: {column_name: array of len(times)}, in percent units (NaN
    allowed). Sigma is a constant `sigma_value` wherever average is valid, NaN
    wherever average is missing -- mirroring real sigma sheets, where a reading
    that wasn't taken has no associated uncertainty either.
    """
    avg_df = pd.DataFrame({"Time": times, **average_cols})
    sigma_cols = {name: np.where(np.isnan(vals), np.nan, sigma_value) for name, vals in average_cols.items()}
    sigma_df = pd.DataFrame({"Time": times, **sigma_cols})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        avg_df.to_excel(writer, sheet_name="Reflectance_Average", index=False)
        sigma_df.to_excel(writer, sheet_name="Reflectance_Sigma", index=False)
        if tilts is not None:
            pd.DataFrame({"Time": times, **tilts}).to_excel(writer, sheet_name="Tilts", index=False)


def _load(path, time_grid=None, **overrides):
    """ReflectanceMeasurements with every default-empty-list gotcha (number_of_
    measurements/reflectometer_*/imported_column_names all default to `[]`, not
    None, via the dataclass's default_factory=list) pinned to None explicitly,
    matching how every real caller in analysis_scripts/*.py already does it."""
    if time_grid is None:
        time_grid = pd.read_excel(path, sheet_name="Reflectance_Average")["Time"].values
    kwargs = dict(
        files=[str(path)],
        time_grids=[time_grid],
        number_of_measurements=None,
        reflectometer_incidence_angle=None,
        reflectometer_acceptance_angle=None,
        import_tilts=False,
        imported_column_names=None,
        verbose=False,
    )
    kwargs.update(overrides)
    return ReflectanceMeasurements(**kwargs)


def _base_columns():
    """2 regular mirrors + 1 reference column. ON_M1_T00 is always valid
    (monotonic soiling). OE_M2_T30 is added mid-campaign (leading NaNs, matching
    the real data/carwarp shape) with a noisy reading that peaks AFTER its own
    first valid row, so its rho0 (nanmax) index differs from its baseline
    (first-valid-row) index. OW_M1_T00_Ref has one internal NaN gap.
    """
    return {
        "ON_M1_T00": np.array([95.0, 94.5, 94.0, 93.5, 93.0, 92.5]),
        "OE_M2_T30": np.array([np.nan, np.nan, 94.0, 94.3, 93.0, 92.0]),
        "OW_M1_T00_Ref": np.array([95.0, 95.0, np.nan, 94.05, 93.1, 92.15]),
    }


# --------------------------------------------------------------------------- #
# import_reflectance_data: detection, exclusion, baseline + drift computation
# --------------------------------------------------------------------------- #


def test_import_excludes_reference_column_from_mirror_names_and_tilts(tmp_path):
    path = tmp_path / "soiling_test_20240101_20240106.xlsx"
    _write_reflectance_file(
        path, TIMES, _base_columns(), tilts={"ON_M1_T00": np.zeros(6), "OE_M2_T30": np.full(6, 30.0), "OW_M1_T00_Ref": np.zeros(6)}
    )
    rdat = _load(path, import_tilts=True)

    assert rdat.mirror_names[0] == ["ON_M1_T00", "OE_M2_T30"]
    assert rdat.reference_mirror_columns[0] == ["OW_M1_T00_Ref"]
    assert rdat.average[0].shape == (6, 2)
    assert rdat.sigma[0].shape == (6, 2)
    assert rdat.tilts[0].shape == (2, 6)
    np.testing.assert_array_equal(rdat.tilts[0][0], np.zeros(6))
    np.testing.assert_array_equal(rdat.tilts[0][1], np.full(6, 30.0))


def test_drift_factor_forward_fills_gap_and_nominal_reflectance_uses_own_baseline(tmp_path):
    path = tmp_path / "soiling_test_drift_20240101_20240106.xlsx"
    _write_reflectance_file(path, TIMES, _base_columns())
    rdat = _load(path)

    expected_drift = np.array([1.0, 1.0, 1.0, 0.99, 0.98, 0.97])
    np.testing.assert_allclose(rdat.drift_factor[0], expected_drift, rtol=1e-10)

    # ON_M1_T00's baseline is its row-0 reading (0.95); OE_M2_T30's is its OWN
    # first-valid row (row 2, 0.94) -- not row 0 (NaN there).
    expected_nominal = np.array(
        [[0.95, 0.94], [0.95, 0.94], [0.95, 0.94], [0.95 * 0.99, 0.94 * 0.99], [0.95 * 0.98, 0.94 * 0.98], [0.95 * 0.97, 0.94 * 0.97]]
    )
    np.testing.assert_allclose(rdat.nominal_reflectance[0], expected_nominal, rtol=1e-10)


def test_rho0_index_self_consistent_and_can_differ_from_baseline_row(tmp_path):
    path = tmp_path / "soiling_test_rho0idx_20240101_20240106.xlsx"
    _write_reflectance_file(path, TIMES, _base_columns())
    rdat = _load(path)

    for mirror in range(rdat.average[0].shape[1]):
        idx = rdat.rho0_index[0][mirror]
        assert rdat.average[0][idx, mirror] == pytest.approx(rdat.rho0[0][mirror])

    # OE_M2_T30's max (0.943, row 3) is a DIFFERENT row than its first-valid
    # baseline (row 2, 0.94) -- a noisy reading exceeding the true clean value.
    assert rdat.rho0_index[0][1] == 3
    assert rdat.rho0[0][1] == pytest.approx(0.943)


def test_nominal_reflectance_anchor_uses_rho0_time_not_baseline_time(tmp_path):
    path = tmp_path / "soiling_test_anchor_20240101_20240106.xlsx"
    _write_reflectance_file(path, TIMES, _base_columns())
    rdat = _load(path)

    anchor = _nominal_reflectance_anchor(rdat, 0, fallback=0.5)
    # mirror 0 (ON_M1_T00): rho0_index=0 -> nominal_reflectance[0, 0] = 0.95
    # mirror 1 (OE_M2_T30): rho0_index=3 -> nominal_reflectance[3, 1] = 0.94*0.99,
    # NOT its baseline 0.94 -- the anchor is evaluated at rho0's own time.
    np.testing.assert_allclose(anchor, [0.95, 0.94 * 0.99], rtol=1e-10)


def test_multiple_reference_columns_are_averaged_with_warning(tmp_path, caplog):
    path = tmp_path / "soiling_test_multiref_20240101_20240106.xlsx"
    cols = {
        "ON_M1_T00": np.array([95.0, 94.5, 94.0, 93.5, 93.0, 92.5]),
        "M1_Ref": np.array([95.0, 95.2, 95.0, 94.8, 95.0, 95.1]),
        "M2_Ref": np.array([95.4, 95.0, 94.8, 95.2, 95.0, 94.9]),
    }
    _write_reflectance_file(path, TIMES, cols)
    with caplog.at_level(logging.WARNING, logger="heliosoil"):
        rdat = _load(path)

    assert sorted(rdat.reference_mirror_columns[0]) == ["M1_Ref", "M2_Ref"]
    assert "averaging" in caplog.text
    assert rdat.mirror_names[0] == ["ON_M1_T00"]

    expected_ref_avg = (cols["M1_Ref"] + cols["M2_Ref"]) / 2 / 100.0
    expected_drift = expected_ref_avg / expected_ref_avg[0]
    np.testing.assert_allclose(rdat.drift_factor[0], expected_drift, rtol=1e-10)


def test_no_reference_column_leaves_nominal_reflectance_unset(tmp_path):
    """Absence of the file's key in nominal_reflectance is precisely what signals
    fitting code to fall back to the fixed helios.nominal_reflectance constant."""
    path = tmp_path / "soiling_test_noref_20240101_20240106.xlsx"
    _write_reflectance_file(path, TIMES, {"ON_M1_T00": np.array([95.0, 94.5, 94.0, 93.5, 93.0, 92.5])})
    rdat = _load(path)

    assert rdat.reference_mirror_columns[0] == []
    np.testing.assert_array_equal(rdat.drift_factor[0], np.ones(6))
    assert 0 not in rdat.nominal_reflectance

    fallback = 0.95
    assert _nominal_reflectance_series(rdat, 0, fallback) == fallback
    assert _nominal_reflectance_anchor(rdat, 0, fallback) == fallback


def test_all_nan_reference_column_is_ignored(tmp_path, caplog):
    path = tmp_path / "soiling_test_allnanref_20240101_20240106.xlsx"
    cols = {"ON_M1_T00": np.array([95.0, 94.5, 94.0, 93.5, 93.0, 92.5]), "ON_M1_T00_Ref": np.full(6, np.nan)}
    _write_reflectance_file(path, TIMES, cols)
    with caplog.at_level(logging.WARNING, logger="heliosoil"):
        rdat = _load(path)

    assert "entirely NaN" in caplog.text
    np.testing.assert_array_equal(rdat.drift_factor[0], np.ones(6))
    assert 0 not in rdat.nominal_reflectance


# --------------------------------------------------------------------------- #
# get_training_data: exclude a reference column common to every file
# --------------------------------------------------------------------------- #


def test_get_training_data_excludes_reference_column_common_to_all_files(tmp_path):
    # Mirrors the real data/yadnarie shape: the SAME "_Ref" column name recurs
    # across every campaign file, so it would otherwise survive the "common to
    # all files" intersection and leak into all_mirrors/training.
    cols1 = {"ONE_M1_T00_Ref": np.array([95.0, 95.1, 95.0, 94.9, 95.0, 95.1]), "OSE_M1_T90": np.array([95.0, 94.0, 93.0, 92.0, 91.0, 90.0])}
    cols2 = {"ONE_M1_T00_Ref": np.array([95.2, 95.1, 95.0]), "OSE_M1_T90": np.array([95.0, 94.5, 94.0])}
    _write_reflectance_file(tmp_path / "experiment_20240101_20240106.xlsx", TIMES, cols1)
    _write_reflectance_file(tmp_path / "experiment_20240201_20240203.xlsx", pd.to_datetime(["2024-02-01", "2024-02-02", "2024-02-03"]), cols2)

    _files, _intervals, mirror_names, common = get_training_data(str(tmp_path), "experiment_")

    for names in mirror_names:
        assert names == ["OSE_M1_T90"]
    assert common == ["OSE_M1_T90"]


# --------------------------------------------------------------------------- #
# trim_experiment_data: nominal_reflectance/drift_factor sliced, NOT recomputed;
# rho0/rho0_index ARE recomputed (existing, intentional slice-relative behavior)
# --------------------------------------------------------------------------- #


def _minimal_sim_stub(f, times):
    n = len(times)
    return types.SimpleNamespace(
        time={f: pd.Series(pd.to_datetime(times))},
        dust_concentration={f: np.ones(n)},
        wind_speed_mov_avg={f: np.ones(n)},
        dust_conc_mov_avg={f: np.ones(n)},
    )


def test_trim_experiment_data_slices_nominal_reflectance_not_recompute():
    f = 0
    times = pd.to_datetime(TIMES).values.astype("datetime64[ns]")

    average = np.array([[0.950, np.nan], [0.945, np.nan], [0.940, 0.940], [0.935, 0.943], [0.930, 0.930], [0.925, 0.920]])
    sigma = np.full_like(average, 0.003)
    nominal_reflectance = np.array([[0.950, 0.940], [0.950, 0.940], [0.950, 0.940], [0.9405, 0.9306], [0.931, 0.9212], [0.9215, 0.9118]])
    drift_factor = np.array([1.0, 1.0, 1.0, 0.99, 0.98, 0.97])

    ref_dat = types.SimpleNamespace(
        times={f: times.copy()},
        average={f: average.copy()},
        sigma={f: sigma.copy()},
        sigma_of_the_mean={f: sigma.copy()},
        nominal_reflectance={f: nominal_reflectance.copy()},
        drift_factor={f: drift_factor.copy()},
        rho0={f: np.array([0.950, 0.943])},
        rho0_index={f: np.array([0, 3])},
        prediction_indices={},
        prediction_times={},
        soiling_rate={},
    )
    sim_dat = _minimal_sim_stub(f, times)

    _trimmed_sim, trimmed_ref = trim_experiment_data(sim_dat, ref_dat, "reflectance_data")

    # Mirror 1's leading NaNs (rows 0-1) force the trim window to rows 2-5.
    np.testing.assert_array_equal(trimmed_ref.nominal_reflectance[f], nominal_reflectance[2:6, :])
    np.testing.assert_array_equal(trimmed_ref.drift_factor[f], drift_factor[2:6])

    # rho0/rho0_index ARE recomputed from the trimmed sub-window: mirror 0's
    # pre-trim max (0.950, row 0) is gone, so its rho0 correctly drops to the
    # highest surviving reading -- this is existing, load-bearing behavior for
    # cross-validation windows, and must be unaffected by the change above.
    np.testing.assert_allclose(trimmed_ref.rho0[f], [0.940, 0.943])
    np.testing.assert_array_equal(trimmed_ref.rho0_index[f], [0, 1])  # local indices into the trimmed array


# --------------------------------------------------------------------------- #
# daily_average: nominal_reflectance/drift_factor aggregated; rho0/rho0_index
# refreshed from the day-aggregated average (previously a stale-rho0 gap)
# --------------------------------------------------------------------------- #


def test_daily_average_aggregates_nominal_reflectance_and_refreshes_rho0():
    f = 0
    times = pd.to_datetime(["2024-01-01 08:00", "2024-01-01 16:00", "2024-01-02 08:00", "2024-01-02 16:00"]).values.astype("datetime64[ns]")

    average = np.array([[0.950], [0.940], [0.900], [0.890]])
    sigma = np.full_like(average, 0.01)
    drift_factor = np.array([1.00, 0.99, 0.98, 0.97])
    nominal_reflectance = 0.95 * drift_factor[:, None]

    ref_dat = types.SimpleNamespace(
        files=[f],
        times={f: times.copy()},
        average={f: average.copy()},
        sigma={f: sigma.copy()},
        sigma_of_the_mean={f: sigma.copy()},
        nominal_reflectance={f: nominal_reflectance.copy()},
        drift_factor={f: drift_factor.copy()},
        rho0={f: np.array([0.950])},
        rho0_index={f: np.array([0])},
        prediction_indices={},
        prediction_times={},
        soiling_rate={f: np.zeros(1)},
        delta_ref={f: np.zeros_like(average)},
        number_of_measurements=[1.0],
    )

    result = daily_average(ref_dat, time_grids=[times])

    assert result.average[f].shape == (2, 1)
    np.testing.assert_allclose(result.drift_factor[f], [(1.00 + 0.99) / 2, (0.98 + 0.97) / 2])
    np.testing.assert_allclose(
        result.nominal_reflectance[f][:, 0],
        [(nominal_reflectance[0, 0] + nominal_reflectance[1, 0]) / 2, (nominal_reflectance[2, 0] + nominal_reflectance[3, 0]) / 2],
    )

    # Day 1's mean (0.945) is now the max -- rho0/rho0_index refreshed from the
    # day-aggregated average, not left stale at the pre-aggregation value (0.950).
    np.testing.assert_allclose(result.rho0[f], [(0.950 + 0.940) / 2])
    np.testing.assert_array_equal(result.rho0_index[f], [0])


# --------------------------------------------------------------------------- #
# compute_soiling_factor / _sse: the anchor-recovery identity, and both
# fallback paths (reflectance_data=None; present but missing the new fields)
# --------------------------------------------------------------------------- #


def _fit_stub(helios):
    model = CommonFittingMethods.__new__(CommonFittingMethods)
    model.helios = helios
    return model


def test_compute_soiling_factor_anchor_recovers_rho0_with_time_varying_nominal_reflectance():
    f = 0
    rho0 = np.array([0.90, 0.80])
    nominal_reflectance = np.array([[0.95, 0.85], [0.94, 0.85], [0.93, 0.83]])  # (n_time=3, n_mirrors=2)
    rho0_index = np.array([1, 0])  # mirror 0's anchor at t=1, mirror 1's at t=0
    inc_ref_factor = np.float64(2.0)  # a plain Python float lacks .squeeze(), unlike every real caller's value

    helios = types.SimpleNamespace(
        tilt={f: np.zeros((2, 3))},
        delta_soiled_area={f: np.zeros((2, 3))},  # isolate the anchor: no accumulation
        inc_ref_factor={f: inc_ref_factor},
        nominal_reflectance=0.5,  # deliberately "wrong" fallback -- must NOT be used
    )
    model = _fit_stub(helios)

    reflectance_data = types.SimpleNamespace(rho0={f: rho0}, nominal_reflectance={f: nominal_reflectance}, rho0_index={f: rho0_index})
    model.compute_soiling_factor(reflectance_data=reflectance_data)

    anchor = _nominal_reflectance_anchor(reflectance_data, f, helios.nominal_reflectance)
    np.testing.assert_allclose(anchor * model.helios.soiling_factor[f][:, 0], rho0, rtol=1e-10)


def test_compute_soiling_factor_falls_back_to_scalar_when_fields_missing():
    """A bare reflectance_data stub with only .rho0 (matching the synthetic
    fixtures in test_horizontal_impaction.py) must reproduce the flat-scalar
    formula exactly -- the backward-compatibility guarantee this design depends on."""
    f = 0
    rho0 = np.array([0.90, 0.80])
    nominal_reflectance_scalar = 0.95
    inc_ref_factor = np.float64(2.0)  # a plain Python float lacks .squeeze(), unlike every real caller's value

    helios = types.SimpleNamespace(
        tilt={f: np.zeros((2, 3))},
        delta_soiled_area={f: np.zeros((2, 3))},
        inc_ref_factor={f: inc_ref_factor},
        nominal_reflectance=nominal_reflectance_scalar,
    )
    model = _fit_stub(helios)
    reflectance_data = types.SimpleNamespace(rho0={f: rho0})  # no nominal_reflectance/rho0_index

    model.compute_soiling_factor(reflectance_data=reflectance_data)

    expected_cumulative_soil0 = (1 - rho0 / nominal_reflectance_scalar) / inc_ref_factor
    expected_col0 = 1 - expected_cumulative_soil0 * inc_ref_factor
    np.testing.assert_allclose(model.helios.soiling_factor[f][:, 0], expected_col0, rtol=1e-10)
    np.testing.assert_allclose(nominal_reflectance_scalar * model.helios.soiling_factor[f][:, 0], rho0, rtol=1e-10)


def test_compute_soiling_factor_clean_start_when_reflectance_data_none():
    f = 0
    helios = types.SimpleNamespace(
        tilt={f: np.zeros((2, 3))}, delta_soiled_area={f: np.full((2, 3), 0.01)}, inc_ref_factor={f: np.float64(2.0)}, nominal_reflectance=0.95
    )
    model = _fit_stub(helios)
    model.compute_soiling_factor(reflectance_data=None)

    expected_col0 = 1 - 0.01 * 2.0
    np.testing.assert_allclose(model.helios.soiling_factor[f][:, 0], expected_col0, rtol=1e-10)


def test_sse_uses_nominal_reflectance_not_rho0_squared():
    """Regression test for the _sse fix: predicted reflectance must be
    nominal_reflectance * soiling_factor, not rho0 * soiling_factor (which
    double-counts rho0, since soiling_factor's own initial condition is already
    anchored to rho0 via compute_soiling_factor's cumulative_soil0)."""
    f = 0
    rho0 = np.array([0.90])
    nominal_reflectance = np.array([[0.94], [0.93]])  # 2 measurement times, 1 mirror
    rho0_index = np.array([0])
    inc_ref_factor = np.float64(2.0)  # a plain Python float lacks .squeeze(), unlike every real caller's value

    helios = types.SimpleNamespace(
        tilt={f: np.zeros((1, 2))},
        delta_soiled_area={f: np.zeros((1, 2))},
        inc_ref_factor={f: inc_ref_factor},
        nominal_reflectance=0.95,
        soiling_factor={},
    )
    model = _fit_stub(helios)
    model.update_model_parameters = lambda params: None
    model.predict_soiling_factor = lambda simulation_inputs, reflectance_data=None, verbose=True: model.compute_soiling_factor(
        reflectance_data=reflectance_data
    )

    reflectance_data = types.SimpleNamespace(
        rho0={f: rho0},
        nominal_reflectance={f: nominal_reflectance},
        rho0_index={f: rho0_index},
        prediction_indices={f: [0, 1]},
        average={f: np.array([[0.90], [0.891]])},
    )
    simulation_inputs = types.SimpleNamespace(time={})  # empty: _check_keys' loop never executes

    sse = model._sse(params=None, simulation_inputs=simulation_inputs, reflectance_data=reflectance_data)

    # Independently re-derive the expected prediction using nominal_reflectance
    # (per measurement time), NOT rho0 -- confirms the fix (the old r0=rho0 bug
    # would give rho0**2/anchor instead).
    anchor = nominal_reflectance[rho0_index[0], 0]
    soiling_factor_col0 = rho0[0] / anchor  # constant over time: delta_soiled_area is zero
    predicted = nominal_reflectance[:, 0] * soiling_factor_col0
    expected_sse = np.sum((predicted - reflectance_data.average[f][:, 0]) ** 2)
    assert sse == pytest.approx(expected_sse, rel=1e-10)
