"""
Unit tests for regression_performance_stats (heliosoil.paper_specific_utilities).

These lock in the metric's two-quantity design: MBE/MAE/RMSE are computed on the *daily
soiling rate* (the drop in soiling factor per day), while R2 stays a *reflectance*
goodness-of-fit. Lightweight SimpleNamespace stubs stand in for the model/reflectance-data
objects; only the attributes the function reads are populated.
"""

import types

import numpy as np
import pytest

from heliosoil.paper_specific_utilities import regression_performance_stats


def _make(soiling_factor, average, r0=0.9, days=(0, 1, 3)):
    """Build (model, reflectance_data) stubs for one experiment (index 0).

    soiling_factor: (n_mirror, n_time) predicted soiling factor on the sim grid.
    average:        (n_time, n_mirror) measured reflectance.
    r0:             scalar nominal (clean) reflectance fallback.
    days:           measurement times, in days from an arbitrary epoch.
    """
    n_time = np.asarray(average).shape[0]
    helios = types.SimpleNamespace(soiling_factor={0: np.asarray(soiling_factor, float)}, nominal_reflectance=r0)
    model = types.SimpleNamespace(helios=helios)
    rdat = types.SimpleNamespace(
        prediction_indices={0: list(range(n_time))},
        average={0: np.asarray(average, float)},
        times={0: np.array(days, dtype="timedelta64[D]") + np.datetime64("2020-01-01")},
        nominal_reflectance={},  # empty -> _nominal_reflectance_series returns the scalar fallback
    )
    return model, rdat


def test_rate_metrics_and_reflectance_r2():
    """MBE/MAE/RMSE match a hand-computed daily soiling rate; R2 matches reflectance."""
    r0 = 0.9
    model, rdat = _make(soiling_factor=[[1.0, 0.98, 0.95]], average=[[0.9], [0.87], [0.84]], r0=r0, days=(0, 1, 3))
    got = regression_performance_stats(model, rdat, [0])

    sf_pred = np.array([1.0, 0.98, 0.95])
    sf_meas = np.array([0.9, 0.87, 0.84]) / r0
    dt = np.array([1.0, 2.0])
    resid = (-np.diff(sf_pred) / dt) - (-np.diff(sf_meas) / dt)
    assert got["MBE"] == pytest.approx(resid.mean())
    assert got["MAE"] == pytest.approx(np.abs(resid).mean())
    assert got["RMSE"] == pytest.approx(np.sqrt((resid**2).mean()))

    pred_refl, meas_refl = r0 * sf_pred, np.array([0.9, 0.87, 0.84])
    ss_res = np.sum((pred_refl - meas_refl) ** 2)
    ss_tot = np.sum((meas_refl - meas_refl.mean()) ** 2)
    assert got["R2"] == pytest.approx(1 - ss_res / ss_tot)
    assert got["N"] == 3


def test_perfect_prediction():
    """When predicted reflectance equals measured, the rate error is ~0 and R2 ~1."""
    r0 = 0.9
    sf = [[1.0, 0.98, 0.95]]
    model, rdat = _make(soiling_factor=sf, average=[[r0 * 1.0], [r0 * 0.98], [r0 * 0.95]], r0=r0)
    got = regression_performance_stats(model, rdat, [0])
    assert got["MAE"] == pytest.approx(0.0, abs=1e-12)
    assert got["RMSE"] == pytest.approx(0.0, abs=1e-12)
    assert got["R2"] == pytest.approx(1.0)


def test_rate_metrics_mask_nan_measurement():
    """A NaN measurement voids only the intervals it bounds; other mirrors still yield
    finite rate metrics (the reflectance R2 stays unmasked by design)."""
    r0 = 0.9
    # mirror 0 has clean data; mirror 1 has a NaN at the middle measurement.
    model, rdat = _make(
        soiling_factor=[[1.0, 0.98, 0.95], [1.0, 0.99, 0.97]], average=[[0.9, 0.9], [0.87, np.nan], [0.84, 0.855]], r0=r0, days=(0, 1, 3)
    )
    got = regression_performance_stats(model, rdat, [0])

    # Only mirror 0's two increments survive -> same numbers as the single-mirror case.
    sf_pred, sf_meas = np.array([1.0, 0.98, 0.95]), np.array([0.9, 0.87, 0.84]) / r0
    dt = np.array([1.0, 2.0])
    resid = (-np.diff(sf_pred) / dt) - (-np.diff(sf_meas) / dt)
    assert np.isfinite(got["MBE"]) and np.isfinite(got["MAE"]) and np.isfinite(got["RMSE"])
    assert got["MAE"] == pytest.approx(np.abs(resid).mean())


def test_all_nan_returns_nan_without_error():
    """A single-measurement-per-interval gap that leaves no finite residual yields NaN
    rate metrics rather than raising or warning."""
    model, rdat = _make(soiling_factor=[[1.0, 0.98, 0.95]], average=[[0.9], [np.nan], [0.84]])
    got = regression_performance_stats(model, rdat, [0])
    assert np.isnan(got["MBE"]) and np.isnan(got["MAE"]) and np.isnan(got["RMSE"])
