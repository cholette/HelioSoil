"""
Unit tests for standalone helpers in heliosoil.utilities.
"""

import logging
import types

import numpy as np
import pytest

from heliosoil.utilities import (
    _is_reference_mirror_column,
    _nominal_reflectance_anchor,
    _nominal_reflectance_series,
    _parse_dust_str,
    _print_if,
    _safe_nanargmax,
    _std_errors_from_cov,
    canonical_dust_spec,
    configure_logging,
    default_training_mirrors,
    dust_cutoff,
    gravitational_settling_factor,
    logger,
    normalize_dust_name,
    parse_dust_spec,
    resolve_dust_concentration,
)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tilt,expected",
    [
        (0.0, 1.0),
        (60.0, 0.5),
        (90.0, 0.0),
        # Past vertical the face points at the ground: no settled dust, and above all
        # not a NEGATIVE deposition that would make the mirror clean itself.
        (90.1, 0.0),
        (120.0, 0.0),
        (180.0, 0.0),
    ],
)
def test_gravitational_settling_factor(tilt, expected):
    np.testing.assert_allclose(gravitational_settling_factor(tilt), expected, atol=1e-12)


def test_gravitational_settling_factor_is_never_negative_and_broadcasts():
    tilt = np.linspace(0.0, 359.5, 720).reshape(2, -1)
    factor = gravitational_settling_factor(tilt)
    assert factor.shape == tilt.shape
    assert (factor >= 0.0).all()
    # Below vertical it is exactly cos(tilt); strictly past vertical, exactly 0. (At exactly
    # 90 deg cosd returns +6e-17 rather than 0, so the clip leaves that float alone -- the
    # long-standing behaviour of cosd, and negligible against any real deposition.)
    upward = tilt < 90.0
    np.testing.assert_allclose(factor[upward], np.cos(np.radians(tilt[upward])), rtol=1e-12)
    assert (factor[(tilt > 90.0) & (tilt < 270.0)] == 0.0).all()


@pytest.mark.parametrize("dust_type,expected", [("TSP", "TSP"), ("PM17", "PM17"), ("PM10", "PM10"), ("PM18", "PM18"), ("PM2.5", "PM2_5")])
def test_parse_dust_str(dust_type, expected):
    assert _parse_dust_str(dust_type) == expected


def test_parse_dust_str_rejects_unknown():
    with pytest.raises(AssertionError):
        _parse_dust_str("dust")


def test_parse_dust_str_difference_spec():
    # A difference spec gets its own attribute on the Dust class, so the band mass is
    # cached and looked up exactly like a single measure's.
    assert _parse_dust_str("PM17-PM10") == "PM17_minus_PM10"
    assert _parse_dust_str("PM10-PM2.5") == "PM10_minus_PM2_5"


# ---------------------------------------------------------------------------
# Dust measures and specs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column,expected",
    [
        ("TSP", "TSP"),
        ("tsp", "TSP"),
        ("PM17", "PM17"),
        ("PM_TOT", "PM17"),
        ("PMTOT", "PM17"),
        ("PM10", "PM10"),
        ("pm10", "PM10"),
        ("PM2.5", "PM2.5"),
        ("PM2p5", "PM2.5"),
        ("PM2_5", "PM2.5"),
        ("PM20", "PM20"),
        (" PM 4 ", "PM4"),
        ("WindSpeed", None),
        ("Time", None),
    ],
)
def test_normalize_dust_name(column, expected):
    assert normalize_dust_name(column) == expected


def test_dust_cutoff():
    assert dust_cutoff("PM2p5") == 2.5
    assert dust_cutoff("PM10") == 10.0
    assert dust_cutoff("TSP") == np.inf
    assert dust_cutoff("PM17") == 17.0
    with pytest.raises(ValueError):
        dust_cutoff("WindSpeed")


def test_parse_dust_spec_single_and_difference():
    assert parse_dust_spec("pm2p5") == ("PM2.5", None)
    assert parse_dust_spec("PM17-PM10") == ("PM17", "PM10")
    assert parse_dust_spec(" pm10 - pm2_5 ") == ("PM10", "PM2.5")
    assert canonical_dust_spec("pm17 - pm10") == "PM17-PM10"


def test_parse_dust_spec_rejects_malformed():
    with pytest.raises(ValueError, match="Unrecognized dust measure"):
        parse_dust_spec("PM17-humidity")
    with pytest.raises(ValueError, match="Unrecognized dust spec"):
        parse_dust_spec("PM17-PM10-PM2.5")
    # The larger cutoff must come first: a difference has to enclose a size band.
    with pytest.raises(ValueError, match="encloses no size band"):
        parse_dust_spec("PM2.5-PM10")
    with pytest.raises(ValueError, match="encloses no size band"):
        parse_dust_spec("PM10-PM10")


def test_resolve_dust_concentration_single_and_difference():
    channels = {"PM2.5": np.array([1.0, 2.0]), "PM10": np.array([4.0, 7.0]), "PM17": np.array([10.0, 12.0])}

    concentration, spec = resolve_dust_concentration(channels, "pm10")
    np.testing.assert_allclose(concentration, [4.0, 7.0])
    assert spec == "PM10"

    concentration, spec = resolve_dust_concentration(channels, "PM17-PM10")
    np.testing.assert_allclose(concentration, [6.0, 5.0])
    assert spec == "PM17-PM10"


def test_resolve_dust_concentration_missing_or_unrecognized():
    channels = {"PM10": np.array([4.0, 7.0])}
    assert resolve_dust_concentration(channels, "PM2.5") == (None, None)  # channel absent
    assert resolve_dust_concentration(channels, "PM17-PM10") == (None, None)  # one side absent
    assert resolve_dust_concentration(channels, "WindSpeed") == (None, None)  # not a dust spec
    assert resolve_dust_concentration(None, "PM10") == (None, None)  # no channels at all


def test_resolve_dust_concentration_warns_on_negative_band(caplog):
    # Measurement noise can make the coarse fraction come out negative; that is
    # reported (it would make a timestep's deposition negative), never clipped away.
    channels = {"PM10": np.array([4.0, 7.0]), "PM17": np.array([10.0, 6.0])}
    with caplog.at_level(logging.WARNING, logger="heliosoil"):
        concentration, _ = resolve_dust_concentration(channels, "PM17-PM10")
    np.testing.assert_allclose(concentration, [6.0, -1.0])
    assert "1 of 2 samples are negative" in caplog.text


def test_default_training_mirrors_wind_variance_uses_all_mirrors():
    mirrors = ["ON_M1_T00", "OE_M2_T85", "OS_M2_T30", "OW_M4_T05"]
    assert default_training_mirrors(mirrors, uses_wind_variance=True) == mirrors


def test_default_training_mirrors_no_wind_variance_picks_zero_tilt():
    mirrors = ["OE_M4_T30", "ON_M1_T00", "OS_M2_T60"]
    assert default_training_mirrors(mirrors, uses_wind_variance=False) == ["ON_M1_T00"]


def test_default_training_mirrors_picks_minimum_tilt_when_no_zero_present():
    mirrors = ["OE_M4_T30", "ON_M3_T85", "OS_M2_T60"]
    assert default_training_mirrors(mirrors, uses_wind_variance=False) == ["OE_M4_T30"]


def test_default_training_mirrors_handles_single_digit_tilt_token():
    # Some datasets use "_T0" instead of "_T00" for the zero-tilt mirror, and may
    # have more than one mirror at the minimum tilt.
    mirrors = ["ON_M1_T0", "ON_M2_T30", "OS_M1_T0", "OS_M4_T90"]
    assert default_training_mirrors(mirrors, uses_wind_variance=False) == ["ON_M1_T0"]


def test_default_training_mirrors_falls_back_to_first_when_unparseable(caplog):
    mirrors = ["mirror_A", "mirror_B"]
    with caplog.at_level(logging.WARNING, logger="heliosoil"):
        result = default_training_mirrors(mirrors, uses_wind_variance=False)
    assert result == ["mirror_A"]
    assert "could not parse a tilt" in caplog.text


# --------------------------------------------------------------------------- #
# Logging / output control
# --------------------------------------------------------------------------- #


def test_print_if_emits_at_info_when_verbose(caplog):
    with caplog.at_level(logging.INFO, logger="heliosoil"):
        _print_if("hello world", True)
    assert "hello world" in caplog.text


def test_print_if_silent_when_not_verbose(caplog):
    with caplog.at_level(logging.INFO, logger="heliosoil"):
        _print_if("should not appear", False)
    assert "should not appear" not in caplog.text


def test_configure_logging_level_controls_output(caplog):
    """A single configure_logging call governs all output: at WARNING the routine
    INFO chatter is suppressed while genuine warnings still pass through."""
    previous_level = logger.level
    configure_logging(logging.WARNING)
    try:
        caplog.set_level(logging.DEBUG)  # capture handler accepts everything
        _print_if("info chatter", True)  # emitted at INFO, below WARNING
        logger.warning("real problem")
        assert "info chatter" not in caplog.text
        assert "real problem" in caplog.text
    finally:
        logger.setLevel(previous_level)


def test_configure_logging_is_idempotent():
    configure_logging(logging.INFO)
    n_handlers = len(logger.handlers)
    configure_logging(logging.WARNING)
    assert len(logger.handlers) == n_handlers
    logger.setLevel(logging.INFO)


def test_std_errors_from_cov_surfaces_non_psd(recwarn, caplog):
    # A non-PSD covariance (negative diagonal) yields NaN. It must NOT emit the
    # cryptic NumPy "invalid value encountered in sqrt" RuntimeWarning, but instead
    # surface a clear, log-level-controllable diagnostic.
    cov = np.array([[-1.0, 0.0], [0.0, 4.0]])
    with caplog.at_level(logging.WARNING, logger="heliosoil"):
        s = _std_errors_from_cov(cov)
    assert not recwarn.list  # no raw RuntimeWarning
    assert "non-positive-definite" in caplog.text
    assert np.isnan(s[0])
    assert s[1] == pytest.approx(2.0)


def test_std_errors_from_cov_quiet_when_psd(recwarn, caplog):
    # A well-conditioned covariance produces no warning of any kind.
    cov = np.array([[4.0, 0.0], [0.0, 9.0]])
    with caplog.at_level(logging.WARNING, logger="heliosoil"):
        s = _std_errors_from_cov(cov)
    assert not recwarn.list
    assert caplog.text == ""
    assert s == pytest.approx([2.0, 3.0])


# --------------------------------------------------------------------------- #
# Reference-mirror handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,expected",
    [
        ("OW_M1_T0_Ref", True),
        ("ONE_M1_T00_ref", True),
        ("SomeMirror_REF", True),
        ("ON_M1_T00", False),
        ("Time", False),
        ("Reference", False),  # contains "ref" but has no "_ref" suffix
    ],
)
def test_is_reference_mirror_column(name, expected):
    assert _is_reference_mirror_column(name) == expected


def test_safe_nanargmax_normal_column():
    a = np.array([[1.0, np.nan], [3.0, 2.0], [2.0, 5.0]])
    np.testing.assert_array_equal(_safe_nanargmax(a), [1, 2])


def test_safe_nanargmax_leading_nan_column():
    # Mirrors the real carwarp data shape: a mirror added mid-campaign has NaN
    # rows before its first valid reading.
    a = np.array([[np.nan, 1.0], [np.nan, 3.0], [5.0, 2.0]])
    np.testing.assert_array_equal(_safe_nanargmax(a), [2, 1])


def test_safe_nanargmax_all_nan_column_returns_zero_without_raising(recwarn):
    a = np.array([[np.nan, 1.0], [np.nan, 3.0]])
    result = _safe_nanargmax(a)
    assert result[0] == 0
    assert result[1] == 1


def test_nominal_reflectance_series_falls_back_when_reflectance_data_is_none():
    fallback = 0.95
    assert _nominal_reflectance_series(None, 0, fallback) == fallback


def test_nominal_reflectance_series_falls_back_when_file_missing():
    fallback = 0.95
    reflectance_data = types.SimpleNamespace(nominal_reflectance={1: np.ones((3, 2))})
    assert _nominal_reflectance_series(reflectance_data, 0, fallback) == fallback


def test_nominal_reflectance_series_returns_stored_array_when_present():
    series = np.array([[0.90, 0.91], [0.89, 0.90]])
    reflectance_data = types.SimpleNamespace(nominal_reflectance={0: series})
    result = _nominal_reflectance_series(reflectance_data, 0, 0.95)
    np.testing.assert_array_equal(result, series)


def test_nominal_reflectance_anchor_falls_back_when_missing():
    fallback = 0.95
    assert _nominal_reflectance_anchor(None, 0, fallback) == fallback
    reflectance_data = types.SimpleNamespace(nominal_reflectance={})
    assert _nominal_reflectance_anchor(reflectance_data, 0, fallback) == fallback


def test_nominal_reflectance_anchor_gathers_at_rho0_index():
    series = np.array([[0.90, 0.80], [0.95, 0.85], [0.93, 0.87]])
    reflectance_data = types.SimpleNamespace(nominal_reflectance={0: series}, rho0_index={0: np.array([1, 2])})
    result = _nominal_reflectance_anchor(reflectance_data, 0, 0.95)
    np.testing.assert_array_equal(result, [0.95, 0.87])
