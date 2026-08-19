"""
The tilt factor in deposition terms: max(0, cos(tilt)), not cos(tilt).

A mirror past vertical presents no upward-facing area, so it collects no settled dust.
With the raw cosine it accumulated NEGATIVE soiled area -- cleaning itself in proportion
to the airborne dust -- while its deposition variance was that of a horizontal mirror.
Yadnarie's OSW_M1_T180 is exactly this case, so it is not hypothetical.

The one place the raw cosine is correct is the gravity term in the adhesion moment, where
it is a force component and MUST reverse past vertical.
"""

import numpy as np
import pytest

from heliosoil.utilities import cosd, gravitational_settling_factor


@pytest.mark.parametrize(
    "tilt,expected",
    [
        (0.0, 1.0),
        (60.0, 0.5),
        (90.0, 0.0),      # to atol; cosd(90) is 6.1e-17, not 0
        (90.1, 0.0),
        (120.0, 0.0),
        (180.0, 0.0),
    ],
)
def test_gravitational_settling_factor(tilt, expected):
    np.testing.assert_allclose(gravitational_settling_factor(tilt), expected, atol=1e-12)


def test_settling_factor_never_negative_and_broadcasts():
    tilt = np.linspace(0.0, 359.5, 720).reshape(2, -1)
    factor = gravitational_settling_factor(tilt)
    assert factor.shape == tilt.shape
    assert (factor >= 0.0).all()


def test_settling_factor_is_the_raw_cosine_below_vertical():
    """Every mirror at or below vertical must be bit-identical to the old behaviour, so
    this change moves no number on a field without a face-down mirror."""
    tilt = np.linspace(0.0, 89.9, 500)
    np.testing.assert_array_equal(gravitational_settling_factor(tilt), cosd(tilt))


def test_settling_factor_is_zero_not_negative_past_vertical():
    tilt = np.array([90.1, 120.0, 180.0, 269.9])
    assert (cosd(tilt) < 0).all()                                  # what it used to be
    np.testing.assert_array_equal(gravitational_settling_factor(tilt), 0.0)


def test_deposition_and_its_variance_use_the_same_factor():
    """The mean was signed and the variance was not, so a face-down mirror used to have a
    negative mean and a maximal variance at the same time. Both are now zero."""
    from heliosoil.base_models import ConstantMeanBase

    import inspect

    source = inspect.getsource(ConstantMeanBase.calculate_delta_soiled_area)
    assert "gravitational_settling_factor" in source
    assert "np.cos(theta)" not in source


def test_adhesion_moment_keeps_the_raw_cosine():
    """Gravity holds dust onto an upward face and pulls it off a downward one, so this
    cosine must stay signed. Guards against a well-meaning sweep clipping it too."""
    from heliosoil.base_models import PhysicalBase

    import inspect

    source = inspect.getsource(PhysicalBase.adhesion_removal)
    assert "F_gravity*cosd(" in source or "F_gravity * cosd(" in source
    assert "F_gravity*gravitational_settling_factor" not in source
