import numpy as np
import pytest

from mpc_spacecraft.guidance.sun_tracker import AstropySunDirectionModel


@pytest.mark.unit
def test_sun_dir_eci_is_unit_vector():
    model = AstropySunDirectionModel()
    sun_dir = model.sun_dir_eci("2026-03-20T00:00:00")

    assert sun_dir.shape == (3,)
    assert np.isclose(np.linalg.norm(sun_dir), 1.0, atol=1e-12)


@pytest.mark.unit
def test_sun_dir_eci_roughly_opposite_across_half_year():
    model = AstropySunDirectionModel()
    sun_march = model.sun_dir_eci("2026-03-20T00:00:00")
    sun_sept = model.sun_dir_eci("2026-09-22T00:00:00")

    # Around half a year apart, heliocentric direction from Earth is roughly opposite.
    assert float(np.dot(sun_march, sun_sept)) < -0.95

