import numpy as np
import pytest

from mpc_spacecraft.utilities.array_view_generic import ArrayView
from mpc_spacecraft.utilities.utils import FullSimState, SensorRigidBodyState


def test_sensor_rigid_body_state_supports_underscore_bias_aliases() -> None:
    state = SensorRigidBodyState.zeros()

    state.gyro_bias[:] = [1.0, 2.0, 3.0]
    state.accel_bias[:] = [4.0, 5.0, 6.0]

    np.testing.assert_allclose(state.sensor_bias.gyro, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state.sensor_bias.accel, [4.0, 5.0, 6.0])


def test_full_sim_state_supports_underscore_bias_aliases() -> None:
    state = FullSimState.zeros()

    state.gyro_bias[:] = [1.0, 2.0, 3.0]
    state.accel_bias[:] = [4.0, 5.0, 6.0]

    np.testing.assert_allclose(state.sensor_rigid_body.gyro_bias, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state.sensor_rigid_body.accel_bias, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(state.sensor_rigid_body.sensor_bias.gyro, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state.sensor_rigid_body.sensor_bias.accel, [4.0, 5.0, 6.0])


def test_full_sim_state_bias_alias_slices_match_expanded_paths() -> None:
    assert FullSimState.slice_of("gyro_bias") == FullSimState.slice_of(
        "sensor_rigid_body.sensor_bias.gyro"
    )
    assert FullSimState.slice_of("accel_bias") == FullSimState.slice_of(
        "sensor_rigid_body.sensor_bias.accel"
    )


def test_invalid_alias_path_fails_during_class_creation() -> None:
    with pytest.raises(ValueError, match="Invalid alias 'broken_alias' on BrokenState"):

        class BrokenState(ArrayView):
            __fields__ = [("value", 3)]
            __aliases__ = {"broken_alias": "missing.value"}
