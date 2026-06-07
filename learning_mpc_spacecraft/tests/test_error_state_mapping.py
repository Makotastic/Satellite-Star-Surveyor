import numpy as np
import pytest
import quaternion as qu

from mpc_spacecraft.controllers.error_state_mapping import ErrorStateMappingService
from mpc_spacecraft.utilities.utils import RotationErrorState, RotationState


@pytest.fixture
def mapping() -> ErrorStateMappingService:
    return ErrorStateMappingService()


@pytest.fixture
def ref_state() -> RotationState:
    return RotationState.zeros()


@pytest.fixture
def small_error_state(ref_state: RotationState) -> RotationState:
    state = ref_state.copy()
    dq = qu.quaternion(1.0, 0.005, 0.01, 0.015).normalized()
    q_ref = qu.quaternion(*ref_state.quat)
    q = q_ref * dq
    state.quat[:] = qu.as_float_array(q)
    state.omega[:] = np.array([0.01, 0.02, 0.0])
    return state


@pytest.mark.unit
def test_state_error_roundtrip(mapping: ErrorStateMappingService, ref_state, small_error_state):
    delta_x = mapping.state_error(small_error_state, ref_state)
    reconstructed = mapping.state_from_error(delta_x, ref_state)
    np.testing.assert_allclose(reconstructed.quat, small_error_state.quat, rtol=1e-3, atol=1e-8)
    np.testing.assert_allclose(reconstructed.omega, small_error_state.omega, rtol=1e-8, atol=1e-10)


@pytest.mark.unit
def test_batch_matches_singleton(mapping: ErrorStateMappingService, ref_state, small_error_state):
    states = RotationState.batch_zeros(2)
    refs = RotationState.batch_zeros(2)
    states.data[0] = small_error_state.data
    states.data[1] = ref_state.data
    refs.data[0] = ref_state.data
    refs.data[1] = ref_state.data

    batch_err = mapping.state_error_batch(states, refs)
    single0 = mapping.state_error(states[0], refs[0])
    single1 = mapping.state_error(states[1], refs[1])

    np.testing.assert_allclose(batch_err[0].data, single0.data)
    np.testing.assert_allclose(batch_err[1].data, single1.data)


@pytest.mark.unit
def test_batch_reconstruction_matches_singleton(mapping: ErrorStateMappingService, ref_state, small_error_state):
    delta0 = mapping.state_error(small_error_state, ref_state)
    delta1 = mapping.state_error(ref_state, ref_state)

    deltas = RotationErrorState.batch_zeros(2)
    refs = RotationState.batch_zeros(2)
    deltas.data[0] = delta0.data
    deltas.data[1] = delta1.data
    refs.data[0] = ref_state.data
    refs.data[1] = ref_state.data

    batch_states = mapping.state_from_error_batch(deltas, refs)

    np.testing.assert_allclose(batch_states[0].data, mapping.state_from_error(delta0, ref_state).data, atol=1e-10)
    np.testing.assert_allclose(batch_states[1].data, mapping.state_from_error(delta1, ref_state).data, atol=1e-10)
