"""Parity and contract tests for ErrorStateMappingService."""

import numpy as np
import pytest
import quaternion as qu

from mpc_spacecraft.controllers.error_state_mapping import ErrorStateMappingService


@pytest.fixture
def mapping() -> ErrorStateMappingService:
    return ErrorStateMappingService()


@pytest.fixture
def ref_state() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def small_error_state(ref_state: np.ndarray) -> np.ndarray:
    state = ref_state.copy()
    dq = qu.quaternion(1.0, 0.005, 0.01, 0.015).normalized()
    q_ref = qu.quaternion(*ref_state[:4])
    q = q_ref * dq
    state[:4] = qu.as_float_array(q)
    state[4:] = np.array([0.01, 0.02, 0.0])
    return state


@pytest.mark.unit
def test_state_error_roundtrip(
    mapping: ErrorStateMappingService, ref_state, small_error_state
):
    delta_x = mapping.state_error(small_error_state, ref_state)
    reconstructed = mapping.state_from_error(delta_x, ref_state)

    np.testing.assert_allclose(
        reconstructed[:4], small_error_state[:4], rtol=1e-3, atol=1e-8
    )
    np.testing.assert_allclose(
        reconstructed[4:], small_error_state[4:], rtol=1e-8, atol=1e-10
    )


@pytest.mark.unit
def test_batch_matches_singleton(
    mapping: ErrorStateMappingService, ref_state, small_error_state
):
    states = np.stack([small_error_state, ref_state], axis=0)
    refs = np.stack([ref_state, ref_state], axis=0)

    batch_err = mapping.state_error_batch(states, refs)

    single0 = mapping.state_error(states[0], refs[0])
    single1 = mapping.state_error(states[1], refs[1])

    np.testing.assert_allclose(batch_err[0], single0)
    np.testing.assert_allclose(batch_err[1], single1)


@pytest.mark.unit
def test_batch_reconstruction_matches_singleton(
    mapping: ErrorStateMappingService,
    ref_state,
    small_error_state,
):
    delta0 = mapping.state_error(small_error_state, ref_state)
    delta1 = mapping.state_error(ref_state, ref_state)

    deltas = np.stack([delta0, delta1], axis=0)
    refs = np.stack([ref_state, ref_state], axis=0)

    batch_states = mapping.state_from_error_batch(deltas, refs)

    np.testing.assert_allclose(
        batch_states[0], mapping.state_from_error(delta0, ref_state), atol=1e-10
    )
    np.testing.assert_allclose(
        batch_states[1], mapping.state_from_error(delta1, ref_state), atol=1e-10
    )
