"""Controller-facing attitude/rotation error-state mapping service."""

import numpy as np
import quaternion as qu

from mpc_spacecraft.utilities.utils import (
    FloatArray,
    Quat,
    RotErrState,
    RotState,
    ROT_STATE_SLICES,
    ROT_ERROR_SLICES,
)

IDX_STATE_QUAT = ROT_STATE_SLICES.quat
IDX_STATE_OMEGA = ROT_STATE_SLICES.omega

IDX_ERR_THETA = ROT_ERROR_SLICES.error_angle
IDX_ERR_OMEGA = ROT_ERROR_SLICES.omega


class ErrorStateMappingService:
    """Maps between full rotational state and 6D error coordinates."""

    def quaternion_error(self, q: Quat, q_ref: Quat) -> FloatArray:
        """Compute small-angle attitude error vector with respect to a reference."""
        q_err = q_ref.conjugate() * q
        q_err = q_err.normalized()
        return 2.0 * qu.as_vector_part(q_err)

    def state_error(self, state: RotState, state_ref: RotState) -> RotErrState:
        """Map full state and reference to 6D error [delta_theta, delta_omega]."""
        q = qu.quaternion(*state[IDX_STATE_QUAT]).normalized()
        omega = state[IDX_STATE_OMEGA]

        q_ref = qu.quaternion(*state_ref[IDX_STATE_QUAT]).normalized()
        omega_ref = state_ref[IDX_STATE_OMEGA]

        delta_theta = self.quaternion_error(q, q_ref)
        delta_omega = omega - omega_ref

        return np.concatenate([delta_theta, delta_omega])

    def state_error_batch(self, states: RotState, states_ref: RotState) -> RotErrState:
        """Vectorized 6D error mapping for batched states."""
        q = qu.as_quat_array(states[:, IDX_STATE_QUAT])
        q = q / np.abs(q)
        omega = states[:, IDX_STATE_OMEGA]

        q_ref = qu.as_quat_array(states_ref[:, IDX_STATE_QUAT])
        q_ref = q_ref / np.abs(q_ref)
        omega_ref = states_ref[:, IDX_STATE_OMEGA]

        q_err = q_ref.conjugate() * q
        q_err = q_err / np.abs(q_err)

        delta_theta = 2.0 * qu.as_vector_part(q_err)
        delta_omega = omega - omega_ref

        return np.concatenate([delta_theta, delta_omega], axis=1)

    def state_from_error(self, delta_x: RotErrState, state_ref: RotState) -> RotState:
        """Reconstruct full [q, omega] state from 6D error and reference."""
        delta_theta = delta_x[IDX_ERR_THETA]
        delta_omega = delta_x[IDX_ERR_OMEGA]

        q_ref = qu.quaternion(*state_ref[IDX_STATE_QUAT]).normalized()
        omega_ref = state_ref[IDX_STATE_OMEGA]

        dq = qu.quaternion(1.0, *(0.5 * delta_theta)).normalized()
        q = (q_ref * dq).normalized()
        omega = omega_ref + delta_omega

        state = np.empty(7)
        state[IDX_STATE_QUAT] = qu.as_float_array(q)
        state[IDX_STATE_OMEGA] = omega
        return state

    def state_from_error_batch(
        self, delta_xs: RotErrState, states_ref: RotState
    ) -> RotState:
        """Vectorized reconstruction of full state from batched error states."""
        delta_theta = delta_xs[:, IDX_ERR_THETA]
        delta_omega = delta_xs[:, IDX_ERR_OMEGA]

        q_ref = qu.as_quat_array(states_ref[:, IDX_STATE_QUAT])
        q_ref = q_ref / np.abs(q_ref)
        omega_ref = states_ref[:, IDX_STATE_OMEGA]

        dq_float = np.concatenate(
            [
                np.ones((delta_xs.shape[0], 1), dtype=float),
                0.5 * delta_theta,
            ],
            axis=1,
        ).astype(float, copy=False)
        dq = qu.as_quat_array(dq_float)
        dq = dq / np.abs(dq)
        q = q_ref * dq
        q = q / np.abs(q)
        omega = omega_ref + delta_omega

        state = np.empty((delta_xs.shape[0], 7), dtype=float)
        state[:, IDX_STATE_QUAT] = qu.as_float_array(q)
        state[:, IDX_STATE_OMEGA] = omega
        return state
