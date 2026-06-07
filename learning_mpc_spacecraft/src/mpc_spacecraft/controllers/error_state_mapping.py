"""Controller-facing attitude/rotation error-state mapping service."""

import numpy as np
import quaternion as qu

from mpc_spacecraft.utilities.utils import (
    FloatArray,
    Quat,
    RotationErrorState,
    RotationState,
)
from mpc_spacecraft.utilities.array_view_generic import BatchArrayView


class ErrorStateMappingService:
    """Maps between full rotational state and 6D error coordinates."""

    def quaternion_error(self, q: Quat, q_ref: Quat) -> FloatArray:
        """Compute small-angle attitude error vector with respect to a reference."""
        q_err = q_ref.conjugate() * q
        q_err = q_err.normalized()
        return 2.0 * qu.as_vector_part(q_err)

    def state_error(
        self,
        state: RotationState,
        state_ref: RotationState,
    ) -> RotationErrorState:
        """Map full state and reference to 6D error [delta_theta, delta_omega]."""
        q = qu.quaternion(*state.quat).normalized()
        omega = state.omega

        q_ref = qu.quaternion(*state_ref.quat).normalized()
        omega_ref = state_ref.omega

        delta_theta = self.quaternion_error(q, q_ref)
        delta_omega = omega - omega_ref

        error_state = RotationErrorState.zeros()
        error_state.error_angle[:] = delta_theta
        error_state.omega[:] = delta_omega
        return error_state

    def state_error_batch(
        self,
        states: BatchArrayView[RotationState],
        states_ref: BatchArrayView[RotationState],
    ) -> BatchArrayView[RotationErrorState]:
        """Vectorized 6D error mapping for batched states."""
        q = qu.as_quat_array(states.quat)
        q = q / np.abs(q)
        omega = states.omega

        q_ref = qu.as_quat_array(states_ref.quat)
        q_ref = q_ref / np.abs(q_ref)
        omega_ref = states_ref.omega

        q_err = q_ref.conjugate() * q
        q_err = q_err / np.abs(q_err)

        delta_theta = 2.0 * qu.as_vector_part(q_err)
        delta_omega = omega - omega_ref

        x = RotationErrorState.batch_zeros(states.count)
        x.error_angle[:] = delta_theta
        x.omega[:] = delta_omega

        return x

    def state_from_error(
        self,
        delta_x: RotationErrorState,
        state_ref: RotationState,
    ) -> RotationState:
        """Reconstruct full [q, omega] state from 6D error and reference."""
        delta_theta = delta_x.error_angle
        delta_omega = delta_x.omega

        q_ref = qu.quaternion(*state_ref.quat).normalized()
        omega_ref = state_ref.omega

        dq = qu.quaternion(1.0, *(0.5 * delta_theta)).normalized()
        q = (q_ref * dq).normalized()
        omega = omega_ref + delta_omega

        state = RotationState.zeros()
        state.quat[:] = qu.as_float_array(q)
        state.omega[:] = omega
        return state

    def state_from_error_batch(
        self,
        delta_xs: BatchArrayView[RotationErrorState],
        states_ref: BatchArrayView[RotationState],
    ) -> BatchArrayView[RotationState]:
        """Vectorized reconstruction of full state from batched error states."""
        delta_theta = delta_xs.error_angle
        delta_omega = delta_xs.omega

        q_ref = qu.as_quat_array(states_ref.quat)
        q_ref = q_ref / np.abs(q_ref)
        omega_ref = states_ref.omega

        dq_float = np.concatenate(
            [
                np.ones((delta_xs.count, 1), dtype=float),
                0.5 * delta_theta,
            ],
            axis=1,
        ).astype(float, copy=False)
        dq = qu.as_quat_array(dq_float)
        dq = dq / np.abs(dq)
        q = q_ref * dq
        q = q / np.abs(q)
        omega = omega_ref + delta_omega

        state = RotationState.batch_zeros(delta_xs.count)
        state.quat[:] = qu.as_float_array(q)
        state.omega[:] = omega

        return state
