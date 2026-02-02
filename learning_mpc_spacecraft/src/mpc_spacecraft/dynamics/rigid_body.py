"""Rigid body dynamics for spacecraft attitude control."""

from typing import cast

import numpy as np
import quaternion as qu
from scipy.linalg import expm

from mpc_spacecraft.utilities.utils import FloatArray, Quat, skew


IDX_QUAT = slice(0, 4)
IDX_OMEGA = slice(4, 7)


class SpacecraftDynamics:
    """
    Spacecraft rigid body dynamics with quaternion attitude representation.

    State vector: [q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z]
    - q: quaternion (4D)
    - omega: angular velocity in body frame (3D)

    Control input: [tau_x, tau_y, tau_z]
    - tau: torque in body frame (3D)
    """

    def __init__(self, inertia: FloatArray, dt: float):
        """
        Initialize spacecraft dynamics.

        Args:
            inertia: 3x3 inertia matrix (kg*m^2)
            dt: Discretization timestep (s)
        """
        self.inertia = inertia
        self.inertia_inv = cast(FloatArray, np.linalg.inv(inertia))
        self.dt = dt

    def continuous_dynamics(
        self,
        state: FloatArray,
        control: FloatArray,
        disturbance: FloatArray | None = None,
    ) -> FloatArray:
        """
        Compute continuous-time dynamics: dx/dt = f(x, u).

        Args:
            state: State vector [q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z]
            control: Control input [tau_x, tau_y, tau_z]
            disturbance: Optional disturbance torque [tau_d_x, tau_d_y, tau_d_z]

        Returns:
            State derivative [dq/dt, d_omega/dt]
        """
        # Extract state components
        q = qu.quaternion(*state[IDX_QUAT])
        omega = state[4:]

        # Normalize quaternion
        q = q.normalized()

        # Quaternion kinematics: dq/dt = 0.5 * q * omega_body
        q_dot = 0.5 * q * qu.from_vector_part(omega)

        # Total torque (control + disturbance)
        total_torque = control.copy()
        if disturbance is not None:
            total_torque += disturbance

        # Euler's equation: d_omega/dt = I^-1 @ (tau - omega x (I @ omega))
        omega_dot = self.inertia_inv @ (
            total_torque - np.cross(omega, self.inertia @ omega)
        )

        return np.concatenate([qu.as_float_array(q_dot), omega_dot])

    def dynamics_error_jacobians(
        self, state: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        I3 = np.eye(3)
        z3 = np.zeros((3))

        omega = state[IDX_OMEGA]
        skew_w = skew(omega)
        skew_jw = skew(self.inertia_inv @ omega)

        dw_block = self.inertia_inv @ (skew_jw - skew_w @ self.inertia)

        A = np.block([[-skew_w, I3], [z3, dw_block]])

        B = np.block([[z3], [self.inertia_inv]])

        return A, B

    def discrete_dynamics_rk4(
        self,
        state: FloatArray,
        control: FloatArray,
        disturbance: FloatArray | None = None,
    ) -> FloatArray:
        """
        Compute discrete-time dynamics using RK4 integration.

        Args:
            state: Current state
            control: Control input
            disturbance: Optional disturbance torque

        Returns:
            Next state
        """
        # RK4 integration
        k1 = self.continuous_dynamics(state, control, disturbance)
        k2 = self.continuous_dynamics(state + 0.5 * self.dt * k1, control, disturbance)
        k3 = self.continuous_dynamics(state + 0.5 * self.dt * k2, control, disturbance)
        k4 = self.continuous_dynamics(state + self.dt * k3, control, disturbance)

        next_state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize quaternion
        q = qu.quaternion(*next_state[IDX_QUAT]).normalized()
        next_state[IDX_QUAT] = qu.as_float_array(q)

        return next_state

    def discrete_dynamics_euler(
        self,
        state: np.ndarray,
        control: np.ndarray,
        disturbance: FloatArray | None = None,
    ) -> np.ndarray:
        """
        Compute discrete-time dynamics using forward Euler integration.

        Args:
            state: Current state
            control: Control input
            disturbance: Optional disturbance torque

        Returns:
            Next state
        """
        state_dot = self.continuous_dynamics(state, control, disturbance)
        next_state = state + self.dt * state_dot

        # Normalize quaternion
        q = qu.quaternion(*next_state[IDX_QUAT]).normalized()
        next_state[IDX_QUAT] = qu.as_float_array(q)

        return next_state

    def quaternion_error(self, q: Quat, q_ref: Quat) -> FloatArray:
        """
        Compute small-angle error vector delta_theta such that
        q ≈ q_ref * [1, 0.5*delta_theta]^T near the reference.
        """
        # Error quaternion: q_err = q_ref* * q
        q_err = q_ref.conjugate() * q
        q_err = q_err.normalized()

        # For small error, q_err ≈ [1, 0.5*delta_theta]
        delta_theta = 2.0 * qu.as_vector_part(q_err)
        return delta_theta

    def state_error(self, state: FloatArray, state_ref: FloatArray) -> FloatArray:
        """
        Map full state and reference state to 6D error state [delta_theta, delta_omega].
        """
        q = qu.quaternion(*state[IDX_QUAT]).normalized()
        omega = state[IDX_OMEGA]

        q_ref = qu.quaternion(*state_ref[IDX_QUAT]).normalized()
        omega_ref = state_ref[IDX_OMEGA]

        delta_theta = self.quaternion_error(q, q_ref)
        delta_omega = omega - omega_ref

        return np.concatenate([delta_theta, delta_omega])

    def state_from_error(
        self, delta_x: FloatArray, state_ref: FloatArray
    ) -> FloatArray:
        """
        Reconstruct full state [q, omega] from error state [delta_theta, delta_omega]
        and reference state.
        """
        delta_theta = delta_x[:3]
        delta_omega = delta_x[3:]

        q_ref = qu.quaternion(*state_ref[IDX_QUAT]).normalized()
        omega_ref = state_ref[IDX_OMEGA]

        # Small-error quaternion: dq ≈ [1, 0.5*delta_theta]
        dq = qu.quaternion(1.0, *(0.5 * delta_theta))
        dq = dq.normalized()

        # Compose attitude: q = q_ref * dq
        q = q_ref * dq
        q = q.normalized()

        omega = omega_ref + delta_omega

        state = np.empty(7)
        state[IDX_QUAT] = qu.as_float_array(q)
        state[IDX_OMEGA] = omega
        return state

    def discretize_linear_system(
        self, A: FloatArray, B: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """
        Discretize linear system using matrix exponential.

        Args:
            A: Continuous-time state matrix
            B: Continuous-time input matrix

        Returns:
            Ad: Discrete-time state matrix
            Bd: Discrete-time input matrix
        """

        n = A.shape[0]
        m = B.shape[1]

        # Build augmented matrix
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A * self.dt
        M[:n, n:] = B * self.dt

        # Matrix exponential
        exp_M = expm(M)

        Ad = exp_M[:n, :n]
        Bd = exp_M[:n, n:]

        return Ad, Bd
