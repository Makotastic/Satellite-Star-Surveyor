"""Rigid body dynamics for spacecraft attitude control."""

from typing import cast

import numpy as np
import quaternion as qu
from scipy.linalg import expm

from mpc_spacecraft.utilities.utils import (
    FloatArray,
    Quat,
    skew,
    RotErrState,
    RotState,
    I3,
    z3,
    jacob_r_lie,
    jacob_r_lie_inv,
    expm_so3,
    logm_so3,
)

IDX_QUAT = slice(0, 4)
IDX_OMEGA = slice(4, 7)

# @TODO: Fix quaternion Integrator. RK4 does not natively work well with RK4, Use Lie algebra methods


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
        state: RotState,
        control: FloatArray,
        disturbance: FloatArray | None = None,
    ) -> RotState:
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

    def dynamics_error_jacobian(
        self,
        state: RotState,
        input: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:

        omega = state[IDX_OMEGA]
        skew_w = skew(omega)
        skew_jw = skew(self.inertia_inv @ omega)

        dw_block = self.inertia_inv @ (skew_jw - skew_w @ self.inertia)

        A = np.block([[-skew_w, I3], [z3, dw_block]])

        B = np.block([[z3], [self.inertia_inv]])

        return A, B

    def discrete_mpc_constraint(
        self,
        x_nom_k: RotState,
        x_nom_kp1: RotState,
        u_nom_k: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:

        omega = x_nom_k[IDX_OMEGA]
        skew_w = skew(omega)
        skew_jw = skew(self.inertia_inv @ omega)

        B_omega_c = self.inertia_inv @ (skew_jw - skew_w @ self.inertia)
        D_omega_c = np.block([self.inertia_inv])

        B_omega, D_omega = self.discretize_linear_system(B_omega_c, D_omega_c)

        x_pred_kp1 = self.discrete_dynamics_rk4(x_nom_k, u_nom_k)

        c_omega = x_pred_kp1[IDX_OMEGA] - x_nom_kp1[IDX_OMEGA]

        A_phi, B_phi, c_phi = self.discrete_phi_lie_shifting_constraint(
            x_nom_k, x_nom_kp1
        )

        A_k = np.block([[A_phi, B_phi], [z3, B_omega]])

        B_k = np.block([[z3], [D_omega]])

        c_k = np.hstack((c_phi, c_omega))

        return A_k, B_k, c_k

    def discrete_phi_lie_shifting_constraint(
        self,
        x_nom_k: RotState,
        x_nom_kp1: RotState,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Computes the correctly centered linearized dynamic
        constraint for angle phi (error) from k to k + 1
        d_phi_k+1 = A @ d_phi_k + B @ d_omega_k + s_k
        """

        q_nom_k = qu.quaternion(*x_nom_k[IDX_QUAT])
        q_nom_kp1 = qu.quaternion(*x_nom_kp1[IDX_QUAT])
        omega_nom_k = x_nom_k[IDX_OMEGA]
        omega_nom_k_angle = omega_nom_k * self.dt

        R_nom_k = qu.as_rotation_matrix(q_nom_k)
        R_nom_kp1 = qu.as_rotation_matrix(q_nom_kp1)
        d_R_centers = R_nom_kp1.T @ R_nom_k

        R_w_nom_k = expm_so3(omega_nom_k_angle)

        s_k = logm_so3(d_R_centers @ R_w_nom_k)

        j_w_k = jacob_r_lie(omega_nom_k_angle)
        j_m1_s = jacob_r_lie_inv(s_k)

        A_phi = j_m1_s @ R_w_nom_k.T
        B_omega = j_m1_s @ j_w_k * self.dt

        return A_phi, B_omega, s_k

    def discrete_dynamics_rk4(
        self,
        state: RotState,
        control: FloatArray,
        disturbance: FloatArray | None = None,
    ) -> RotState:
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

    def state_error(self, state: RotState, state_ref: RotState) -> RotErrState:
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

    def state_error_batch(self, states: RotState, states_ref: RotState) -> RotErrState:
        """
        Vectorized 6D error state [delta_theta, delta_omega] for batches.

        Args:
            states: Array of shape (N, 7)
            states_ref: Array of shape (N, 7)

        Returns:
            Array of shape (N, 6)
        """
        q = qu.as_quat_array(states[:, IDX_QUAT])
        q = q / np.abs(q)
        omega = states[:, IDX_OMEGA]

        q_ref = qu.as_quat_array(states_ref[:, IDX_QUAT])
        q_ref = q_ref / np.abs(q_ref)
        omega_ref = states_ref[:, IDX_OMEGA]

        q_err = q_ref.conjugate() * q
        q_err = q_err / np.abs(q_err)

        delta_theta = 2.0 * qu.as_vector_part(q_err)
        delta_omega = omega - omega_ref

        return np.concatenate([delta_theta, delta_omega], axis=1)

    def state_from_error(self, delta_x: RotErrState, state_ref: RotState) -> RotState:
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

    def state_from_error_batch(
        self, delta_xs: RotErrState, states_ref: RotState
    ) -> RotState:
        """
        Reconstruct full state [q, omega] from batched error states and references.

        Args:
            delta_xs: Array of shape (N, 6) with [delta_theta, delta_omega]
            states_ref: Array of shape (N, 7)

        Returns:
            Array of shape (N, 7)
        """
        delta_theta = delta_xs[:, :3]
        delta_omega = delta_xs[:, 3:]

        q_ref = qu.as_quat_array(states_ref[:, IDX_QUAT]).normalized()
        omega_ref = states_ref[:, IDX_OMEGA]

        dq_float = np.concatenate(
            [
                np.ones((delta_xs.shape[0], 1), dtype=float),
                0.5 * delta_theta,
            ],
            axis=1,
        ).astype(float, copy=False)
        dq = qu.as_quat_array(dq_float).normalized()

        q = q_ref * dq
        q = q / np.abs(q)
        omega = omega_ref + delta_omega

        state = np.empty((delta_xs.shape[0], 7), dtype=float)
        state[:, IDX_QUAT] = qu.as_float_array(q)
        state[:, IDX_OMEGA] = omega
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

        Ad = cast(FloatArray, exp_M[:n, :n])
        Bd = cast(FloatArray, exp_M[:n, n:])

        # Supposedly More Robust Solution
        # I = np.eye(6)
        # Ad = np.linalg.solve(I - 0.5 * self.dt * A, I + 0.5 * self.dt * A)
        # Bd = np.linalg.solve(I - 0.5 * self.dt * A, self.dt * B)

        return Ad, Bd
