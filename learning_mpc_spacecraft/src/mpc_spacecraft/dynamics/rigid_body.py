"""Rigid-body spacecraft dynamics with rotational and translational branches."""

from typing import cast

import numpy as np
import quaternion as qu
from scipy.linalg import expm

from mpc_spacecraft.utilities.utils import (
    FloatArray,
    RotState,
    TransState,
    Vec3,
    I3,
    skew,
    z3,
    ROT_STATE_SLICES,
    TRANS_STATE_SLICES,
)

IDX_STATE_QUAT = ROT_STATE_SLICES.quat
IDX_STATE_OMEGA = ROT_STATE_SLICES.omega
IDX_TRANS_POS = TRANS_STATE_SLICES.position
IDX_TRANS_VEL = TRANS_STATE_SLICES.velocity

# @TODO: Fix quaternion Integrator. RK4 does not natively work well with RK4, Use Lie algebra methods.


class SpacecraftDynamics:
    """
    Spacecraft dynamics with separate rotation and translation integrations.

    Rotation state vector: [q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z]
    Translation state vector: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]

    Rotation input: [tau_x, tau_y, tau_z]
    Translation input: [acc_x, acc_y, acc_z]
    """

    def __init__(self, inertia: FloatArray, dt: float, mass: float = 1.0):
        """
        Initialize spacecraft dynamics.

        Args:
            inertia: 3x3 inertia matrix (kg*m^2).
            dt: Discretization timestep (s)
            mass: Spacecraft mass (kg) for translational branch.
        """
        self._inertia = inertia
        self._inertia_inv = cast(FloatArray, np.linalg.inv(inertia))
        self._mass = float(mass)
        self.dt = dt

    @property
    def inertia(self) -> FloatArray:
        return self._inertia

    @property
    def inertia_inv(self) -> FloatArray:
        return self._inertia_inv

    @property
    def mass(self) -> float:
        return self._mass

    def set_inertia(self, inertia: FloatArray) -> None:
        """Set inertia and keep inverse inertia synchronized."""
        self._inertia = inertia
        self._inertia_inv = cast(FloatArray, np.linalg.inv(inertia))

    def _continuous_dynamics_rotation(
        self,
        state: RotState,
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
        q = qu.quaternion(*state[IDX_STATE_QUAT])
        omega = state[IDX_STATE_OMEGA]

        # Normalize quaternion
        q = q.normalized()

        # Quaternion kinematics: dq/dt = 0.5 * q * omega_body
        q_dot = 0.5 * q * qu.from_vector_part(omega)

        # Total torque (control + disturbance)
        total_torque = control.copy()
        if disturbance is not None:
            total_torque += disturbance

        # Euler's equation: d_omega/dt = I^-1 @ (tau - omega x (I @ omega))
        omega_dot = self._inertia_inv @ (
            total_torque - np.cross(omega, self._inertia @ omega)
        )

        return np.concatenate([qu.as_float_array(q_dot), omega_dot])

    def _continuous_dynamics_translation(
        self,
        state: TransState,
        acceleration: Vec3,
    ) -> FloatArray:
        state_dt = np.zeros(6)
        state_dt[IDX_TRANS_POS] = state[IDX_TRANS_VEL]
        state_dt[IDX_TRANS_VEL] = acceleration
        return state_dt

    def rotational_dynamics_error_jacobian(
        self,
        state: RotState,
        input: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:

        omega = state[IDX_STATE_OMEGA]
        skew_w = skew(omega)
        skew_jw = skew(self._inertia_inv @ omega)

        dw_block = self._inertia_inv @ (skew_jw - skew_w @ self._inertia)

        A = np.block([[-skew_w, I3], [z3, dw_block]])

        B = np.block([[z3], [self._inertia_inv]])

        return A, B

    def discrete_dynamics_rk4_rotation(
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
        k1 = self._continuous_dynamics_rotation(state, control, disturbance)
        k2 = self._continuous_dynamics_rotation(
            state + 0.5 * self.dt * k1, control, disturbance
        )
        k3 = self._continuous_dynamics_rotation(
            state + 0.5 * self.dt * k2, control, disturbance
        )
        k4 = self._continuous_dynamics_rotation(
            state + self.dt * k3, control, disturbance
        )

        next_state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize quaternion
        q = qu.quaternion(*next_state[IDX_STATE_QUAT]).normalized()
        next_state[IDX_STATE_QUAT] = qu.as_float_array(q)

        return next_state

    def discrete_dynamics_rk4_translation(
        self,
        state: TransState,
        acceleration: Vec3,
    ) -> TransState:
        """Compute translational discrete-time dynamics using RK4 integration."""
        k1 = self._continuous_dynamics_translation(state, acceleration)
        k2 = self._continuous_dynamics_translation(
            state + 0.5 * self.dt * k1, acceleration
        )
        k3 = self._continuous_dynamics_translation(
            state + 0.5 * self.dt * k2, acceleration
        )
        k4 = self._continuous_dynamics_translation(state + self.dt * k3, acceleration)

        return state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

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
