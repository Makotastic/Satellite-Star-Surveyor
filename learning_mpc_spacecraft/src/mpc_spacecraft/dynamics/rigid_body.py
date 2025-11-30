"""Rigid body dynamics for spacecraft attitude control."""

import numpy as np
import quaternion
from typing import Tuple, Optional


class SpacecraftDynamics:
    """
    Spacecraft rigid body dynamics with quaternion attitude representation.
    
    State vector: [q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z]
    - q: quaternion (4D)
    - omega: angular velocity in body frame (3D)
    
    Control input: [tau_x, tau_y, tau_z]
    - tau: torque in body frame (3D)
    """
    
    def __init__(self, inertia: np.ndarray, dt: float = 0.1):
        """
        Initialize spacecraft dynamics.
        
        Args:
            inertia: 3x3 inertia matrix (kg*m^2)
            dt: Discretization timestep (s)
        """
        self.inertia = inertia
        self.inertia_inv = np.linalg.inv(inertia)
        self.dt = dt
        
    def continuous_dynamics(
        self, 
        state: np.ndarray, 
        control: np.ndarray,
        disturbance: Optional[np.ndarray] = None
    ) -> np.ndarray:
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
        q = np.quaternion(*state[:4])
        omega = state[4:]
        
        # Normalize quaternion
        q = q.normalized()
        
        # Quaternion kinematics: dq/dt = 0.5 * q * omega_body
        q_dot = 0.5 * q * quaternion.from_vector_part(omega)
        
        # Total torque (control + disturbance)
        total_torque = control.copy()
        if disturbance is not None:
            total_torque += disturbance
        
        # Euler's equation: d_omega/dt = I^-1 @ (tau - omega x (I @ omega))
        omega_dot = self.inertia_inv @ (total_torque - np.cross(omega, self.inertia @ omega))
        
        return np.concatenate([quaternion.as_float_array(q_dot), omega_dot])
    
    def discrete_dynamics_rk4(
        self,
        state: np.ndarray,
        control: np.ndarray,
        disturbance: Optional[np.ndarray] = None
    ) -> np.ndarray:
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
        
        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize quaternion
        q = np.quaternion(*next_state[:4]).normalized()
        next_state[:4] = quaternion.as_float_array(q)
        
        return next_state
    
    def discrete_dynamics_euler(
        self,
        state: np.ndarray,
        control: np.ndarray,
        disturbance: Optional[np.ndarray] = None
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
        q = np.quaternion(*next_state[:4]).normalized()
        next_state[:4] = quaternion.as_float_array(q)
        
        return next_state
    
    def quaternion_error(self, q: np.quaternion, q_ref: np.quaternion) -> np.ndarray:
        """
        Compute small-angle error vector delta_theta such that
        q ≈ q_ref * [1, 0.5*delta_theta]^T near the reference.
        """
        # Error quaternion: q_err = q_ref* * q
        q_err = q_ref.conjugate() * q
        q_err = q_err.normalized()
        
        # For small error, q_err ≈ [1, 0.5*delta_theta]
        delta_theta = 2.0 * quaternion.as_vector_part(q_err)
        return delta_theta


    def state_error(self, state: np.ndarray, state_ref: np.ndarray) -> np.ndarray:
        """
        Map full state and reference state to 6D error state [delta_theta, delta_omega].
        """
        q = np.quaternion(*state[:4]).normalized()
        omega = state[4:]
        
        q_ref = np.quaternion(*state_ref[:4]).normalized()
        omega_ref = state_ref[4:]
        
        delta_theta = self.quaternion_error(q, q_ref)
        delta_omega = omega - omega_ref
        
        return np.concatenate([delta_theta, delta_omega])

    def state_from_error(
    self,
    delta_x: np.ndarray,
    state_ref: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct full state [q, omega] from error state [delta_theta, delta_omega]
        and reference state.
        """
        delta_theta = delta_x[:3]
        delta_omega = delta_x[3:]
        
        q_ref = np.quaternion(*state_ref[:4]).normalized()
        omega_ref = state_ref[4:]
        
        # Small-error quaternion: dq ≈ [1, 0.5*delta_theta]
        dq = np.quaternion(1.0, *(0.5 * delta_theta))
        dq = dq.normalized()
        
        # Compose attitude: q = q_ref * dq
        q = q_ref * dq
        q = q.normalized()
        
        omega = omega_ref + delta_omega
        
        state = np.empty(7)
        state[:4] = quaternion.as_float_array(q)
        state[4:] = omega
        return state

    
    def linearize(
    self,
    state_ref: np.ndarray,
    control_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize error dynamics around a reference trajectory point.

        Returns A, B matrices for: d(delta_x)/dt = A*delta_x + B*delta_u

        where delta_x = [delta_theta, delta_omega].

        Args:
            state_ref: Reference state [q_ref, omega_ref]
            control_ref: Reference control

        Returns:
            A: 6x6 state transition matrix
            B: 6x3 control input matrix
        """
        epsilon = 1e-6

        n = 6  # error state dimension
        m = 3  # control dimension

        A = np.zeros((n, n))
        B = np.zeros((n, m))

        # Helper: computes error-state derivative for given delta_x, delta_u
        def f_err(delta_x: np.ndarray, delta_u: np.ndarray) -> np.ndarray:
            # Build actual state and control from errors
            state = self.state_from_error(delta_x, state_ref)
            control = control_ref + delta_u

            # Continuous dynamics on full state
            state_dot = self.continuous_dynamics(state, control)

            # Map state_dot to error derivative:
            # We approximate this numerically via a small step, to avoid deriving it analytically.

            dt_small = 1e-4  # small internal step for numerical derivative mapping

            # Step forward in full state:
            state_next = state + dt_small * state_dot

            # Compute error at t and t+dt_small:
            err_now = self.state_error(state, state_ref)
            err_next = self.state_error(state_next, state_ref)

            # Approximate error derivative:
            err_dot = (err_next - err_now) / dt_small
            return err_dot

        # A matrix: df_err / d(delta_x)
        delta_x0 = np.zeros(n)
        delta_u0 = np.zeros(m)

        for i in range(n):
            dx_p = delta_x0.copy()
            dx_m = delta_x0.copy()
            dx_p[i] += epsilon
            dx_m[i] -= epsilon

            f_p = f_err(dx_p, delta_u0)
            f_m = f_err(dx_m, delta_u0)

            A[:, i] = (f_p - f_m) / (2 * epsilon)

        # B matrix: df_err / d(delta_u)
        for i in range(m):
            du_p = delta_u0.copy()
            du_m = delta_u0.copy()
            du_p[i] += epsilon
            du_m[i] -= epsilon

            f_p = f_err(delta_x0, du_p)
            f_m = f_err(delta_x0, du_m)

            B[:, i] = (f_p - f_m) / (2 * epsilon)

        return A, B
    
    def discretize_linear_system(
        self,
        A: np.ndarray,
        B: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize linear system using matrix exponential.
        
        Args:
            A: Continuous-time state matrix
            B: Continuous-time input matrix
            
        Returns:
            Ad: Discrete-time state matrix
            Bd: Discrete-time input matrix
        """
        from scipy.linalg import expm
        
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