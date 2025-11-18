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
        q_dot = 0.5 * q * np.quaternion.from_vector_part(omega)
        
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
    
    def linearize(
        self,
        state_ref: np.ndarray,
        control_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize dynamics around a reference trajectory point.
        
        Returns A, B matrices for: dx = A*dx + B*du
        
        Args:
            state_ref: Reference state
            control_ref: Reference control
            
        Returns:
            A: State transition matrix (7x7)
            B: Control input matrix (7x3)
        """
        # Numerical differentiation for linearization
        epsilon = 1e-6
        
        # Compute A matrix (df/dx)
        A = np.zeros((7, 7))
        
        for i in range(7):
            state_perturbed_p = state_ref.copy()
            state_perturbed_m = state_ref.copy()
            state_perturbed_p[i] += epsilon
            state_perturbed_m[i] -= epsilon

            f_p = self.continuous_dynamics(state_perturbed_p, control_ref)
            f_m = self.continuous_dynamics(state_perturbed_m, control_ref)

            A[:, i] = (f_p - f_m) / (2 * epsilon)
                
        # Compute B matrix (df/du)
        B = np.zeros((7, 3))
        
        for i in range(3):
            control_perturbed_p = state_ref.copy()
            control_perturbed_m = state_ref.copy()
            control_perturbed_p[i] += epsilon
            control_perturbed_m[i] -= epsilon

            f_p = self.continuous_dynamics(control_perturbed_p, control_ref)
            f_m = self.continuous_dynamics(control_perturbed_m, control_ref)

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