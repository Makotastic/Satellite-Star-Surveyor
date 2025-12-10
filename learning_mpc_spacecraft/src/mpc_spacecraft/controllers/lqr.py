"""LQR controller for spacecraft attitude control."""

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from typing import Tuple, Optional


class LQRController:
    """
    Linear Quadratic Regulator (LQR) controller for spacecraft attitude control.
    
    Computes optimal feedback gain K such that u = -K(x - x_ref).
    """
    
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        discrete: bool = True
    ):
        """
        Initialize LQR controller.
        
        Args:
            A: State transition matrix (n x n)
            B: Control input matrix (n x m)
            Q: State cost matrix (n x n)
            R: Control cost matrix (m x m)
            discrete: If True, solve discrete-time LQR; otherwise continuous-time
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.discrete = discrete
        
        # Compute LQR gain
        self.K, self.S = self.compute_lqr_gain()
    
    def compute_lqr_gain(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LQR feedback gain matrix.
        
        Returns:
            K: Feedback gain matrix (m x n)
            S: Solution to Riccati equation (n x n)
        """
        if self.discrete:
            # Discrete-time Algebraic Riccati Equation (DARE)
            S = solve_discrete_are(self.A, self.B, self.Q, self.R)
            K = np.linalg.inv(self.R + self.B.T @ S @ self.B) @ (self.B.T @ S @ self.A)
        else:
            # Continuous-time Algebraic Riccati Equation (CARE)
            S = solve_continuous_are(self.A, self.B, self.Q, self.R)
            K = np.linalg.inv(self.R) @ (self.B.T @ S)
        
        return K, S
    
    def compute_control(
        self,
        state: np.ndarray,
        state_ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute control input using LQR feedback law.
        
        Args:
            state: Current state vector
            state_ref: Reference state vector (default: zeros)
            
        Returns:
            Control input u = -K(x - x_ref)
        """
        if state_ref is None:
            state_ref = np.zeros_like(state)
        
        control = -self.K @ (state - state_ref)
        
        return control
    
    def compute_control_with_saturation(
        self,
        state: np.ndarray,
        state_ref: Optional[np.ndarray] = None,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute control input with saturation limits.
        
        Args:
            state: Current state vector
            state_ref: Reference state vector
            u_min: Minimum control values
            u_max: Maximum control values
            
        Returns:
            Saturated control input
        """
        control = self.compute_control(state, state_ref)
        
        if u_min is not None:
            control = np.maximum(control, u_min)
        if u_max is not None:
            control = np.minimum(control, u_max)
        
        return control
    
    def get_closed_loop_eigenvalues(self) -> np.ndarray:
        """
        Compute eigenvalues of the closed-loop system.
        
        Returns:
            Eigenvalues of (A - BK)
        """
        A_cl = self.A - self.B @ self.K
        eigenvalues = np.linalg.eigvals(A_cl)
        return eigenvalues
    
    def is_stable(self) -> bool:
        """
        Check if the closed-loop system is stable.
        
        Returns:
            True if all eigenvalues have negative real parts (continuous)
            or magnitude < 1 (discrete)
        """
        eigenvalues = self.get_closed_loop_eigenvalues()
        
        if self.discrete:
            # Discrete-time: all eigenvalues must be inside unit circle
            return np.all(np.abs(eigenvalues) < 1.0)
        else:
            # Continuous-time: all eigenvalues must have negative real parts
            return np.all(np.real(eigenvalues) < 0.0)
    
    def compute_cost_to_go(self, state: np.ndarray) -> float:
        """
        Compute the cost-to-go (value function) for a given state.
        
        V(x) = x^T S x
        
        Args:
            state: State vector
            
        Returns:
            Cost-to-go value
        """
        return float(state.T @ self.S @ state)