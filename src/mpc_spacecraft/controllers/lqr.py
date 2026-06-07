"""LQR controller for spacecraft attitude control."""

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from mpc_spacecraft.utilities.utils import FloatArray


class LQRController:
    """
    Linear Quadratic Regulator (LQR) controller for spacecraft attitude control.

    Computes optimal feedback gain K such that u = -K(x - x_ref).
    """

    def __init__(
        self,
        A: FloatArray,
        B: FloatArray,
        Q: FloatArray,
        R: FloatArray,
        u_min: FloatArray | None = None,
        u_max: FloatArray | None = None,
        discrete: bool = True,
    ):
        """
        Initialize LQR controller.

        Args:
            A: State transition matrix (n x n)
            B: Control input matrix (n x m)
            Q: State cost matrix (n x n)
            R: Control cost matrix (m x m)
            u_min: Minimum control values
            u_max: Maximum control values
            discrete: If True, solve discrete-time LQR; otherwise continuous-time
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.u_min = u_min
        self.u_max = u_max
        self.discrete = discrete

        # Compute LQR gain
        self.K, self.S = self.compute_lqr_gain()

    def compute_lqr_gain(self) -> tuple[FloatArray, FloatArray]:
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
        self, state: FloatArray, state_ref: FloatArray | None = None
    ) -> FloatArray:
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

        if self.u_min is not None:
            control = np.maximum(control, self.u_min)
        if self.u_max is not None:
            control = np.minimum(control, self.u_max)

        return control

    def get_closed_loop_eigenvalues(self) -> FloatArray:
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

    def compute_cost_to_go(self, state: FloatArray) -> float:
        """
        Compute the cost-to-go (value function) for a given state.

        V(x) = x^T S x

        Args:
            state: State vector

        Returns:
            Cost-to-go value
        """
        return float(state.T @ self.S @ state)
