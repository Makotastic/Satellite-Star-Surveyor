"""Stability analysis utilities."""

import numpy as np
from scipy.linalg import eig
from typing import Tuple, Dict, Optional


def analyze_closed_loop_stability(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    discrete: bool = True
) -> Dict[str, any]:
    """
    Analyze closed-loop stability for linear system with feedback.
    
    Args:
        A: State matrix
        B: Input matrix
        K: Feedback gain matrix
        discrete: Whether system is discrete-time
        
    Returns:
        Dictionary containing stability analysis results
    """
    # Closed-loop system matrix
    A_cl = A - B @ K
    
    # Compute eigenvalues
    eigenvalues, eigenvectors = eig(A_cl)
    
    # Check stability
    if discrete:
        # Discrete-time: eigenvalues must be inside unit circle
        is_stable = np.all(np.abs(eigenvalues) < 1.0)
        margin = 1.0 - np.max(np.abs(eigenvalues))
    else:
        # Continuous-time: eigenvalues must have negative real parts
        is_stable = np.all(np.real(eigenvalues) < 0.0)
        margin = -np.max(np.real(eigenvalues))
    
    # Dominant eigenvalue (slowest mode)
    if discrete:
        dominant_idx = np.argmax(np.abs(eigenvalues))
    else:
        dominant_idx = np.argmax(np.real(eigenvalues))
    
    dominant_eigenvalue = eigenvalues[dominant_idx]
    
    # Time constant (for continuous) or settling time (for discrete)
    if discrete:
        if np.abs(dominant_eigenvalue) > 0:
            settling_time = -1 / np.log(np.abs(dominant_eigenvalue))
        else:
            settling_time = 0
    else:
        if np.real(dominant_eigenvalue) < 0:
            time_constant = -1 / np.real(dominant_eigenvalue)
        else:
            time_constant = np.inf
    
    results = {
        'is_stable': is_stable,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'stability_margin': margin,
        'dominant_eigenvalue': dominant_eigenvalue,
        'A_cl': A_cl
    }
    
    if discrete:
        results['settling_time'] = settling_time
    else:
        results['time_constant'] = time_constant
    
    return results


def compute_lyapunov_function(
    P: np.ndarray,
    state: np.ndarray
) -> float:
    """
    Compute Lyapunov function value V(x) = x^T P x.
    
    Args:
        P: Positive definite matrix (from Riccati equation)
        state: State vector
        
    Returns:
        Lyapunov function value
    """
    return float(state.T @ P @ state)


def verify_lyapunov_stability(
    A: np.ndarray,
    P: np.ndarray,
    Q: Optional[np.ndarray] = None
) -> Dict[str, bool]:
    """
    Verify Lyapunov stability conditions.
    
    For continuous-time: A^T P + P A + Q = 0
    For discrete-time: A^T P A - P + Q = 0
    
    Args:
        A: System matrix
        P: Lyapunov matrix candidate
        Q: Positive definite matrix (optional, uses identity if None)
        
    Returns:
        Dictionary with verification results
    """
    if Q is None:
        Q = np.eye(A.shape[0])
    
    # Check if P is positive definite
    eigenvalues_P = np.linalg.eigvals(P)
    is_P_positive_definite = np.all(eigenvalues_P > 0)
    
    # Check if Q is positive definite
    eigenvalues_Q = np.linalg.eigvals(Q)
    is_Q_positive_definite = np.all(eigenvalues_Q > 0)
    
    # Check Lyapunov equation (continuous-time)
    lyapunov_residual = A.T @ P + P @ A + Q
    lyapunov_error = np.linalg.norm(lyapunov_residual, 'fro')
    satisfies_lyapunov = lyapunov_error < 1e-6
    
    return {
        'is_P_positive_definite': is_P_positive_definite,
        'is_Q_positive_definite': is_Q_positive_definite,
        'satisfies_lyapunov_equation': satisfies_lyapunov,
        'lyapunov_error': lyapunov_error,
        'min_eigenvalue_P': np.min(eigenvalues_P),
        'max_eigenvalue_P': np.max(eigenvalues_P)
    }


def compute_region_of_attraction(
    P: np.ndarray,
    level_set: float = 1.0,
    num_samples: int = 1000
) -> np.ndarray:
    """
    Estimate region of attraction using level sets of Lyapunov function.
    
    Args:
        P: Lyapunov matrix
        level_set: Level set value
        num_samples: Number of samples for boundary
        
    Returns:
        Boundary points of region of attraction
    """
    n = P.shape[0]
    
    # Generate random directions
    directions = np.random.randn(num_samples, n)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    # Find boundary points
    boundary_points = []
    
    for direction in directions:
        # Binary search for boundary
        alpha_min, alpha_max = 0.0, 10.0
        
        for _ in range(20):  # Binary search iterations
            alpha = (alpha_min + alpha_max) / 2
            x = alpha * direction
            V = compute_lyapunov_function(P, x)
            
            if V < level_set:
                alpha_min = alpha
            else:
                alpha_max = alpha
        
        boundary_points.append(alpha * direction)
    
    return np.array(boundary_points)


def compute_controllability_matrix(
    A: np.ndarray,
    B: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """
    Compute controllability matrix and check controllability.
    
    Args:
        A: State matrix (n x n)
        B: Input matrix (n x m)
        
    Returns:
        Controllability matrix, is_controllable
    """
    n = A.shape[0]
    
    # Build controllability matrix [B, AB, A^2B, ..., A^(n-1)B]
    C = B.copy()
    
    for i in range(1, n):
        C = np.hstack([C, np.linalg.matrix_power(A, i) @ B])
    
    # Check rank
    rank = np.linalg.matrix_rank(C)
    is_controllable = (rank == n)
    
    return C, is_controllable


def compute_observability_matrix(
    A: np.ndarray,
    C: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """
    Compute observability matrix and check observability.
    
    Args:
        A: State matrix (n x n)
        C: Output matrix (p x n)
        
    Returns:
        Observability matrix, is_observable
    """
    n = A.shape[0]
    
    # Build observability matrix [C; CA; CA^2; ...; CA^(n-1)]
    O = C.copy()
    
    for i in range(1, n):
        O = np.vstack([O, C @ np.linalg.matrix_power(A, i)])
    
    # Check rank
    rank = np.linalg.matrix_rank(O)
    is_observable = (rank == n)
    
    return O, is_observable