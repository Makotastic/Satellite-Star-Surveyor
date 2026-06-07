"""Performance metrics for controller evaluation."""

import numpy as np
from typing import Dict, Optional


def compute_tracking_error(
    states: np.ndarray,
    references: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute tracking error metrics.
    
    Args:
        states: State trajectory [N, state_dim]
        references: Reference trajectory [N, state_dim]
        weights: Weight vector for different state components (optional)
        
    Returns:
        Dictionary of error metrics
    """
    errors = states - references
    
    if weights is not None:
        errors = errors * weights
    
    # Mean absolute error
    mae = np.mean(np.abs(errors), axis=0)
    
    # Root mean squared error
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    
    # Maximum error
    max_error = np.max(np.abs(errors), axis=0)
    
    # Integral of absolute error
    iae = np.sum(np.abs(errors), axis=0)
    
    # Integral of squared error
    ise = np.sum(errors**2, axis=0)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'iae': iae,
        'ise': ise,
        'mae_total': np.mean(mae),
        'rmse_total': np.sqrt(np.mean(rmse**2))
    }


def compute_control_effort(
    controls: np.ndarray,
    dt: float
) -> Dict[str, float]:
    """
    Compute control effort metrics.
    
    Args:
        controls: Control trajectory [N, control_dim]
        dt: Timestep
        
    Returns:
        Dictionary of control effort metrics
    """
    # Total variation (sum of absolute control changes)
    control_changes = np.diff(controls, axis=0)
    total_variation = np.sum(np.abs(control_changes), axis=0)
    
    # Average control magnitude
    avg_magnitude = np.mean(np.linalg.norm(controls, axis=1))
    
    # Maximum control magnitude
    max_magnitude = np.max(np.linalg.norm(controls, axis=1))
    
    # Integral of control squared
    control_squared = np.sum(controls**2, axis=0) * dt
    
    # Control smoothness (variance of control changes)
    smoothness = np.var(control_changes, axis=0)
    
    return {
        'total_variation': total_variation,
        'avg_magnitude': avg_magnitude,
        'max_magnitude': max_magnitude,
        'control_squared': control_squared,
        'smoothness': smoothness,
        'total_variation_sum': np.sum(total_variation),
        'control_squared_sum': np.sum(control_squared)
    }


def compute_stability_metrics(
    states: np.ndarray,
    equilibrium: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute stability metrics.
    
    Args:
        states: State trajectory [N, state_dim]
        equilibrium: Equilibrium state (default: zeros)
        
    Returns:
        Dictionary of stability metrics
    """
    if equilibrium is None:
        equilibrium = np.zeros(states.shape[1])
    
    # Deviation from equilibrium
    deviations = states - equilibrium
    
    # Convergence rate (exponential fit)
    norms = np.linalg.norm(deviations, axis=1)
    
    # Check if converging
    is_converging = norms[-1] < norms[0]
    
    # Settling time (time to reach 2% of initial error)
    threshold = 0.02 * norms[0]
    settling_idx = np.where(norms < threshold)[0]
    settling_time = settling_idx[0] if len(settling_idx) > 0 else len(norms)
    
    # Overshoot
    max_deviation = np.max(norms)
    overshoot = (max_deviation - norms[0]) / norms[0] if norms[0] > 0 else 0
    
    # Final error
    final_error = norms[-1]
    
    return {
        'is_converging': is_converging,
        'settling_time': settling_time,
        'overshoot': overshoot,
        'final_error': final_error,
        'max_deviation': max_deviation,
        'initial_error': norms[0]
    }


def compute_quaternion_error(
    q: np.ndarray,
    q_ref: np.ndarray
) -> float:
    """
    Compute quaternion tracking error.
    
    Args:
        q: Current quaternion [w, x, y, z]
        q_ref: Reference quaternion [w, x, y, z]
        
    Returns:
        Angular error in radians
    """
    # Ensure unit quaternions
    q = q / np.linalg.norm(q)
    q_ref = q_ref / np.linalg.norm(q_ref)
    
    # Compute dot product
    dot = np.abs(np.dot(q, q_ref))
    dot = np.clip(dot, -1.0, 1.0)
    
    # Angular error
    error = 2 * np.arccos(dot)
    
    return error


def compute_quaternion_trajectory_errors(
    states: np.ndarray,
    goal_quat: np.ndarray
) -> np.ndarray:
    """
    Compute quaternion angle errors for entire trajectory.
    
    Args:
        states: State trajectory [N, 7] with quaternions in first 4 columns
        goal_quat: Goal quaternion [4]
        
    Returns:
        Array of angle errors [N]
    """
    quat_errors = []
    for s in states:
        error = compute_quaternion_error(s[:4], goal_quat)
        quat_errors.append(error)
    return np.array(quat_errors)


def compute_velocity_errors(
    states: np.ndarray,
    goal_omega: np.ndarray = np.zeros(3)
) -> np.ndarray:
    """
    Compute angular velocity errors for trajectory.
    
    Args:
        states: State trajectory [N, 7] with omega in last 3 columns
        goal_omega: Goal angular velocity [3] (default zero)
        
    Returns:
        Array of velocity error norms [N]
    """
    omega_errors = []
    for s in states:
        omega_error = np.linalg.norm(s[4:] - goal_omega)
        omega_errors.append(omega_error)
    return np.array(omega_errors)


def compute_performance_index(
    states: np.ndarray,
    controls: np.ndarray,
    references: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    dt: float
) -> float:
    """
    Compute quadratic performance index (cost).
    
    J = sum_k [(x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k] * dt
    
    Args:
        states: State trajectory [N, state_dim]
        controls: Control trajectory [N-1, control_dim]
        references: Reference trajectory [N, state_dim]
        Q: State cost matrix
        R: Control cost matrix
        dt: Timestep
        
    Returns:
        Total cost
    """
    total_cost = 0.0
    
    # State cost
    for i in range(len(states) - 1):
        x_error = states[i] - references[i]

        state_cost = x_error.T @ Q @ x_error
        
        control_cost = controls[i].T @ R @ controls[i]
        
        total_cost += (state_cost + control_cost) * dt
    
    # Terminal cost
    x_error_final = states[-1] - references[-1]
    total_cost += x_error_final.T @ Q @ x_error_final
    
    return total_cost


def compare_controllers(
    results: Dict[str, Dict[str, np.ndarray]],
    dt: float,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple controller results.
    
    Args:
        results: Dictionary of {controller_name: {states, controls, references}}
        dt: Timestep
        Q: State cost matrix (optional)
        R: Control cost matrix (optional)
        dynamics: Optional dynamics model for error state computation
        
    Returns:
        Dictionary of {controller_name: metrics}
    """
    comparison = {}
    
    for name, data in results.items():
        states = data['states']
        controls = data['controls']
        references = data.get('references', np.zeros_like(states))
        
        metrics = {}
        
        # Tracking error
        tracking = compute_tracking_error(states, references)
        metrics.update({f'tracking_{k}': v for k, v in tracking.items()})
        
        # Control effort
        effort = compute_control_effort(controls, dt)
        metrics.update({f'effort_{k}': v for k, v in effort.items()})
        
        # Stability
        stability = compute_stability_metrics(states)
        metrics.update({f'stability_{k}': v for k, v in stability.items()})
        
        comparison[name] = metrics
    
    return comparison