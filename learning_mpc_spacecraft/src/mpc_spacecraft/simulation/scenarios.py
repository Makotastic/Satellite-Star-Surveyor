"""Scenario generation for testing controllers."""

import numpy as np
import quaternion
from typing import Callable, Optional, Tuple


class ScenarioGenerator:
    """Generate test scenarios for spacecraft control."""
    
    @staticmethod
    def rest_to_rest(
        duration: float,
        dt: float,
        q_target: Optional[np.ndarray] = None
    ) -> Callable:
        """
        Rest-to-rest maneuver: rotate from initial to target attitude.
        
        Args:
            duration: Maneuver duration
            dt: Timestep
            q_target: Target quaternion (default: identity)
            
        Returns:
            Reference trajectory function
        """
        if q_target is None:
            q_target = np.array([1.0, 0.0, 0.0, 0.0])
        
        q_target = np.quaternion(*q_target).normalized()
        q_target = np.quaternion.as_float_array(q_target)
        
        def reference_trajectory(t: float) -> np.ndarray:
            """Return reference state at time t."""
            # Target: desired quaternion with zero angular velocity
            return np.concatenate([q_target, np.zeros(3)])
        
        return reference_trajectory
    
    @staticmethod
    def slew_maneuver(
        q_initial: np.ndarray,
        q_final: np.ndarray,
        duration: float,
        dt: float
    ) -> Callable:
        """
        Slew maneuver with smooth trajectory.
        
        Args:
            q_initial: Initial quaternion
            q_final: Final quaternion
            duration: Maneuver duration
            
        Returns:
            Reference trajectory function
        """

        q_initial = np.quaternion(*q_initial).normalized()

        q_final = np.quaternion(*q_final).normalized()
        
        def reference_trajectory(t: float) -> np.ndarray:
            """Return reference state at time t using SLERP."""
            
            q_ref = np.quaternion.slerp(q_initial, q_final, 0, duration, t).normalized()

            q_ref = np.quaternion.as_float_array(q_ref)
            
            # Zero angular velocity reference
            omega_ref = np.zeros(3)
            
            return np.concatenate([q_ref, omega_ref])
        
        return reference_trajectory
    
    @staticmethod
    def constant_rate_spin(
        omega_target: np.ndarray,
        duration: float,
        dt: float
    ) -> Callable:
        """
        Constant angular velocity spin.
        
        Args:
            omega_target: Target angular velocity [rad/s]
            duration: Duration
            dt: Timestep
            
        Returns:
            Reference trajectory function
        """
        def reference_trajectory(t: float) -> np.ndarray:
            """Return reference state at time t."""
            # Integrate angular velocity to get quaternion

            q_ref = np.quaternion.integrate_angular_velocity(omega_target, 0, t)
            
            return np.concatenate([q_ref, omega_target])
        
        return reference_trajectory
    
    @staticmethod
    def tracking_trajectory(
        waypoints: list,
        times: list,
        dt: float
    ) -> Callable:
        """
        Track a sequence of waypoints.
        
        Args:
            waypoints: List of target states
            times: List of times for each waypoint
            dt: Timestep
            
        Returns:
            Reference trajectory function
        """
        waypoints = np.array(waypoints)
        times = np.array(times)
        
        def reference_trajectory(t: float) -> np.ndarray:
            """Return reference state at time t using interpolation."""
            if t <= times[0]:
                return waypoints[0]
            elif t >= times[-1]:
                return waypoints[-1]
            else:
                # Find surrounding waypoints
                idx = np.searchsorted(times, t)
                t0, t1 = times[idx - 1], times[idx]
                x0, x1 = waypoints[idx - 1], waypoints[idx]
                
                x_q_0 = np.quaternion(*x0[:4]).normalized()
                x_q_1 = np.quaternion(*x1[:4]).normalized()
                x_q_ref = np.quaternion.slerp(x_q_0, x_q_1, t0, t1, t).normalized()
                x_q_ref = np.quaternion.as_float_array(x_q_ref).squeeze()

                x_w_0 = x0[4:]
                x_w_1 = x1[4:]
                alpha = (t - t0) / (t1 - t0)
                x_w_ref = (1 - alpha) * x_w_0 + alpha * x_w_1
                
                return np.concatenate([x_q_ref, x_w_ref])
        
        return reference_trajectory


def create_reference_trajectory(
    scenario_type: str,
    **kwargs
) -> Callable:
    """
    Factory function to create reference trajectories.
    
    Args:
        scenario_type: Type of scenario ('rest_to_rest', 'slew', 'spin', 'tracking')
        **kwargs: Scenario-specific parameters
        
    Returns:
        Reference trajectory function
    """
    generator = ScenarioGenerator()
    
    if scenario_type == 'rest_to_rest':
        return generator.rest_to_rest(**kwargs)
    elif scenario_type == 'slew':
        return generator.slew_maneuver(**kwargs)
    elif scenario_type == 'spin':
        return generator.constant_rate_spin(**kwargs)
    elif scenario_type == 'tracking':
        return generator.tracking_trajectory(**kwargs)
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")


def generate_random_initial_conditions(
    num_samples: int,
    max_angle: float = np.pi,
    max_omega: float = 0.1,
    seed: Optional[int] = None
) -> list:
    """
    Generate random initial conditions for testing.
    
    Args:
        num_samples: Number of samples to generate
        max_angle: Maximum rotation angle from identity (rad)
        max_omega: Maximum angular velocity magnitude (rad/s)
        seed: Random seed
        
    Returns:
        List of initial state vectors
    """
    if seed is not None:
        np.random.seed(seed)
    
    initial_conditions = []
    
    for _ in range(num_samples):
        # Random rotation axis
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # Random angle
        angle = np.random.uniform(0, max_angle)

        # Quaternion from axis-angle
        q = np.array([
            np.cos(angle / 2),
            axis[0] * np.sin(angle / 2),
            axis[1] * np.sin(angle / 2),
            axis[2] * np.sin(angle / 2)
        ])
        
        # Random angular velocity
        omega = np.random.uniform(-max_omega, max_omega, 3)
        
        # Combine into state
        x0 = np.concatenate([q, omega])
        initial_conditions.append(x0)
    
    return initial_conditions