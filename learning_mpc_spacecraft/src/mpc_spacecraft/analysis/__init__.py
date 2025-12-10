"""Analysis module for performance evaluation and metrics."""

from .metrics import (
    compute_tracking_error, compute_control_effort, compute_stability_metrics,
    compute_quaternion_trajectory_errors, compute_velocity_errors,
    compute_performance_index, compare_controllers
)
from .plotting import (
    plot_state_trajectory, plot_control_inputs, plot_comparison,
    plot_quaternion_trajectory, plot_angular_velocity,
    plot_performance_comparison, plot_error_comparison,
    plot_all_comparisons
)
from .stability import analyze_closed_loop_stability, compute_lyapunov_function

__all__ = [
    "compute_tracking_error",
    "compute_control_effort",
    "compute_stability_metrics",
    "compute_quaternion_trajectory_errors",
    "compute_velocity_errors",
    "compute_performance_index",
    "compare_controllers",
    "plot_state_trajectory",
    "plot_control_inputs",
    "plot_comparison",
    "plot_quaternion_trajectory",
    "plot_angular_velocity",
    "plot_performance_comparison",
    "plot_error_comparison",
    "analyze_closed_loop_stability",
    "compute_lyapunov_function",
    "plot_all_comparisons"
]