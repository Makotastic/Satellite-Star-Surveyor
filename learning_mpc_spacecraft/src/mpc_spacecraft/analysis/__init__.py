"""Analysis module for performance evaluation and metrics."""

from .metrics import compute_tracking_error, compute_control_effort, compute_stability_metrics
from .plotting import plot_state_trajectory, plot_control_inputs, plot_comparison
from .stability import analyze_closed_loop_stability, compute_lyapunov_function

__all__ = [
    "compute_tracking_error",
    "compute_control_effort",
    "compute_stability_metrics",
    "plot_state_trajectory",
    "plot_control_inputs",
    "plot_comparison",
    "analyze_closed_loop_stability",
    "compute_lyapunov_function",
]