"""Analysis module for performance evaluation and metrics."""

from .metrics import (
    compute_tracking_error, compute_control_effort, compute_stability_metrics,
    compute_quaternion_trajectory_errors, compute_velocity_errors,
    compute_performance_index, compare_controllers
)
from .mission_results import (
    MissionArtifactPaths,
    build_derived_timeseries,
    build_mission_analysis_artifact,
    compute_mission_summary,
    quaternion_angle_error_deg,
    save_mission_artifact,
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
    "MissionArtifactPaths",
    "build_derived_timeseries",
    "build_mission_analysis_artifact",
    "compute_mission_summary",
    "quaternion_angle_error_deg",
    "save_mission_artifact",
    "analyze_closed_loop_stability",
    "compute_lyapunov_function",
]
