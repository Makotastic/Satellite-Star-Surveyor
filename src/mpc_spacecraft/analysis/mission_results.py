"""Closed-loop mission result metrics and artifact construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import quaternion as qu

from mpc_spacecraft.config import ClosedLoopTestConfig


COVARIANCE_LABELS = [
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "attitude_x",
    "attitude_y",
    "attitude_z",
    "gyro_bias_x",
    "gyro_bias_y",
    "gyro_bias_z",
    "accel_bias_x",
    "accel_bias_y",
    "accel_bias_z",
]


@dataclass(frozen=True)
class MissionArtifactPaths:
    """Paths produced by the report experiment artifact writer."""

    npz: Path
    summary_json: Path
    timeseries_csv: Path | None = None
    metadata_json: Path | None = None


def build_mission_analysis_artifact(
    logs: pd.DataFrame,
    targets: pd.DataFrame,
    config: ClosedLoopTestConfig,
) -> dict[str, Any]:
    """Build a stable, plot-ready mission analysis artifact.

    The returned dictionary separates dense numeric arrays from JSON-safe summary
    metadata so the same artifact can support website metrics, report plots, and
    later notebook analysis without rerunning the simulation.
    """
    derived = build_derived_timeseries(logs)
    summary = compute_mission_summary(logs, targets, config, derived)
    metadata = build_metadata(config)

    numeric_log_columns = [
        column
        for column in logs.columns
        if pd.api.types.is_numeric_dtype(logs[column]) or pd.api.types.is_bool_dtype(logs[column])
    ]
    numeric_logs = logs[numeric_log_columns].copy()
    arrays = {column: numeric_logs[column].to_numpy() for column in numeric_logs.columns}
    for key, value in derived.items():
        arrays[f"derived_{key}"] = np.asarray(value)

    timeseries = pd.concat(
        [logs.reset_index(drop=True), _derived_timeseries_dataframe(derived)],
        axis=1,
    )

    return {
        "schema_version": "mpc_spacecraft_mission_results_v1",
        "summary": summary,
        "metadata": metadata,
        "arrays": arrays,
        "targets": targets.copy(),
        "timeseries": timeseries,
    }


def build_derived_timeseries(logs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute plot-friendly derived time series from flattened closed-loop logs."""
    true_quat = logs[["true_quat_w", "true_quat_x", "true_quat_y", "true_quat_z"]].to_numpy()
    goal_quat = logs[["goal_quat_w", "goal_quat_x", "goal_quat_y", "goal_quat_z"]].to_numpy()
    estimated_quat = logs[
        ["estimated_quat_w", "estimated_quat_x", "estimated_quat_y", "estimated_quat_z"]
    ].to_numpy()

    true_position = logs[["true_position_x", "true_position_y", "true_position_z"]].to_numpy()
    estimated_position = logs[
        ["estimated_position_x", "estimated_position_y", "estimated_position_z"]
    ].to_numpy()
    true_velocity = logs[["true_velocity_x", "true_velocity_y", "true_velocity_z"]].to_numpy()
    estimated_velocity = logs[
        ["estimated_velocity_x", "estimated_velocity_y", "estimated_velocity_z"]
    ].to_numpy()
    true_omega = logs[["true_omega_x", "true_omega_y", "true_omega_z"]].to_numpy()
    estimated_omega = logs[["estimated_omega_x", "estimated_omega_y", "estimated_omega_z"]].to_numpy()
    control = logs[["control_tx", "control_ty", "control_tz"]].to_numpy()

    covariance_diag = _covariance_diag_from_logs(logs)
    covariance_sigma = np.sqrt(np.clip(covariance_diag, 0.0, np.inf))

    return {
        "goal_angle_error_deg": quaternion_angle_error_deg(true_quat, goal_quat),
        "attitude_estimation_error_deg": quaternion_angle_error_deg(true_quat, estimated_quat),
        "position_error_m": np.linalg.norm(estimated_position - true_position, axis=1),
        "velocity_error_mps": np.linalg.norm(estimated_velocity - true_velocity, axis=1),
        "angular_rate_error_radps": np.linalg.norm(estimated_omega - true_omega, axis=1),
        "true_angular_rate_norm_radps": np.linalg.norm(true_omega, axis=1),
        "estimated_angular_rate_norm_radps": np.linalg.norm(estimated_omega, axis=1),
        "torque_norm_nm": np.linalg.norm(control, axis=1),
        "mekf_covariance_diag": covariance_diag,
        "mekf_three_sigma": 3.0 * covariance_sigma,
    }


def compute_mission_summary(
    logs: pd.DataFrame,
    targets: pd.DataFrame,
    config: ClosedLoopTestConfig,
    derived: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compute website/report summary metrics for one closed-loop mission."""
    if derived is None:
        derived = build_derived_timeseries(logs)

    time = logs["time"].to_numpy(dtype=np.float64)
    sim_dt = float(config.timing.sim_dt)
    final_target_obs = _final_target_observation_times(logs, len(targets))
    req_obs = targets["req_obs_time"].to_numpy(dtype=np.float64)
    target_complete = final_target_obs >= req_obs
    observing_mask = logs["guidance_mode"].to_numpy() == "OBSERVING"
    mpc_update_mask = logs["mpc_updated"].to_numpy(dtype=bool)
    mpc_success_at_updates = logs.loc[mpc_update_mask, "mpc_solve_success"].to_numpy(dtype=bool)
    mpc_fallback_at_updates = logs.loc[mpc_update_mask, "mpc_fallback_used"].to_numpy(dtype=bool)

    pointing_error = np.asarray(derived["goal_angle_error_deg"], dtype=np.float64)
    boresight_error = logs.get("active_target_boresight_angle_deg", pd.Series(np.nan, index=logs.index)).to_numpy(
        dtype=np.float64
    )
    pointing_for_observation = boresight_error if np.isfinite(boresight_error).any() else pointing_error

    mode_counts = logs["guidance_mode"].value_counts().sort_index()
    mode_seconds = {str(mode): float(count * sim_dt) for mode, count in mode_counts.items()}
    mode_fraction = {str(mode): float(count / len(logs)) for mode, count in mode_counts.items()}

    target_index = logs["current_target_idx"].fillna(-1).to_numpy(dtype=np.int64)
    target_switch_count = int(np.count_nonzero(np.diff(target_index) != 0)) if len(target_index) > 1 else 0
    settling_times = estimate_settling_times_seconds(logs, threshold_deg=0.5)

    summary = {
        "outcome": make_outcome_sentence(
            completed=int(np.count_nonzero(target_complete)),
            total=len(targets),
            observing_pointing_max_deg=_masked_stat(pointing_for_observation, observing_mask, np.nanmax),
        ),
        "targets_completed": int(np.count_nonzero(target_complete)),
        "target_count": int(len(targets)),
        "mission_duration_s": float(time[-1] - time[0]) if len(time) else 0.0,
        "sim_dt_s": sim_dt,
        "mpc_dt_s": float(config.timing.mpc_dt),
        "observation_time_by_target_s": final_target_obs.tolist(),
        "required_observation_time_by_target_s": req_obs.tolist(),
        "target_complete": target_complete.tolist(),
        "observing_time_s": float(np.count_nonzero(observing_mask) * sim_dt),
        "target_switch_count": target_switch_count,
        "settling_time_after_switch_s": settling_times,
        "planner_mode_seconds": mode_seconds,
        "planner_mode_fraction": mode_fraction,
        "pointing_error_mean_deg": _safe_stat(pointing_error, np.nanmean),
        "pointing_error_max_deg": _safe_stat(pointing_error, np.nanmax),
        "observing_pointing_error_mean_deg": _masked_stat(pointing_for_observation, observing_mask, np.nanmean),
        "observing_pointing_error_max_deg": _masked_stat(pointing_for_observation, observing_mask, np.nanmax),
        "torque_mean_norm_nm": _safe_stat(derived["torque_norm_nm"], np.nanmean),
        "torque_max_norm_nm": _safe_stat(derived["torque_norm_nm"], np.nanmax),
        "angular_rate_max_radps": _safe_stat(derived["true_angular_rate_norm_radps"], np.nanmax),
        "attitude_estimation_error_mean_deg": _safe_stat(
            derived["attitude_estimation_error_deg"], np.nanmean
        ),
        "attitude_estimation_error_max_deg": _safe_stat(
            derived["attitude_estimation_error_deg"], np.nanmax
        ),
        "position_error_mean_m": _safe_stat(derived["position_error_m"], np.nanmean),
        "position_error_max_m": _safe_stat(derived["position_error_m"], np.nanmax),
        "velocity_error_mean_mps": _safe_stat(derived["velocity_error_mps"], np.nanmean),
        "velocity_error_max_mps": _safe_stat(derived["velocity_error_mps"], np.nanmax),
        "mpc_update_count": int(np.count_nonzero(mpc_update_mask)),
        "mpc_solve_success_count": int(np.count_nonzero(mpc_success_at_updates)),
        "mpc_fallback_count": int(np.count_nonzero(mpc_fallback_at_updates)),
        "mpc_solve_success_rate": float(np.mean(mpc_success_at_updates))
        if len(mpc_success_at_updates)
        else None,
        "gnss_update_count": int(logs["estimator_used_gnss"].sum())
        if "estimator_used_gnss" in logs
        else None,
        "star_tracker_update_count": int(logs["estimator_used_star_tracker"].sum())
        if "estimator_used_star_tracker" in logs
        else None,
        "sun_keepout_margin_min_deg": _series_stat(logs, "active_target_sun_keepout_margin_deg", np.nanmin),
        "earth_keepout_margin_min_deg": _series_stat(logs, "active_target_earth_keepout_margin_deg", np.nanmin),
    }
    return _json_safe(summary)


def estimate_settling_times_seconds(logs: pd.DataFrame, threshold_deg: float) -> list[float]:
    """Estimate time from active-target changes until pointing error is below a threshold."""
    if "active_target_boresight_angle_deg" not in logs or "current_target_idx" not in logs:
        return []
    time = logs["time"].to_numpy(dtype=np.float64)
    target_idx = logs["current_target_idx"].fillna(-1).to_numpy(dtype=np.int64)
    error = logs["active_target_boresight_angle_deg"].to_numpy(dtype=np.float64)
    switch_indices = np.flatnonzero(np.diff(target_idx) != 0) + 1
    settling_times: list[float] = []
    for start in switch_indices:
        if target_idx[start] < 0:
            continue
        end_candidates = np.flatnonzero(target_idx[start:] != target_idx[start])
        end = start + int(end_candidates[0]) if len(end_candidates) else len(target_idx)
        settled = np.flatnonzero(error[start:end] <= threshold_deg)
        if len(settled):
            settling_times.append(float(time[start + int(settled[0])] - time[start]))
    return settling_times


def save_mission_artifact(
    artifact: dict[str, Any],
    output_dir: Path,
    *,
    stem: str = "mission_results",
    save_csv: bool = True,
) -> MissionArtifactPaths:
    """Save a mission artifact as compressed NPZ plus JSON summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{stem}.npz"
    summary_path = output_dir / f"{stem}_summary.json"
    metadata_path = output_dir / f"{stem}_metadata.json"
    csv_path = output_dir / f"{stem}_timeseries.csv" if save_csv else None

    arrays = dict(artifact["arrays"])
    targets = artifact["targets"]
    target_numeric_columns = [
        column
        for column in targets.columns
        if pd.api.types.is_numeric_dtype(targets[column]) or pd.api.types.is_bool_dtype(targets[column])
    ]
    targets_numeric = targets[target_numeric_columns]
    for column in targets_numeric.columns:
        arrays[f"target_{column}"] = targets_numeric[column].to_numpy()
    arrays["summary_json"] = np.array(json.dumps(artifact["summary"], indent=2))
    arrays["metadata_json"] = np.array(json.dumps(artifact["metadata"], indent=2))
    arrays["schema_version"] = np.array(str(artifact["schema_version"]))
    np.savez_compressed(npz_path, **arrays)

    summary_path.write_text(json.dumps(artifact["summary"], indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(artifact["metadata"], indent=2), encoding="utf-8")
    if csv_path is not None:
        artifact["timeseries"].to_csv(csv_path, index=False)

    return MissionArtifactPaths(
        npz=npz_path,
        summary_json=summary_path,
        timeseries_csv=csv_path,
        metadata_json=metadata_path,
    )


def build_metadata(config: ClosedLoopTestConfig) -> dict[str, Any]:
    """Return JSON-safe run metadata from the dataclass configuration."""
    metadata = {
        "rng_seed": config.rng_seed,
        "timing": {
            "sim_dt": config.timing.sim_dt,
            "mpc_dt": config.timing.mpc_dt,
            "sim_cycles": config.timing.sim_cycles,
            "start_epoch_utc": config.timing.start_epoch_utc.isoformat(),
        },
        "environment": _json_safe(asdict(config.environment)),
        "spacecraft": _json_safe(asdict(config.spacecraft)),
        "mpc": _json_safe(asdict(config.mpc)),
        "sensors": _json_safe(asdict(config.sensors)),
        "initial_state": _json_safe(asdict(config.initial_state)),
        "covariance_labels": COVARIANCE_LABELS,
        "artifact_format_notes": {
            "npz": "Primary dependency-free compressed numerical artifact.",
            "json": "Human-readable summary and metadata for website/report copy.",
            "csv": "Optional quick-inspection table; not canonical for large covariance-heavy data.",
        },
    }
    return _json_safe(metadata)


def make_outcome_sentence(completed: int, total: int, observing_pointing_max_deg: float | None) -> str:
    """Create concise recruiter-friendly outcome text."""
    if observing_pointing_max_deg is None or not np.isfinite(observing_pointing_max_deg):
        return f"Completed {completed}/{total} targets in one closed-loop mission run."
    return (
        f"Completed {completed}/{total} targets in one closed-loop mission run while "
        f"holding observing pointing error below {observing_pointing_max_deg:.4g} deg."
    )


def quaternion_angle_error_deg(q_current: np.ndarray, q_reference: np.ndarray) -> np.ndarray:
    """Return shortest-axis quaternion angle error for two quaternion histories."""
    current = qu.as_quat_array(q_current)
    reference = qu.as_quat_array(q_reference)
    error = reference * current.conjugate()
    error_float = qu.as_float_array(error)
    error_float = error_float / np.linalg.norm(error_float, axis=1, keepdims=True)
    scalar = np.clip(np.abs(error_float[:, 0]), -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(scalar))


def _covariance_diag_from_logs(logs: pd.DataFrame) -> np.ndarray:
    columns = [f"mekf_cov_diag_{idx:02d}" for idx in range(15)]
    if all(column in logs.columns for column in columns):
        return logs[columns].to_numpy(dtype=np.float64)
    return np.full((len(logs), 15), np.nan, dtype=np.float64)


def _derived_timeseries_dataframe(derived: dict[str, np.ndarray]) -> pd.DataFrame:
    """Convert 1D and 2D derived arrays into a flat dataframe for CSV export."""
    columns: dict[str, np.ndarray] = {}
    for key, value in derived.items():
        arr = np.asarray(value)
        if arr.ndim == 1:
            columns[f"derived_{key}"] = arr
            continue
        if arr.ndim == 2:
            labels = COVARIANCE_LABELS if arr.shape[1] == len(COVARIANCE_LABELS) else None
            for idx in range(arr.shape[1]):
                suffix = labels[idx] if labels is not None else f"{idx:02d}"
                columns[f"derived_{key}_{suffix}"] = arr[:, idx]
            continue
        raise ValueError(f"Derived time series '{key}' must be 1D or 2D, got shape {arr.shape}.")
    return pd.DataFrame(columns)


def _final_target_observation_times(logs: pd.DataFrame, target_count: int) -> np.ndarray:
    values = np.zeros(target_count, dtype=np.float64)
    if logs.empty:
        return values
    final = logs.iloc[-1]
    for idx in range(target_count):
        column = f"target_{idx}_obs_time"
        if column in logs.columns:
            values[idx] = float(final[column])
    return values


def _safe_stat(values: np.ndarray, stat_fn: Any) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return None
    return float(stat_fn(arr))


def _masked_stat(values: np.ndarray, mask: np.ndarray, stat_fn: Any) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if arr.size == 0 or not np.any(mask):
        return None
    return _safe_stat(arr[mask], stat_fn)


def _series_stat(logs: pd.DataFrame, column: str, stat_fn: Any) -> float | None:
    if column not in logs:
        return None
    return _safe_stat(logs[column].to_numpy(dtype=np.float64), stat_fn)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
