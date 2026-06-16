"""Reusable closed-loop spacecraft simulation test API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import cast

import numpy as np
import pandas as pd

from mpc_spacecraft.config import ClosedLoopTestConfig
from mpc_spacecraft.controllers import NominalMPC
from mpc_spacecraft.controllers.error_dynamics_adapters import (
    SpacecraftErrorDynamicsProvider,
)
from mpc_spacecraft.dynamics import DisturbanceModel, SpacecraftDynamics
from mpc_spacecraft.estimation.sensor_models import GyroConfig, IMUConfig
from mpc_spacecraft.guidance import AstropySunDirectionModel, Guidance
from mpc_spacecraft.planner import StarPlanner
from mpc_spacecraft.simulation.clock import SimulationClock
from mpc_spacecraft.simulation.sensor_state_updater import SensorBiasUpdater
from mpc_spacecraft.simulation.state_estimation_sim_handler import (
    StateEstimationSimHandler,
)
from mpc_spacecraft.utilities.utils import (
    FloatArray,
    FullSimState,
    RigidBodyState,
    RigidBodyControl,
    RotationState,
    SensorRigidBodyState,
)

@dataclass(frozen=True)
class ClosedLoopLogRecord:
    """A single plotting-friendly closed-loop simulation log record."""

    cycle: int
    time: float
    epoch_utc: datetime
    true_state: FloatArray
    estimated_state: FloatArray
    goal_rotation_state: FloatArray
    control: FloatArray
    guidance_mode_value: int
    guidance_mode_name: str
    is_complete: bool
    mpc_updated: bool
    current_target_idx: int | None
    target_obs_times: FloatArray | None


@dataclass
class ClosedLoopTestResult:
    """Closed-loop test result and log extraction helper."""

    config: ClosedLoopTestConfig
    records: list[ClosedLoopLogRecord] = field(default_factory=list)
    final_state: FullSimState | None = None

    def append(self, record: ClosedLoopLogRecord) -> None:
        """Append one log record."""
        self.records.append(record)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Return logged data as numpy arrays for plotting and analysis."""
        if not self.records:
            return {}

        return {
            "cycle": np.array([r.cycle for r in self.records], dtype=np.int64),
            "time": np.array([r.time for r in self.records], dtype=np.float64),
            "true_state": np.array([r.true_state for r in self.records], dtype=np.float64),
            "estimated_state": np.array(
                [r.estimated_state for r in self.records], dtype=np.float64
            ),
            "goal_rotation_state": np.array(
                [r.goal_rotation_state for r in self.records], dtype=np.float64
            ),
            "control": np.array([r.control for r in self.records], dtype=np.float64),
            "guidance_mode_value": np.array(
                [r.guidance_mode_value for r in self.records], dtype=np.int64
            ),
            "is_complete": np.array([r.is_complete for r in self.records], dtype=bool),
            "mpc_updated": np.array([r.mpc_updated for r in self.records], dtype=bool),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return flattened logged data as a pandas DataFrame."""
        rows = []
        for record in self.records:
            true_state = SensorRigidBodyState.from_array(record.true_state)
            estimated_state = SensorRigidBodyState.from_array(record.estimated_state)
            goal_state = RotationState.from_array(record.goal_rotation_state)
            row = {
                "cycle": record.cycle,
                "time": record.time,
                "epoch_utc": record.epoch_utc,
                "guidance_mode_value": record.guidance_mode_value,
                "guidance_mode": record.guidance_mode_name,
                "is_complete": record.is_complete,
                "mpc_updated": record.mpc_updated,
                "current_target_idx": record.current_target_idx,
                "control_tx": record.control[0],
                "control_ty": record.control[1],
                "control_tz": record.control[2],
            }
            _add_vec_columns(row, "true_position", true_state.position)
            _add_vec_columns(row, "true_velocity", true_state.velocity)
            _add_quat_columns(row, "true_quat", true_state.quat)
            _add_vec_columns(row, "true_omega", true_state.omega)
            _add_vec_columns(row, "estimated_position", estimated_state.position)
            _add_vec_columns(row, "estimated_velocity", estimated_state.velocity)
            _add_quat_columns(row, "estimated_quat", estimated_state.quat)
            _add_vec_columns(row, "estimated_omega", estimated_state.omega)
            _add_vec_columns(row, "gyro_bias", true_state.gyro_bias)
            _add_vec_columns(row, "accel_bias", true_state.accel_bias)
            _add_quat_columns(row, "goal_quat", goal_state.quat)
            _add_vec_columns(row, "goal_omega", goal_state.omega)
            rows.append(row)

        return pd.DataFrame(rows)


def run_closed_loop_test(config: ClosedLoopTestConfig | None = None) -> ClosedLoopTestResult:
    """Run a reusable closed-loop spacecraft test and return structured logs."""
    if config is None:
        config = ClosedLoopTestConfig.defaults()

    timing = config.timing
    environment = config.environment
    spacecraft = config.spacecraft
    mpc_config = config.mpc
    sensors = config.sensors

    sim_state = config.initial_state.to_full_sim_state()
    clock = SimulationClock(timing.start_epoch_utc, timing.sim_dt)
    sun_model = AstropySunDirectionModel()
    planner = StarPlanner(
        config.targets.targets.copy(),
        environment.sun_keep_out_deg,
        environment.earth_margin_deg,
    )
    guidance = Guidance(planner, sun_model)

    disturbance_model = DisturbanceModel(
        environment.disturbance_bias,
        environment.disturbance_noise_std,
        environment.disturbance_sinusoidal_amplitude,
        environment.disturbance_sinusoidal_frequency,
        config.rng_seed,
        enable_gravity=environment.enable_gravity,
    )
    dynamics = SpacecraftDynamics(spacecraft.inertia, disturbance_model, spacecraft.mass)
    error_dynamics_provider = SpacecraftErrorDynamicsProvider(
        dynamics,
        sun_model,
        np.deg2rad(environment.sun_keep_out_deg),
        timing.mpc_dt,
    )
    mpc = NominalMPC(
        mpc_config.step_horizon,
        mpc_config.q_weight,
        mpc_config.r_weight,
        error_dynamics_provider,
        mpc_config.q_terminal if mpc_config.q_terminal is not None else mpc_config.q_weight,
        spacecraft.u_min,
        spacecraft.u_max,
    )
    estimator = StateEstimationSimHandler(
        sim_state.sensor_rigid_body,
        sensors.gnss_measurement_period,
        sensors.star_tracker_measurement_period,
        config.rng_seed,
    )
    sensor_bias_updater = SensorBiasUpdater(IMUConfig(), GyroConfig(), config.rng_seed)

    result = ClosedLoopTestResult(config=config)
    total_control = RigidBodyControl.zeros()
    previous_mpc_guidance_update: float | None = None
    goal_rotation_state = sim_state.rotation.copy()
    mode: IntEnum | None = None
    is_complete = False

    iterator = range(timing.sim_cycles)
    if config.show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator)

    for sim_cycle in iterator:
        estimated_state = (
            estimator.tick(clock.current_time, clock.last_dt, sim_state)
            if clock.last_dt != 0
            else sim_state.sensor_rigid_body
        )

        mpc_update = previous_mpc_guidance_update is None or (
            clock.current_time - previous_mpc_guidance_update >= timing.mpc_dt
        )
        if mpc_update:
            goal_rotation_state, mode, is_complete = guidance.tick(
                clock.now(), timing.sim_dt, cast(RigidBodyState, estimated_state.rigid_body)
            )
            rotation_control = mpc.get_first_control(
                x0=estimated_state.rotation,
                x_goal=goal_rotation_state,
                current_epoch_utc=clock.now(),
            )
            total_control = RigidBodyControl.zeros()
            total_control.rotation[:] = rotation_control
            previous_mpc_guidance_update = clock.current_time

        result.append(
            ClosedLoopLogRecord(
                cycle=sim_cycle,
                time=clock.current_time,
                epoch_utc=clock.now(),
                true_state=sim_state.sensor_rigid_body.data.copy(),
                estimated_state=estimated_state.data.copy(),
                goal_rotation_state=goal_rotation_state.data.copy(),
                control=total_control.torque.copy(),
                guidance_mode_value=int(mode) if mode is not None else -1,
                guidance_mode_name=mode.name if mode is not None else "UNKNOWN",
                is_complete=bool(is_complete),
                mpc_updated=bool(mpc_update),
                current_target_idx=getattr(planner, "_current_target_idx", None),
                target_obs_times=planner._targets["obs_time"].to_numpy().copy(),
            )
        )

        time_step = clock.advance(1)
        sim_state.rigid_body[:] = dynamics.discretize_dynamics_rk4_full_state(
            sim_state.rigid_body,
            total_control,
            time_step.t0,
            time_step.dt,
        ).data
        sim_state = sensor_bias_updater.tick(sim_state, time_step.dt)

    result.final_state = sim_state.copy()
    return result


def _add_vec_columns(row: dict[str, object], prefix: str, values: FloatArray) -> None:
    row[f"{prefix}_x"] = values[0]
    row[f"{prefix}_y"] = values[1]
    row[f"{prefix}_z"] = values[2]


def _add_quat_columns(row: dict[str, object], prefix: str, values: FloatArray) -> None:
    row[f"{prefix}_w"] = values[0]
    row[f"{prefix}_x"] = values[1]
    row[f"{prefix}_y"] = values[2]
    row[f"{prefix}_z"] = values[3]
