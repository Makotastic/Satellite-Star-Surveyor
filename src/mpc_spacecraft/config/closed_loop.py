"""Closed-loop simulation configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from mpc_spacecraft.utilities.utils import FloatArray, FullSimState


@dataclass(frozen=True)
class ClosedLoopTimingConfig:
    """Timing parameters for a closed-loop test."""

    sim_dt: float = 0.1
    mpc_dt: float = 1.0
    sim_cycles: int = 6_000
    start_epoch_utc: datetime = field(default_factory=lambda: datetime(2025, 1, 9))


@dataclass(frozen=True)
class ClosedLoopEnvironmentConfig:
    """Environment and guidance constraint parameters."""

    sun_keep_out_deg: float = 35.0
    earth_margin_deg: float = 10.0
    disturbance_bias: FloatArray | None = field(
        default_factory=lambda: np.array([2.0e-6, -1.0e-6, 1.0e-6], dtype=np.float64)
    )
    disturbance_noise_std: float | None = 5.0e-7
    disturbance_sinusoidal_amplitude: float | None = 2.0e-6
    disturbance_sinusoidal_frequency: float | None = 0.01
    enable_gravity: bool = True


@dataclass(frozen=True)
class SpacecraftPhysicalConfig:
    """Spacecraft physical parameters for closed-loop testing.

    The mass remains representative of a medium space telescope, but the default
    inertia is intentionally scaled down by roughly 10x from a full observatory
    model so end-to-end closed-loop functionality can be exercised without long
    settling simulations. Use ``medium_space_telescope()`` when you want the
    slower, more realistic inertia values.
    """

    inertia: FloatArray = field(
        default_factory=lambda: np.diag([260.0, 240.0, 150.0]).astype(np.float64)
    )
    mass: float = 1_200.0
    u_min: FloatArray = field(
        default_factory=lambda: -0.25 * np.ones(3, dtype=np.float64)
    )
    u_max: FloatArray = field(
        default_factory=lambda: 0.25 * np.ones(3, dtype=np.float64)
    )

    @classmethod
    def medium_space_telescope(cls) -> "SpacecraftPhysicalConfig":
        """Return slower, more realistic medium space telescope inertia values."""
        return cls(
            inertia=np.diag([2_600.0, 2_400.0, 1_500.0]).astype(np.float64),
            mass=1_200.0,
            u_min=-0.25 * np.ones(3, dtype=np.float64),
            u_max=0.25 * np.ones(3, dtype=np.float64),
        )


@dataclass(frozen=True)
class ClosedLoopMPCConfig:
    """MPC tuning parameters."""

    step_horizon: int = 30
    q_weight: FloatArray = field(
        default_factory=lambda: np.diag(
            [20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
        ).astype(np.float64)
    )
    r_weight: FloatArray = field(
        default_factory=lambda: 5.0 * np.eye(3, dtype=np.float64)
    )
    q_terminal: FloatArray | None = None


@dataclass(frozen=True)
class SensorScheduleConfig:
    """Sensor sampling schedule used by the estimator simulation."""

    gnss_measurement_period: float = 5.0
    star_tracker_measurement_period: float = 1.0


@dataclass(frozen=True)
class InitialStateConfig:
    """Initial full simulation state parameters."""

    position: FloatArray = field(
        default_factory=lambda: np.array([7_078e3, 0.0, 0.0], dtype=np.float64)
    )
    velocity: FloatArray = field(
        default_factory=lambda: np.array([0.0, 7.504e3, 0.0], dtype=np.float64)
    )
    quat: FloatArray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    omega: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    gyro_bias: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    accel_bias: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def to_full_sim_state(self) -> FullSimState:
        """Create a mutable full simulation state from this config."""
        state = FullSimState.zeros()
        state.position[:] = self.position
        state.velocity[:] = self.velocity
        state.quat[:] = self.quat
        state.omega[:] = self.omega
        state.gyro_bias[:] = self.gyro_bias
        state.accel_bias[:] = self.accel_bias
        return state


@dataclass(frozen=True)
class TargetConfig:
    """Target table parameters for star-observation closed-loop tests."""

    targets: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            {
                "req_obs_time": [20.0, 20.0, 20.0, 20.0, 20.0],
                "Dec": [0.174533, -0.349066, 0.610865, 0.087266, -0.785398],
                "RA": [0.261799, 1.396263, 2.530727, 3.839724, 5.410521],
            }
        )
    )


@dataclass(frozen=True)
class ClosedLoopTestConfig:
    """Top-level configuration for ``run_closed_loop_test``."""

    timing: ClosedLoopTimingConfig = field(default_factory=ClosedLoopTimingConfig)
    environment: ClosedLoopEnvironmentConfig = field(default_factory=ClosedLoopEnvironmentConfig)
    spacecraft: SpacecraftPhysicalConfig = field(default_factory=SpacecraftPhysicalConfig)
    mpc: ClosedLoopMPCConfig = field(default_factory=ClosedLoopMPCConfig)
    sensors: SensorScheduleConfig = field(default_factory=SensorScheduleConfig)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    rng_seed: int = 2
    show_progress: bool = True

    @classmethod
    def defaults(cls, **overrides: Any) -> "ClosedLoopTestConfig":
        """Return the default config, optionally overriding top-level fields."""
        return cls(**overrides)
