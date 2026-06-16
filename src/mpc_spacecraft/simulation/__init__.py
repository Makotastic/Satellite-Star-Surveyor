"""Simulation module for closed-loop control."""

from mpc_spacecraft.config import (
    ClosedLoopEnvironmentConfig,
    ClosedLoopMPCConfig,
    ClosedLoopTestConfig,
    ClosedLoopTimingConfig,
    InitialStateConfig,
    SensorScheduleConfig,
    SpacecraftPhysicalConfig,
    TargetConfig,
)

from .closed_loop_testing import (
    ClosedLoopLogRecord,
    ClosedLoopTestResult,
    run_closed_loop_test,
)

__all__ = [
    "ClosedLoopEnvironmentConfig",
    "ClosedLoopLogRecord",
    "ClosedLoopMPCConfig",
    "ClosedLoopTestConfig",
    "ClosedLoopTestResult",
    "ClosedLoopTimingConfig",
    "InitialStateConfig",
    "SensorScheduleConfig",
    "SpacecraftPhysicalConfig",
    "TargetConfig",
    "run_closed_loop_test",
]
