"""Simulation clock implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class TimeStep:
    """Elapsed simulation interval produced by a clock advancement."""

    t0: float
    t1: float
    dt: float


class SimulationClock:
    """Simulation clock initialized with a UTC start time."""

    def __init__(
        self, start_time_utc: datetime | None = None, tick_duration: float | None = None
    ):
        if start_time_utc is None:
            raise TypeError("start_time_utc is required")

        normalized_start = self._normalize_to_utc(start_time_utc)
        self._tick_duration = 0.1 if tick_duration is None else float(tick_duration)
        self._start_time_utc = normalized_start
        self._current_time = 0.0
        self._last_dt = 0.0

    @staticmethod
    def _normalize_to_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @property
    def current_time(self) -> float:
        """Elapsed simulation time in seconds since reset."""
        return self._current_time

    @property
    def tick_duration(self) -> float:
        """Duration configured for the next simulation tick in seconds."""
        return self._tick_duration

    @tick_duration.setter
    def tick_duration(self, value: float) -> None:
        """Set the duration to use for subsequent simulation ticks."""
        self._tick_duration = float(value)

    @property
    def last_dt(self) -> float:
        """Elapsed seconds from the previous tick to the current tick."""
        return self._last_dt

    @property
    def start_time_utc(self) -> datetime:
        """UTC start time for this simulation timeline."""
        return self._start_time_utc

    def now(self) -> datetime:
        """Return absolute UTC time at the current simulation instant."""
        return self._start_time_utc + timedelta(seconds=self._current_time)

    def advance(self, steps: int = 1) -> TimeStep:
        """Advance simulation by a number of ticks and return the elapsed interval."""
        if steps < 0:
            raise ValueError("steps must be >= 0")

        step_count = int(steps)
        tick_duration = self._tick_duration
        t0 = self._current_time
        self._last_dt = step_count * tick_duration
        self._current_time += self._last_dt
        return TimeStep(t0=t0, t1=self._current_time, dt=self._last_dt)

    def reset(self) -> None:
        """Reset elapsed simulation state back to the configured UTC start time."""
        self._current_time = 0.0
        self._last_dt = 0.0
