"""Simulation clock implementation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


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
        self._tick_count = 0

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
        """Duration of one simulation tick in seconds."""
        return self._tick_duration

    @property
    def tick_count(self) -> int:
        """Number of elapsed ticks since reset."""
        return self._tick_count

    @property
    def start_time_utc(self) -> datetime:
        """UTC start time for this simulation timeline."""
        return self._start_time_utc

    def now(self) -> datetime:
        """Return absolute UTC time at the current simulation instant."""
        return self._start_time_utc + timedelta(seconds=self._current_time)

    def advance(self, steps: int = 1) -> float:
        """Advance simulation by a number of ticks and return elapsed seconds."""
        if steps < 0:
            raise ValueError("steps must be >= 0")

        self._tick_count += int(steps)
        self._current_time = self._tick_count * self._tick_duration
        return self._current_time

    def reset(self) -> None:
        """Reset elapsed simulation state back to the configured UTC start time."""
        self._current_time = 0.0
        self._tick_count = 0
