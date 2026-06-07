from datetime import datetime, timedelta, timezone

import pytest

from mpc_spacecraft.simulation.clock import SimulationClock, TimeStep


def test_advance_records_last_dt_for_fixed_tick_duration() -> None:
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = SimulationClock(start_time_utc=start_time, tick_duration=0.25)

    assert clock.last_dt == 0.0

    step = clock.advance()

    assert step == TimeStep(t0=0.0, t1=pytest.approx(0.25), dt=pytest.approx(0.25))
    assert clock.current_time == pytest.approx(0.25)
    assert clock.last_dt == pytest.approx(0.25)
    assert clock.now() == start_time + timedelta(seconds=0.25)


def test_advance_uses_tick_duration_captured_for_that_call() -> None:
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = SimulationClock(start_time_utc=start_time, tick_duration=0.1)

    clock.advance()
    clock.tick_duration = 0.5
    step = clock.advance(2)

    assert step == TimeStep(t0=pytest.approx(0.1), t1=pytest.approx(1.1), dt=pytest.approx(1.0))
    assert clock.current_time == pytest.approx(1.1)
    assert clock.last_dt == pytest.approx(1.0)
    assert clock.now() == start_time + timedelta(seconds=1.1)


def test_advance_zero_steps_records_zero_last_dt_without_changing_time() -> None:
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = SimulationClock(start_time_utc=start_time, tick_duration=0.2)

    clock.advance(3)
    step = clock.advance(0)

    assert step == TimeStep(t0=pytest.approx(0.6), t1=pytest.approx(0.6), dt=pytest.approx(0.0))
    assert clock.current_time == pytest.approx(0.6)
    assert clock.last_dt == pytest.approx(0.0)


def test_reset_clears_elapsed_state_and_last_dt() -> None:
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = SimulationClock(start_time_utc=start_time, tick_duration=0.3)

    clock.advance(2)
    clock.reset()

    assert clock.current_time == pytest.approx(0.0)
    assert clock.last_dt == pytest.approx(0.0)
    assert clock.now() == start_time


def test_advance_rejects_negative_steps() -> None:
    clock = SimulationClock(
        start_time_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        tick_duration=0.1,
    )

    with pytest.raises(ValueError, match="steps must be >= 0"):
        clock.advance(-1)
