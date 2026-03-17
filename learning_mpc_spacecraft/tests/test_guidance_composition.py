import numpy as np
import pytest

from mpc_spacecraft.guidance.guidance import Guideance


class _PlannerStub:
    def __init__(self):
        self.last_sun_dir = None

    def tick(self, body_state, dt, sun_dir_I):
        self.last_sun_dir = np.asarray(sun_dir_I, dtype=float)
        return None, None, True


class _SunModelStub:
    def sun_dir_eci(self, epoch_utc):
        return np.array([0.0, 1.0, 0.0], dtype=float)


@pytest.mark.unit
def test_guidance_uses_composed_sun_model_for_planner_input():
    planner = _PlannerStub()
    sun_model = _SunModelStub()
    guidance = Guideance(planner=planner, sun_model=sun_model)

    body_state = np.zeros(13)
    body_state[6] = 1.0

    goal_rot = guidance.tick(dt=0.1, body_state=body_state, epoch_utc="2026-03-20T00:00:00")

    np.testing.assert_allclose(planner.last_sun_dir, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(goal_rot, body_state[6:13])

