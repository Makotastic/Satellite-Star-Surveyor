"""Tests for Guidance composition with current tick API."""

from datetime import datetime, timezone
from typing import Final

import numpy as np
import pandas as pd
import pytest
import quaternion as qu

from mpc_spacecraft.guidance.guidance import Guidance
from mpc_spacecraft.guidance.sun_tracker import AstropySunDirectionModel, TimeLike
from mpc_spacecraft.planner.star_planner import (
    PlanModes,
    StarPlanner,
    compute_earth_body_directions,
    compute_earth_sun_body_angles,
)
from mpc_spacecraft.utilities.utils import (
    BODY_FORWARD_VEC3,
    RigidBodyState,
    Vec3,
    quat_rotate_vector_a_to_b,
    unit,
)


EPOCH_UTC: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


class DeterministicSunDirectionModel(AstropySunDirectionModel):
    def __init__(self, sun_dir_I: np.ndarray):
        self.sun_dir_I = unit(sun_dir_I)

    def sun_dir_eci(self, epoch_utc: TimeLike) -> Vec3:
        return self.sun_dir_I


@pytest.fixture
def sun_model() -> AstropySunDirectionModel:
    return AstropySunDirectionModel()


@pytest.fixture(scope="function")
def deterministic_sun_model() -> DeterministicSunDirectionModel:
    return DeterministicSunDirectionModel(np.array([0.0, 0.0, 1.0]))


@pytest.fixture
def targets() -> pd.DataFrame:
    return pd.DataFrame(
        [{"req_obs_time": 1.0, "Dec": 0.0, "RA": 0.0}],
        dtype=float,
    )


@pytest.fixture(scope="function")
def multi_targets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"req_obs_time": 1.0, "Dec": 0.0, "RA": 0.0},
            {"req_obs_time": 1.0, "Dec": 0.0, "RA": np.pi / 2.0},
            {"req_obs_time": 1.0, "Dec": 0.0, "RA": np.pi},
        ],
        dtype=float,
    )


@pytest.fixture
def planner(targets: pd.DataFrame) -> StarPlanner:
    return StarPlanner(
        targets=targets,
        sun_keepout_deg=5.0,
        earth_margin_deg=5.0,
        stabilizing_threshold_deg=10.0,
        observing_threshold_deg=1.0,
        observing_max_angular_rate=10.0,
    )


@pytest.fixture(scope="function")
def multi_target_planner(multi_targets: pd.DataFrame) -> StarPlanner:
    return StarPlanner(
        targets=multi_targets,
        sun_keepout_deg=5.0,
        earth_margin_deg=5.0,
        stabilizing_threshold_deg=10.0,
        observing_threshold_deg=1.0,
        observing_max_angular_rate=1.0,
    )


@pytest.fixture
def no_target_planner(targets: pd.DataFrame) -> StarPlanner:
    planner = StarPlanner(
        targets=targets,
        sun_keepout_deg=180.0,
        earth_margin_deg=180.0,
        stabilizing_threshold_deg=10.0,
        observing_threshold_deg=1.0,
        observing_max_angular_rate=10.0,
    )
    planner._targets.loc[:, "req_obs_time"] = 10.0
    return planner


@pytest.fixture
def guidance(planner: StarPlanner, sun_model: AstropySunDirectionModel) -> Guidance:
    return Guidance(planner=planner, sun_model=sun_model)


@pytest.fixture(scope="function")
def deterministic_guidance(
    multi_target_planner: StarPlanner,
    deterministic_sun_model: DeterministicSunDirectionModel,
) -> Guidance:
    return Guidance(planner=multi_target_planner, sun_model=deterministic_sun_model)


@pytest.fixture
def no_target_guidance(
    no_target_planner: StarPlanner, sun_model: AstropySunDirectionModel
) -> Guidance:
    return Guidance(planner=no_target_planner, sun_model=sun_model)


@pytest.fixture
def body_state() -> RigidBodyState:
    state = RigidBodyState.zeros()
    state.position[:] = np.array([7.0e6, 0.0, 0.0])
    state.quat[:] = np.array([1.0, 0.0, 0.0, 0.0])
    state.omega[:] = np.zeros(3)
    return state


@pytest.fixture(scope="function")
def unobstructed_body_state() -> RigidBodyState:
    state = RigidBodyState.zeros()
    state.position[:] = np.array([0.0, 0.0, 7.0e6])
    state.quat[:] = np.array([1.0, 0.0, 0.0, 0.0])
    state.omega[:] = np.zeros(3)
    return state


def set_body_direction(
    body_state: RigidBodyState,
    body_dir_I: np.ndarray,
    omega_norm_rad_s: float = 0.0,
) -> None:
    body_state.quat[:] = qu.as_float_array(
        quat_rotate_vector_a_to_b(BODY_FORWARD_VEC3, body_dir_I)
    )
    body_state.omega[:] = np.array([omega_norm_rad_s, 0.0, 0.0])


def assert_goal_points_to(goal_quat: np.ndarray, expected_dir_I: np.ndarray) -> None:
    goal_dir_I = qu.rotate_vectors(qu.quaternion(*goal_quat), BODY_FORWARD_VEC3)

    np.testing.assert_allclose(goal_dir_I, unit(expected_dir_I), atol=1.0e-12)


def test_guidance_tick_returns_runtime_goal_state(
    guidance: Guidance,
    body_state: RigidBodyState,
):
    current_epoch_utc = EPOCH_UTC

    goal_state, mode, complete = guidance.tick(
        current_epoch_utc=current_epoch_utc,
        delta_time=0.1,
        body_state=body_state,
    )

    assert mode in PlanModes
    assert isinstance(complete, (bool, np.bool_))
    assert np.isfinite(goal_state.data).all()


def test_guidance_tick_preserves_body_attitude_when_no_target(
    no_target_guidance: Guidance,
    body_state: RigidBodyState,
):
    goal_state, mode, complete = no_target_guidance.tick(
        current_epoch_utc=EPOCH_UTC,
        delta_time=0.1,
        body_state=body_state,
    )

    assert mode == PlanModes.NO_TARGET
    assert not bool(complete)
    np.testing.assert_allclose(goal_state.quat, body_state.rotation.quat)
    np.testing.assert_allclose(goal_state.omega, np.zeros(3))


@pytest.mark.parametrize(
    ("body_dir_I", "omega_norm_rad_s", "expected_mode"),
    [
        (np.array([0.0, 1.0, 0.0]), 0.0, PlanModes.SLEWING),
        (unit(np.array([1.0, 0.1, 0.0])), np.deg2rad(5.0), PlanModes.SETTLING),
        (np.array([1.0, 0.0, 0.0]), np.deg2rad(0.5), PlanModes.OBSERVING),
    ],
)
def test_guidance_tick_reports_mode_from_pointing_error_and_rate(
    deterministic_guidance: Guidance,
    unobstructed_body_state: RigidBodyState,
    body_dir_I: np.ndarray,
    omega_norm_rad_s: float,
    expected_mode: PlanModes,
) -> None:
    set_body_direction(unobstructed_body_state, np.array([1.0, 0.0, 0.0]))
    deterministic_guidance.tick(
        current_epoch_utc=EPOCH_UTC,
        delta_time=0.1,
        body_state=unobstructed_body_state,
    )
    set_body_direction(unobstructed_body_state, body_dir_I, omega_norm_rad_s)

    goal_state, mode, complete = deterministic_guidance.tick(
        current_epoch_utc=EPOCH_UTC,
        delta_time=0.1,
        body_state=unobstructed_body_state,
    )

    assert mode == expected_mode
    assert not bool(complete)
    assert_goal_points_to(goal_state.quat, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(goal_state.omega, np.zeros(3))


def test_guidance_tick_observes_targets_then_slews_to_next_target(
    deterministic_guidance: Guidance,
    unobstructed_body_state: RigidBodyState,
) -> None:
    expected_target_dirs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
    ]

    goal_state, mode, complete = deterministic_guidance.tick(
        current_epoch_utc=EPOCH_UTC,
        delta_time=0.1,
        body_state=unobstructed_body_state,
    )

    assert mode == PlanModes.SLEWING
    assert not bool(complete)
    assert_goal_points_to(goal_state.quat, expected_target_dirs[0])

    set_body_direction(unobstructed_body_state, expected_target_dirs[0])

    goal_state, mode, complete = deterministic_guidance.tick(
        current_epoch_utc=EPOCH_UTC,
        delta_time=0.1,
        body_state=unobstructed_body_state,
    )

    assert mode == PlanModes.OBSERVING
    assert not bool(complete)
    assert_goal_points_to(goal_state.quat, expected_target_dirs[0])

    goal_state, mode, complete = deterministic_guidance.tick(
        current_epoch_utc=EPOCH_UTC,
        delta_time=1.0,
        body_state=unobstructed_body_state,
    )

    assert mode == PlanModes.SLEWING
    assert not bool(complete)
    assert_goal_points_to(goal_state.quat, expected_target_dirs[1])

    set_body_direction(unobstructed_body_state, expected_target_dirs[1])
    deterministic_guidance.tick(
        current_epoch_utc=EPOCH_UTC,
        delta_time=0.1,
        body_state=unobstructed_body_state,
    )

    goal_state, mode, complete = deterministic_guidance.tick(
        current_epoch_utc=EPOCH_UTC,
        delta_time=1.0,
        body_state=unobstructed_body_state,
    )

    assert mode == PlanModes.SLEWING
    assert not bool(complete)
    assert_goal_points_to(goal_state.quat, expected_target_dirs[2])

    set_body_direction(unobstructed_body_state, expected_target_dirs[2])
    deterministic_guidance.tick(
        current_epoch_utc=EPOCH_UTC,
        delta_time=0.1,
        body_state=unobstructed_body_state,
    )

    goal_state, mode, complete = deterministic_guidance.tick(
        current_epoch_utc=EPOCH_UTC,
        delta_time=1.0,
        body_state=unobstructed_body_state,
    )

    assert mode == PlanModes.NO_TARGET
    assert bool(complete)
    np.testing.assert_allclose(goal_state.quat, unobstructed_body_state.rotation.quat)
    np.testing.assert_allclose(goal_state.omega, np.zeros(3))


def test_guidance_tick_rejects_sun_bound_and_earth_bound_targets() -> None:
    targets = pd.DataFrame(
        [
            {"req_obs_time": 1.0, "Dec": 0.0, "RA": 0.0},
            {"req_obs_time": 1.0, "Dec": 0.0, "RA": np.pi},
            {"req_obs_time": 1.0, "Dec": 0.0, "RA": np.pi / 2.0},
        ],
        dtype=float,
    )
    planner = StarPlanner(
        targets=targets,
        sun_keepout_deg=20.0,
        earth_margin_deg=5.0,
        stabilizing_threshold_deg=10.0,
        observing_threshold_deg=1.0,
        observing_max_angular_rate=1.0,
    )
    guidance = Guidance(
        planner=planner,
        sun_model=DeterministicSunDirectionModel(np.array([1.0, 0.0, 0.0])),
    )
    state = RigidBodyState.zeros()
    state.position[:] = np.array([7.0e6, 0.0, 0.0])
    state.quat[:] = np.array([1.0, 0.0, 0.0, 0.0])
    state.omega[:] = np.zeros(3)

    goal_state, mode, complete = guidance.tick(
        current_epoch_utc=EPOCH_UTC,
        delta_time=0.1,
        body_state=state,
    )

    assert mode == PlanModes.SLEWING
    assert not bool(complete)
    assert_goal_points_to(goal_state.quat, np.array([0.0, 1.0, 0.0]))


def test_earth_sun_and_body_angle_calculations_match_geometry(
    body_state: RigidBodyState,
) -> None:
    star_vecs = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    sun_dir_I = np.array([0.0, 1.0, 0.0])

    earth_dir_I, body_dir_I = compute_earth_body_directions(body_state)
    earth_angle, sun_angle, dist_angle = compute_earth_sun_body_angles(
        star_vecs,
        earth_dir_I,
        sun_dir_I,
        body_dir_I,
    )

    np.testing.assert_allclose(earth_dir_I, np.array([-1.0, 0.0, 0.0]))
    np.testing.assert_allclose(body_dir_I, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(earth_angle, np.array([0.0, np.pi / 2.0, np.pi / 2.0]))
    np.testing.assert_allclose(sun_angle, np.array([np.pi / 2.0, 0.0, np.pi / 2.0]))
    np.testing.assert_allclose(dist_angle, np.array([np.pi, np.pi / 2.0, np.pi / 2.0]))
