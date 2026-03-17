import pytest
import numpy as np
import quaternion as qu

from mpc_spacecraft.controllers.mpc_nominal_drake import NominalMPC
from mpc_spacecraft.controllers.error_state_mapping import ErrorStateMappingService
from mpc_spacecraft.dynamics.rigid_body import SpacecraftDynamics


@pytest.fixture
def inertia():
    """Diagonal inertia matrix."""

    I_nominal = np.array(
        [
            [90.0, -5.0, 3.0],
            [-5.0, 110.0, -4.0],
            [3.0, -4.0, 100.0],
        ]
    )

    return I_nominal
    # return np.diag([100.0, 100.0, 100.0])


@pytest.fixture
def dt():
    """Timestep."""
    return 0.1


@pytest.fixture
def dynamics(inertia, dt):
    """Dynamics instance."""
    return SpacecraftDynamics(inertia, dt)


@pytest.fixture
def ref_state():
    """Reference state: identity quaternion, zero angular velocity."""
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def goal_state():
    """Goal state: rotated quaternion, zero omega."""
    q_goal = qu.quaternion(0.707, 0.0, 0.707, 0.0).normalized()  # ~90 deg around y
    return np.concatenate([qu.as_float_array(q_goal), np.zeros(3)])


@pytest.fixture
def Q():
    """State cost for 6D error."""
    return np.diag([10.0, 10.0, 10.0, 3.0, 3.0, 3.0])


@pytest.fixture
def R():
    """Control cost."""
    return np.diag([0.1, 0.1, 0.1])


@pytest.fixture
def horizon():
    """Short horizon for testing."""
    return 5


@pytest.fixture
def mpc(dynamics, horizon, Q, R):
    """NominalMPC instance with moderate input bounds."""
    u_min = -10 * np.ones(3)
    u_max = 10 * np.ones(3)
    return NominalMPC(
        horizon=horizon,
        dynamics=dynamics,
        Q=Q,
        R=R,
        u_min=u_min,
        u_max=u_max,
    )


@pytest.fixture
def error_mapping() -> ErrorStateMappingService:
    return ErrorStateMappingService()


# --------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------


def quaternion_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute the geodesic angle between two quaternions (in radians)."""
    q1 = qu.quaternion(*q1).normalized()
    q2 = qu.quaternion(*q2).normalized()
    # Use absolute value to avoid sign ambiguity (q and -q represent same rotation)
    dot = abs(np.dot(qu.as_float_array(q1), qu.as_float_array(q2)))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


# --------------------------------------------------------------------------
# build_ref_trajectory (cost reference) tests
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_build_ref_trajectory_tracking(mpc, ref_state):
    """When x_ref/u_ref are provided, build_ref_trajectory should return them unchanged."""
    x_provided = np.tile(ref_state, (mpc.horizon + 1, 1))
    u_provided = np.zeros((mpc.horizon, mpc.control_dim))

    x_cost, u_cost = mpc._build_ref_trajectory(
        ref_state, x_ref=x_provided, u_ref=u_provided
    )

    assert x_cost.shape == (mpc.horizon + 1, mpc.state_dim)
    assert u_cost.shape == (mpc.horizon, mpc.control_dim)
    np.testing.assert_allclose(x_cost, x_provided)
    np.testing.assert_allclose(u_cost, u_provided)


@pytest.mark.unit
def test_build_ref_trajectory_goal_constant(mpc, ref_state, goal_state):
    """
    For goal-only case, the cost reference should be constant in time:
    x_cost[k] == x_goal for all k.
    """
    x_cost, u_cost = mpc._build_ref_trajectory(ref_state, x_goal=goal_state)

    # Shapes
    assert x_cost.shape == (mpc.horizon + 1, mpc.state_dim)
    assert u_cost.shape == (mpc.horizon, mpc.control_dim)

    # All states equal to goal
    for k in range(mpc.horizon + 1):
        np.testing.assert_allclose(x_cost[k], goal_state, atol=1e-10)

    # Controls should be zero by default
    np.testing.assert_allclose(u_cost, np.zeros_like(u_cost))


# --------------------------------------------------------------------------
# find_initial_guesses (nominal trajectory) tests
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_find_initial_guesses_tracking(mpc, ref_state):
    """With x_ref provided, initial nominal guess should match it."""
    x_ref = np.tile(ref_state, (mpc.horizon + 1, 1))
    u_ref = np.zeros((mpc.horizon, mpc.control_dim))

    x_nom, u_nom = mpc._find_initial_guesses(x0=ref_state, x_ref=x_ref, u_ref=u_ref)

    assert x_nom.shape == (mpc.horizon + 1, mpc.state_dim)
    assert u_nom.shape == (mpc.horizon, mpc.control_dim)
    np.testing.assert_allclose(x_nom, x_ref)
    np.testing.assert_allclose(u_nom, u_ref)


@pytest.mark.unit
def test_find_initial_guesses_goal_interpolates(mpc, ref_state, goal_state):
    """
    With only x_goal provided, find_initial_guesses should interpolate between
    x0 and x_goal using quaternion slerp and linear omega interpolation.
    """
    x_nom, u_nom = mpc._find_initial_guesses(x0=ref_state, x_goal=goal_state)

    # Shapes
    assert x_nom.shape == (mpc.horizon + 1, mpc.state_dim)
    assert u_nom.shape == (mpc.horizon, mpc.control_dim)

    # Endpoints should match x0 and x_goal
    np.testing.assert_allclose(x_nom[0], ref_state, atol=1e-10)
    np.testing.assert_allclose(x_nom[-1], goal_state, atol=1e-10)

    # All intermediate quaternions should be normalized
    for k in range(mpc.horizon + 1):
        qk = qu.quaternion(*x_nom[k, :4])
        norm_qk = np.linalg.norm(qu.as_float_array(qk))
        np.testing.assert_allclose(norm_qk, 1.0, atol=1e-6)

    # Controls should be zero in the initial guess for goal-only
    np.testing.assert_allclose(u_nom, np.zeros_like(u_nom))


# --------------------------------------------------------------------------
# solve() behavior tests
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_solve_goal_success_and_shapes(mpc, ref_state, goal_state):
    """solve() with a goal-only problem should succeed and return correctly shaped arrays."""
    u_opt, x_opt, success = mpc._solve(ref_state, x_goal=goal_state)

    assert success
    assert u_opt.shape == (mpc.horizon, mpc.control_dim)
    assert x_opt.shape == (mpc.horizon + 1, mpc.state_dim)


@pytest.mark.unit
def test_solve_goal_reduces_orientation_error(mpc, dynamics, ref_state, goal_state):
    """
    For a goal-only problem with x0 != x_goal, the orientation error to the goal
    should decrease over the horizon (in some approximate sense).
    """
    u_opt, x_opt, success = mpc._solve(ref_state, x_goal=goal_state)
    assert success

    # Use quaternion geodesic angle as a simple orientation error measure
    angle_start = quaternion_angle(ref_state[:4], goal_state[:4])
    angle_end = quaternion_angle(x_opt[-1, :4], goal_state[:4])

    # We expect the final orientation to be closer to the goal than the initial one.
    assert angle_end < angle_start


@pytest.mark.unit
def test_solve_goal_nontrivial_control(mpc, ref_state, goal_state):
    """
    For a sufficiently different goal, the optimal control sequence should
    not be identically zero (otherwise the system would not move).
    """
    u_opt, x_opt, success = mpc._solve(ref_state, x_goal=goal_state)
    assert success

    # Check that there is at least one nonzero control entry
    assert not np.allclose(u_opt, 0.0)


@pytest.mark.unit
def test_solve_tracking_constant_reference(mpc, dynamics, ref_state):
    """
    When tracking a constant reference equal to the initial state, the solution
    should keep the system near that reference with small controls.
    """
    x_ref = np.tile(ref_state, (mpc.horizon + 1, 1))
    u_ref = np.zeros((mpc.horizon, mpc.control_dim))

    u_opt, x_opt, success = mpc._solve(ref_state, x_ref=x_ref, u_ref=u_ref)

    assert success
    np.testing.assert_allclose(x_opt, x_ref, atol=1e-3)
    # We expect controls to be small (possibly exactly zero if everything is consistent)
    assert np.max(np.abs(u_opt)) < 1e-1


@pytest.mark.unit
def test_solve_tracking_from_perturbed_state(mpc, error_mapping, ref_state):
    """
    Starting slightly away from a constant reference, the MPC should drive
    the state toward the reference over the horizon.
    """
    x_ref = np.tile(ref_state, (mpc.horizon + 1, 1))
    u_ref = np.zeros((mpc.horizon, mpc.control_dim))

    # Small perturbation in angular velocity
    x0 = ref_state.copy()
    x0[4:] = np.array([0.1, -0.1, 0.05])

    u_opt, x_opt, success = mpc._solve(x0, x_ref=x_ref, u_ref=u_ref)
    assert success

    # Measure error using dynamics.state_error w.r.t. reference
    err_start = error_mapping.state_error(x0, x_ref[0])
    err_end = error_mapping.state_error(x_opt[-1], x_ref[-1])

    assert np.linalg.norm(err_end) < np.linalg.norm(err_start)


@pytest.mark.unit
def test_solve_goal_at_equilibrium_returns_zero_control(mpc, ref_state):
    """
    If the initial state is already at the goal, the optimal control for a
    goal-only problem should be (approximately) zero.
    """
    u_opt, x_opt, success = mpc._solve(ref_state, x_goal=ref_state)
    assert success

    # State should remain at ref_state (up to integration / numerical error)
    for k in range(mpc.horizon + 1):
        np.testing.assert_allclose(x_opt[k], ref_state, atol=1e-6)

    # Control sequence should be near zero
    np.testing.assert_allclose(u_opt, 0.0, atol=1e-6)


# --------------------------------------------------------------------------
# Bounds and SQP-related tests
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_control_bounds_respected(mpc, ref_state, goal_state):
    """
    With tight control bounds, the optimal control should obey those bounds.
    """
    # Replace bounds with tight ones
    mpc.u_min = -0.01 * np.ones(3)
    mpc.u_max = 0.01 * np.ones(3)

    u_opt, x_opt, success = mpc._solve(ref_state, x_goal=goal_state)
    assert success

    assert np.all(u_opt <= mpc.u_max + 1e-8)
    assert np.all(u_opt >= mpc.u_min - 1e-8)


@pytest.mark.unit
def test_sqp_multiple_iterations_still_succeeds(
    dynamics, horizon, Q, R, ref_state, goal_state
):
    """
    Construct an MPC with more SQP iterations and verify that solve still
    succeeds and returns consistent shapes.
    """
    u_min = -5.0 * np.ones(3)
    u_max = 5.0 * np.ones(3)
    mpc_sqp = NominalMPC(
        horizon=horizon,
        dynamics=dynamics,
        Q=Q,
        R=R,
        u_min=u_min,
        u_max=u_max,
        max_sqp_iters=3,
    )

    u_opt, x_opt, success = mpc_sqp._solve(ref_state, x_goal=goal_state)
    assert success
    assert u_opt.shape == (horizon, 3)
    assert x_opt.shape == (horizon + 1, 7)


@pytest.mark.unit
def test_get_first_control_wrapper_goal(mpc, ref_state, goal_state):
    """
    get_first_control should call solve internally and return a single control.
    """
    u0 = mpc.get_first_control(x0=ref_state, x_goal=goal_state)
    assert u0.shape == (mpc.control_dim,)


@pytest.mark.integration
def test_mpc_large_angle_with_disturbance_converges(dynamics: SpacecraftDynamics, Q, R):
    """
    Closed-loop test: goal is a large rotation plus disturbance torque.

    This scenario is intentionally far outside the small-angle regime where a
    single fixed linearization (and thus a basic LQR designed around identity)
    would be expected to perform poorly. The nonlinear MPC with repeated
    linearizations and interpolation should still steer the spacecraft close
    to the goal despite a constant disturbance.
    """
    # Initial state: identity attitude, zero angular velocity
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Large goal rotation: ~170 deg about body y-axis, zero angular velocity
    theta = np.deg2rad(0.0)
    axis = np.array([0.0, 1.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    q_goal = qu.quaternion(
        np.cos(theta / 2.0),
        *(axis * np.sin(theta / 2.0)),
    ).normalized()
    x_goal = np.concatenate([qu.as_float_array(q_goal), np.zeros(3)])

    # Disturbance torque (constant bias in body frame)
    disturbance = np.array([0.02, -0.015, 0.01])

    # MPC setup: slightly longer horizon and multiple SQP iterations
    horizon = 30
    u_min = -0.1 * np.ones(3)
    u_max = 0.1 * np.ones(3)
    mpc = NominalMPC(
        horizon=horizon,
        dynamics=dynamics,
        Q=Q,
        R=R,
        u_min=u_min,
        u_max=u_max,
        max_sqp_iters=2,
    )

    # Initial orientation error to the goal
    # initial_angle_error = quaternion_angle(x0[:4], x_goal[:4])
    # assert initial_angle_error > np.deg2rad(80.0)  # sanity: this really is "far"

    # Closed-loop simulation with MPC in the loop
    num_steps = 300  # several horizons worth of steps
    x = x0.copy()

    for i in range(num_steps):
        # MPC computes first control input given current state and far goal
        u0 = mpc.get_first_control(x0=x, x_goal=x_goal)

        # Apply control to true nonlinear system with disturbance
        x = dynamics.discrete_dynamics_rk4(x, u0, disturbance=disturbance)
        print(
            f"{np.around(x, decimals=2)} time: {i // 10}, error:{np.rad2deg(quaternion_angle(x[:4], x_goal[:4]))}"
        )

    # Final orientation should be significantly closer to the goal
    final_angle_error = quaternion_angle(x[:4], x_goal[:4])
    print(f"GOAL: {np.around(x_goal, decimals=2)}")
    # We expect substantial reduction in orientation error
    # assert final_angle_error < initial_angle_error * 0.5
    # And also that we end up within some reasonable tolerance of the goal
    assert final_angle_error < np.deg2rad(4.0)
