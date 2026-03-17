"""Tests for MPC prediction model adapters and interface parity."""

import numpy as np
import pytest
import quaternion as qu

from mpc_spacecraft.controllers.mpc_nominal_drake import NominalMPC
from mpc_spacecraft.controllers.prediction_adapters import (
    SpacecraftDynamicsPredictionAdapter,
)
from mpc_spacecraft.controllers.prediction_model import MPCPredictionModel
from mpc_spacecraft.dynamics.rigid_body import SpacecraftDynamics
from mpc_spacecraft.dynamics.rigid_body_error_constraints import (
    RigidBodyErrorConstraintBuilder,
)


@pytest.fixture
def inertia() -> np.ndarray:
    return np.array(
        [
            [90.0, -5.0, 3.0],
            [-5.0, 110.0, -4.0],
            [3.0, -4.0, 100.0],
        ]
    )


@pytest.fixture
def dt() -> float:
    return 0.1


@pytest.fixture
def dynamics(inertia: np.ndarray, dt: float) -> SpacecraftDynamics:
    return SpacecraftDynamics(inertia, dt)


@pytest.fixture
def ref_state() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def goal_state() -> np.ndarray:
    q_goal = qu.quaternion(0.707, 0.0, 0.707, 0.0).normalized()
    return np.concatenate([qu.as_float_array(q_goal), np.zeros(3)])


@pytest.fixture
def Q() -> np.ndarray:
    return np.diag([10.0, 10.0, 10.0, 3.0, 3.0, 3.0])


@pytest.fixture
def R() -> np.ndarray:
    return np.diag([0.1, 0.1, 0.1])


@pytest.mark.unit
def test_adapter_satisfies_prediction_protocol(dynamics: SpacecraftDynamics):
    adapter = SpacecraftDynamicsPredictionAdapter(dynamics)
    assert isinstance(adapter, MPCPredictionModel)


@pytest.mark.unit
def test_adapter_affine_step_matches_legacy_constraint(dynamics: SpacecraftDynamics):
    adapter = SpacecraftDynamicsPredictionAdapter(dynamics)
    builder = RigidBodyErrorConstraintBuilder(inertia=dynamics.inertia, dt=dynamics.dt)

    x_nom_k = np.array([1.0, 0.0, 0.0, 0.0, 0.02, -0.01, 0.015])
    u_nom_k = np.array([0.01, -0.02, 0.005])
    x_nom_kp1 = dynamics.discrete_dynamics_rk4(x_nom_k, u_nom_k)

    A_ref, B_ref, c_ref = builder.discrete_mpc_constraint(
        x_nom_k=x_nom_k,
        x_nom_kp1=x_nom_kp1,
        u_nom_k=u_nom_k,
        discrete_dynamics_step=dynamics.discrete_dynamics_rk4,
        discretize_linear_system=dynamics.discretize_linear_system,
    )
    step = adapter.affine_error_dynamics_step(x_nom_k, x_nom_kp1, u_nom_k)

    np.testing.assert_allclose(step.A, A_ref)
    np.testing.assert_allclose(step.B, B_ref)
    np.testing.assert_allclose(step.c, c_ref)


@pytest.mark.unit
def test_error_constraint_builder_matches_legacy_constraint(
    dynamics: SpacecraftDynamics,
):
    builder = RigidBodyErrorConstraintBuilder(inertia=dynamics.inertia, dt=dynamics.dt)
    adapter = SpacecraftDynamicsPredictionAdapter(dynamics)

    x_nom_k = np.array([1.0, 0.0, 0.0, 0.0, 0.02, -0.01, 0.015])
    u_nom_k = np.array([0.01, -0.02, 0.005])
    x_nom_kp1 = dynamics.discrete_dynamics_rk4(x_nom_k, u_nom_k)

    step = adapter.affine_error_dynamics_step(x_nom_k, x_nom_kp1, u_nom_k)
    A_new, B_new, c_new = builder.discrete_mpc_constraint(
        x_nom_k=x_nom_k,
        x_nom_kp1=x_nom_kp1,
        u_nom_k=u_nom_k,
        discrete_dynamics_step=dynamics.discrete_dynamics_rk4,
        discretize_linear_system=dynamics.discretize_linear_system,
    )

    np.testing.assert_allclose(A_new, step.A)
    np.testing.assert_allclose(B_new, step.B)
    np.testing.assert_allclose(c_new, step.c)


class _CountingPredictionAdapter(SpacecraftDynamicsPredictionAdapter):
    def __init__(self, dynamics: SpacecraftDynamics):
        super().__init__(dynamics)
        self.state_error_calls = 0
        self.state_error_batch_calls = 0
        self.state_from_error_calls = 0
        self.affine_step_calls = 0

    def state_error(self, state: np.ndarray, state_ref: np.ndarray) -> np.ndarray:
        self.state_error_calls += 1
        return super().state_error(state, state_ref)

    def state_error_batch(
        self, states: np.ndarray, states_ref: np.ndarray
    ) -> np.ndarray:
        self.state_error_batch_calls += 1
        return super().state_error_batch(states, states_ref)

    def state_from_error(
        self, delta_x: np.ndarray, state_ref: np.ndarray
    ) -> np.ndarray:
        self.state_from_error_calls += 1
        return super().state_from_error(delta_x, state_ref)

    def affine_error_dynamics_step(
        self,
        x_nom_k: np.ndarray,
        x_nom_kp1: np.ndarray,
        u_nom_k: np.ndarray,
    ):
        self.affine_step_calls += 1
        return super().affine_error_dynamics_step(x_nom_k, x_nom_kp1, u_nom_k)


@pytest.mark.unit
def test_nominal_mpc_uses_injected_prediction_model(
    dynamics: SpacecraftDynamics,
    ref_state: np.ndarray,
    goal_state: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
):
    adapter = _CountingPredictionAdapter(dynamics)
    mpc = NominalMPC(
        horizon=4,
        dynamics=dynamics,
        prediction_model=adapter,
        Q=Q,
        R=R,
        u_min=-10.0 * np.ones(3),
        u_max=10.0 * np.ones(3),
        max_sqp_iters=1,
    )

    _, _, success = mpc._solve(ref_state, x_goal=goal_state)
    assert success

    assert adapter.state_error_calls > 0
    assert adapter.state_error_batch_calls > 0
    assert adapter.state_from_error_calls > 0
    assert adapter.affine_step_calls == mpc.horizon


@pytest.mark.unit
def test_nominal_mpc_default_vs_explicit_adapter_parity(
    dynamics: SpacecraftDynamics,
    ref_state: np.ndarray,
    goal_state: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
):
    mpc_default = NominalMPC(
        horizon=5,
        dynamics=dynamics,
        Q=Q,
        R=R,
        u_min=-10.0 * np.ones(3),
        u_max=10.0 * np.ones(3),
        max_sqp_iters=1,
    )
    mpc_explicit = NominalMPC(
        horizon=5,
        dynamics=dynamics,
        prediction_model=SpacecraftDynamicsPredictionAdapter(dynamics),
        Q=Q,
        R=R,
        u_min=-10.0 * np.ones(3),
        u_max=10.0 * np.ones(3),
        max_sqp_iters=1,
    )

    u_default, x_default, success_default = mpc_default._solve(
        ref_state, x_goal=goal_state
    )
    u_explicit, x_explicit, success_explicit = mpc_explicit._solve(
        ref_state, x_goal=goal_state
    )

    assert success_default == success_explicit
    assert success_default

    np.testing.assert_allclose(u_default, u_explicit, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(x_default, x_explicit, atol=1e-9, rtol=0.0)
