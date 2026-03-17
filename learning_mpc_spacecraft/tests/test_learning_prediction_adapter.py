"""Targeted tests for learning/hybrid prediction adapter wiring."""

import numpy as np
import pytest
import torch

from mpc_spacecraft.controllers.mpc_learning_augmented import (
    HybridSpacecraftDynamics,
    LearningAugmentedMPC,
)
from mpc_spacecraft.controllers.prediction_adapters import (
    HybridSpacecraftPredictionAdapter,
)
from mpc_spacecraft.dynamics.rigid_body import SpacecraftDynamics
from mpc_spacecraft.learning.residual_model import ResidualDynamicsModel


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
def dynamics(inertia: np.ndarray) -> SpacecraftDynamics:
    return SpacecraftDynamics(inertia=inertia, dt=0.1)


@pytest.fixture
def residual_model() -> ResidualDynamicsModel:
    torch.manual_seed(0)
    return ResidualDynamicsModel(
        state_dim=7, control_dim=3, hidden_layers=[8], dropout=0.0
    )


@pytest.fixture
def q_weights() -> np.ndarray:
    return np.diag([10.0, 10.0, 10.0, 3.0, 3.0, 3.0])


@pytest.fixture
def r_weights() -> np.ndarray:
    return np.diag([0.1, 0.1, 0.1])


@pytest.mark.unit
def test_learning_mpc_uses_hybrid_prediction_adapter(
    dynamics: SpacecraftDynamics,
    residual_model: ResidualDynamicsModel,
    q_weights: np.ndarray,
    r_weights: np.ndarray,
):
    controller = LearningAugmentedMPC(
        horizon=4,
        dynamics=dynamics,
        Q=q_weights,
        R=r_weights,
        residual_model=residual_model,
        max_sqp_iters=1,
    )

    assert isinstance(controller.prediction_model, HybridSpacecraftPredictionAdapter)


@pytest.mark.unit
def test_hybrid_adapter_matches_identity_dynamics_when_residual_zero(
    dynamics: SpacecraftDynamics,
    residual_model: ResidualDynamicsModel,
):
    for param in residual_model.parameters():
        param.data.zero_()

    hybrid = HybridSpacecraftDynamics(
        base_dynamics=dynamics,
        residual_model=residual_model,
        residual_scale=1.0,
    )

    adapter = HybridSpacecraftPredictionAdapter(hybrid)

    x_nom_k = np.array([1.0, 0.0, 0.0, 0.0, 0.02, -0.01, 0.015])
    u_nom_k = np.array([0.01, -0.02, 0.005])
    x_nom_kp1 = hybrid.discrete_dynamics_rk4(x_nom_k, u_nom_k)

    step = adapter.affine_error_dynamics_step(x_nom_k, x_nom_kp1, u_nom_k)

    assert step.A.shape == (6, 6)
    assert step.B.shape == (6, 3)
    assert step.c.shape == (6,)

    np.testing.assert_allclose(step.c, np.zeros(6), atol=1e-5)
