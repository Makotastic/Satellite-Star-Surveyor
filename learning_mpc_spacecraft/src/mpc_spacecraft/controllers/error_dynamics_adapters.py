from typing import cast

import numpy as np

from mpc_spacecraft.utilities.utils import (
    FloatArray,
    RotErrState,
    RotState,
    quat_rotate_vector_a_to_b,
    BODY_FORWARD_VEC3,
)
from mpc_spacecraft.guidance.sun_tracker import AstropySunDirectionModel
from ..dynamics.rigid_body import SpacecraftDynamics
from ..dynamics.rigid_body_error_constraints import RigidBodyErrorConstraintBuilder
from .error_state_mapping import ErrorStateMappingService
from .error_dynamics_providers import AffineErrorDynamicsStep, ErrorDynamicsProvider


class SpacecraftErrorDynamicsProvider(ErrorDynamicsProvider):
    """Adapter that exposes `SpacecraftDynamics` through `MPCPredictionModel`.

    This intentionally delegates to existing methods in
    `SpacecraftDynamics` so behavior remains unchanged while we decouple
    controller code from concrete plant implementations.
    """

    def __init__(
        self,
        dynamics: SpacecraftDynamics,
        sun_model: AstropySunDirectionModel,
        sun_bc_margin_rad: float,
    ):
        self.dynamics = dynamics
        self._sun_model = sun_model
        self._sun_margin = sun_bc_margin_rad
        self._error_mapping = ErrorStateMappingService()
        self._constraint_builder = RigidBodyErrorConstraintBuilder(
            inertia=dynamics.inertia,
            dt=dynamics.dt,
        )

    def state_error(self, state: RotState, state_ref: RotState) -> RotErrState:
        return self._error_mapping.state_error(state, state_ref)

    def state_error_batch(
        self,
        states: RotState,
        states_ref: RotState,
    ) -> RotErrState:
        return self._error_mapping.state_error_batch(states, states_ref)

    def state_from_error(self, delta_x: RotErrState, state_ref: RotState) -> RotState:
        return self._error_mapping.state_from_error(delta_x, state_ref)

    def state_from_error_batch(
        self,
        delta_xs: RotErrState,
        states_ref: RotState,
    ) -> RotState:
        return self._error_mapping.state_from_error_batch(delta_xs, states_ref)

    def affine_error_dynamics_step(
        self,
        x_nom_k: RotState,
        x_nom_kp1: RotState,
        u_nom_k: FloatArray,
    ) -> AffineErrorDynamicsStep:
        A_k, B_k, c_k = self._constraint_builder.discrete_mpc_constraint(
            x_nom_k=x_nom_k,
            x_nom_kp1=x_nom_kp1,
            u_nom_k=u_nom_k,
            discrete_dynamics_step=self.dynamics.discrete_dynamics_rk4_rotation,
            discretize_linear_system=self.dynamics.discretize_linear_system,
        )
        return AffineErrorDynamicsStep(
            A=cast(FloatArray, A_k),
            B=cast(FloatArray, B_k),
            c=cast(FloatArray, c_k),
        )

    def affine_error_theta_bc(self, x_nom_k: RotState) -> tuple[FloatArray, FloatArray]:
        sun_vec = self._sun_model.sun_dir_eci()
        sun_quat = quat_rotate_vector_a_to_b(BODY_FORWARD_VEC3, sun_vec)
        return self._constraint_builder.affine_error_theta_bc(
            x_nom_k, sun_quat, self._sun_margin
        )


class HybridErrorDynamicsProvider(SpacecraftErrorDynamicsProvider):
    """Scaffolding adapter for hybrid/learned dynamics.

    Uses finite-difference linearization of the hybrid one-step map in
    error coordinates around the current nominal trajectory point.
    """

    def __init__(self, dynamics: SpacecraftDynamics, epsilon: float = 1e-6):
        super().__init__(dynamics)
        self.epsilon = float(epsilon)

    def affine_error_dynamics_step(
        self,
        x_nom_k: RotState,
        x_nom_kp1: RotState,
        u_nom_k: FloatArray,
    ) -> AffineErrorDynamicsStep:
        n_err = 6
        m_ctrl = int(u_nom_k.shape[0])

        A = np.zeros((n_err, n_err), dtype=float)
        B = np.zeros((n_err, m_ctrl), dtype=float)

        dx0 = np.zeros(n_err, dtype=float)
        du0 = np.zeros(m_ctrl, dtype=float)

        def f_err(delta_x: FloatArray, delta_u: FloatArray) -> FloatArray:
            state = self.state_from_error(delta_x, x_nom_k)
            control = u_nom_k + delta_u
            state_next = self.dynamics.discrete_dynamics_rk4_rotation(state, control)
            return self.state_error(state_next, x_nom_kp1)

        c = f_err(dx0, du0)

        for i in range(n_err):
            dx_p = dx0.copy()
            dx_m = dx0.copy()
            dx_p[i] += self.epsilon
            dx_m[i] -= self.epsilon

            f_p = f_err(dx_p, du0)
            f_m = f_err(dx_m, du0)
            A[:, i] = (f_p - f_m) / (2.0 * self.epsilon)

        for i in range(m_ctrl):
            du_p = du0.copy()
            du_m = du0.copy()
            du_p[i] += self.epsilon
            du_m[i] -= self.epsilon

            f_p = f_err(dx0, du_p)
            f_m = f_err(dx0, du_m)
            B[:, i] = (f_p - f_m) / (2.0 * self.epsilon)

        return AffineErrorDynamicsStep(
            A=cast(FloatArray, A),
            B=cast(FloatArray, B),
            c=cast(FloatArray, c),
        )
