"""Learning-augmented MPC controller.

Defines:
    - HybridSpacecraftDynamics: SpacecraftDynamics subclass that adds a
      learned residual to the nominal rigid-body dynamics and linearizes
      the resulting *hybrid* discrete-time error dynamics.

    - LearningAugmentedMPC: A thin subclass of NominalMPC that simply
      plugs HybridSpacecraftDynamics into the existing MPC formulation.
"""

from typing import Optional, Tuple

import numpy as np
import quaternion
import torch

from ..dynamics.rigid_body import SpacecraftDynamics
from ..learning.residual_model import ResidualDynamicsModel
from ..learning.dataset import DynamicsDataset
from .error_state_mapping import ErrorStateMappingService
from .mpc_nominal_drake import NominalMPC
from .prediction_adapters import HybridSpacecraftPredictionAdapter


class HybridSpacecraftDynamics(SpacecraftDynamics):
    """
    Spacecraft dynamics with a learned residual model.

    The hybrid discrete-time dynamics used by MPC are:

        x_{k+1} = f_nominal(x_k, u_k) + alpha * f_residual(x_k, u_k)

    where:
        - f_nominal is the nominal rigid-body dynamics (RK4 discretization)
        - f_residual is a neural network residual model
        - alpha = residual_scale is a trust factor in [0, 1].

    Additionally, the linearization used by MPC is ALWAYS done on the
    hybrid *discrete-time* error dynamics:

        delta_x_{k+1} ≈ A_d * delta_x_k + B_d * delta_u_k

    where delta_x is the 6D error state [delta_theta, delta_omega].
    """

    def __init__(
        self,
        base_dynamics: SpacecraftDynamics,
        residual_model: ResidualDynamicsModel,
        normalizer: Optional[DynamicsDataset] = None,
        residual_scale: float = 1.0,
        device: str = "cpu",
    ):
        """
        Args:
            base_dynamics: Nominal spacecraft dynamics (for inertia, dt, etc.).
            residual_model: Trained residual dynamics model.
            normalizer: Optional DynamicsDataset (or similar) object exposing
                `normalize`, `input_mean`, `input_std`, `output_mean`,
                `output_std` for normalization.
            residual_scale: Trust factor alpha applied to residual output.
            device: Torch device ("cpu" or "cuda").
        """
        # Initialize as a standard SpacecraftDynamics with same inertia, dt.
        super().__init__(inertia=base_dynamics.inertia, dt=base_dynamics.dt)

        self.residual_model = residual_model.to(device)
        self.normalizer = normalizer
        self.residual_scale = float(residual_scale)
        self.device = device
        self.error_mapping = ErrorStateMappingService()

        # Optional: move normalizer stats to device
        if self.normalizer is not None and getattr(self.normalizer, "normalize", False):
            self.normalizer.input_mean = self.normalizer.input_mean.to(device)
            self.normalizer.input_std = self.normalizer.input_std.to(device)
            self.normalizer.output_mean = self.normalizer.output_mean.to(device)
            self.normalizer.output_std = self.normalizer.output_std.to(device)

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------
    def set_device(self, device: str = "cpu") -> None:
        """Move residual model (and normalizer stats) to a given device."""
        self.device = device
        self.residual_model = self.residual_model.to(device)

        if self.normalizer is not None and getattr(self.normalizer, "normalize", False):
            self.normalizer.input_mean = self.normalizer.input_mean.to(device)
            self.normalizer.input_std = self.normalizer.input_std.to(device)
            self.normalizer.output_mean = self.normalizer.output_mean.to(device)
            self.normalizer.output_std = self.normalizer.output_std.to(device)

    # ------------------------------------------------------------------
    # Residual model helpers
    # ------------------------------------------------------------------
    def _build_model_input(
        self, state: np.ndarray, control: np.ndarray
    ) -> torch.Tensor:
        """Create a (1, input_dim) normalized tensor [state, control]."""
        inp = np.concatenate([state, control], axis=-1)
        inp_t = torch.as_tensor(inp, dtype=torch.float32, device=self.device)

        if self.normalizer is None or not getattr(self.normalizer, "normalize", False):
            return inp_t.unsqueeze(0)

        mean = self.normalizer.input_mean
        std = self.normalizer.input_std
        return ((inp_t - mean) / std).unsqueeze(0)

    def _denormalize_residual(self, res_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized residual prediction back to physical state units."""
        if self.normalizer is None or not getattr(self.normalizer, "normalize", False):
            return res_norm

        return res_norm * self.normalizer.output_std + self.normalizer.output_mean

    # ------------------------------------------------------------------
    # Hybrid discrete-time dynamics
    # ------------------------------------------------------------------
    def discrete_dynamics_rk4(
        self,
        state: np.ndarray,
        control: np.ndarray,
        disturbance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Hybrid RK4 discrete step:

            x_{k+1} = x_nominal_next + alpha * residual(x_k, u_k)

        where x_nominal_next is the nominal RK4 step from SpacecraftDynamics.
        """
        # 1) nominal step (super() uses continuous_dynamics + RK4)
        x_nom_next = super().discrete_dynamics_rk4(state, control, disturbance)

        # 2) residual prediction
        self.residual_model.eval()
        with torch.no_grad():
            inp = self._build_model_input(state, control)  # [1, input_dim]
            res_norm = self.residual_model(inp)  # [1, state_dim]
            res = self._denormalize_residual(res_norm)  # [1, state_dim]

        res_np = res.detach().cpu().numpy().squeeze(0)

        # 3) hybrid next state
        x_next = x_nom_next + self.residual_scale * res_np

        # 4) Normalize quaternion part
        q = quaternion.quaternion(*x_next[:4]).normalized()
        x_next[:4] = quaternion.as_float_array(q)

        return x_next

    def discrete_dynamics_euler(
        self,
        state: np.ndarray,
        control: np.ndarray,
        disturbance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Hybrid forward-Euler step.

        Provided for completeness; MPC uses RK4 for nominal trajectory
        building, but in case Euler is used anywhere, we keep behavior
        consistent.
        """
        # 1) nominal Euler step
        state_dot = self.continuous_dynamics(state, control, disturbance)
        x_nom_next = state + self.dt * state_dot

        # 2) residual prediction
        self.residual_model.eval()
        with torch.no_grad():
            inp = self._build_model_input(state, control)
            res_norm = self.residual_model(inp)
            res = self._denormalize_residual(res_norm)

        res_np = res.detach().cpu().numpy().squeeze(0)
        x_next = x_nom_next + self.residual_scale * res_np

        # Normalize quaternion
        q = quaternion.quaternion(*x_next[:4]).normalized()
        x_next[:4] = quaternion.as_float_array(q)

        return x_next

    # ------------------------------------------------------------------
    # Hybrid linearization (always on, discrete-time error dynamics)
    # ------------------------------------------------------------------
    def linearize(
        self,
        state_ref: np.ndarray,
        control_ref: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the hybrid *discrete-time* error dynamics around a reference.

        Returns A_d, B_d for:

            delta_x_{k+1} ≈ A_d * delta_x_k + B_d * delta_u_k

        where delta_x = [delta_theta, delta_omega] (6D error state).

        This is a finite-difference linearization using the hybrid RK4
        step defined above.
        """
        n = 6  # error state dimension
        m = 3  # control dimension

        A_d = np.zeros((n, n))
        B_d = np.zeros((n, m))

        epsilon = 1e-6

        delta_x0 = np.zeros(n)
        delta_u0 = np.zeros(m)

        def f_err(delta_x: np.ndarray, delta_u: np.ndarray) -> np.ndarray:
            """
            Map (delta_x, delta_u) to next-step error in 6D coordinates,
            using hybrid discrete dynamics.
            """
            # Map error to full state/control
            state = self.error_mapping.state_from_error(delta_x, state_ref)
            control = control_ref + delta_u

            # One-step hybrid discrete dynamics
            state_next = self.discrete_dynamics_rk4(state, control)

            # Error of next state w.r.t. same reference
            err_next = self.error_mapping.state_error(state_next, state_ref)
            return err_next

        # Columns of A_d: derivative w.r.t. delta_x
        for i in range(n):
            dx_p = delta_x0.copy()
            dx_m = delta_x0.copy()
            dx_p[i] += epsilon
            dx_m[i] -= epsilon

            f_p = f_err(dx_p, delta_u0)
            f_m = f_err(dx_m, delta_u0)

            A_d[:, i] = (f_p - f_m) / (2.0 * epsilon)

        # Columns of B_d: derivative w.r.t. delta_u
        for i in range(m):
            du_p = delta_u0.copy()
            du_m = delta_u0.copy()
            du_p[i] += epsilon
            du_m[i] -= epsilon

            f_p = f_err(delta_x0, du_p)
            f_m = f_err(delta_x0, du_m)

            B_d[:, i] = (f_p - f_m) / (2.0 * epsilon)

        return A_d, B_d

    def discretize_linear_system(
        self,
        A: np.ndarray,
        B: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For the hybrid model, `linearize` already returns discrete-time
        A_d, B_d. So this is a no-op to keep the MPC interface intact.
        """
        return A, B


class LearningAugmentedMPC(NominalMPC):
    """
    MPC controller that uses HybridSpacecraftDynamics as its model.

    This is essentially a NominalMPC where:
        - Dynamics are hybrid (nominal + learned residual).
        - Linearization is always performed on the hybrid discrete-time
          error dynamics (via HybridSpacecraftDynamics.linearize).
        - Nominal trajectories, warm-starts, etc. are all based on the
          hybrid dynamics, so the optimized hybrid trajectory is what
          gets hot-started from one MPC cycle to the next.
    """

    def __init__(
        self,
        horizon: int,
        dynamics: SpacecraftDynamics,
        Q: np.ndarray,
        R: np.ndarray,
        residual_model: ResidualDynamicsModel,
        normalizer: Optional[DynamicsDataset] = None,
        Q_terminal: Optional[np.ndarray] = None,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
        x_min: Optional[np.ndarray] = None,
        x_max: Optional[np.ndarray] = None,
        max_sqp_iters: int = 2,
        residual_scale: float = 1.0,
        device: str = "cpu",
    ):
        """
        Args:
            horizon: Prediction horizon.
            dynamics: Nominal spacecraft dynamics model.
            dt: Sampling time (should match dynamics.dt).
            Q: State error cost matrix (6x6).
            R: Control error cost matrix (3x3).
            residual_model: Trained residual dynamics model.
            normalizer: Optional DynamicsDataset with normalization stats.
            Q_terminal: Terminal state error cost matrix.
            u_min, u_max: Control bounds.
            x_min, x_max: State bounds (currently unused).
            max_sqp_iters: Number of SQP iterations in goal-only MPC.
            residual_scale: Trust factor for learned residual.
            device: Torch device ("cpu" or "cuda").
        """

        # Wrap the nominal dynamics in a hybrid dynamics object.
        hybrid_dynamics = HybridSpacecraftDynamics(
            base_dynamics=dynamics,
            residual_model=residual_model,
            normalizer=normalizer,
            residual_scale=residual_scale,
            device=device,
        )

        # Initialize the parent MPC with the hybrid dynamics.
        super().__init__(
            horizon=horizon,
            dynamics=hybrid_dynamics,
            prediction_model=HybridSpacecraftPredictionAdapter(hybrid_dynamics),
            Q=Q,
            R=R,
            Q_terminal=Q_terminal,
            u_min=u_min,
            u_max=u_max,
            x_min=x_min,
            x_max=x_max,
            max_sqp_iters=max_sqp_iters,
        )

        # For convenience, keep typed access to the hybrid dynamics.
        self.dynamics: HybridSpacecraftDynamics = hybrid_dynamics

    def set_device(self, device: str = "cpu") -> None:
        """
        Convenience method to move the residual model (and normalizer
        stats) to a different device via the dynamics object.
        """
        self.dynamics.set_device(device)
