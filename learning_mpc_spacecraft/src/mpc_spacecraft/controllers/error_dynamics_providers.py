"""Controller-facing prediction model contracts for MPC.

This module introduces a small, stable interface layer so controllers can
depend on *what they need* (error mapping + affine error dynamics) instead of a
concrete dynamics implementation.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mpc_spacecraft.utilities.utils import FloatArray, RotErrState, RotState


@dataclass(frozen=True)
class AffineErrorDynamicsStep:
    """Affine error dynamics for one horizon step.

    Represents:
        delta_x_{k+1} = A @ delta_x_k + B @ delta_u_k + c
    """

    A: FloatArray
    B: FloatArray
    c: FloatArray


@runtime_checkable
class ErrorDynamicsProvider(Protocol):
    """Port interface required by MPC formulations in this package."""

    def state_error(self, state: RotState, state_ref: RotState) -> RotErrState:
        """Map full states to 6D error coordinates."""
        ...

    def state_error_batch(
        self,
        states: RotState,
        states_ref: RotState,
    ) -> RotErrState:
        """Vectorized variant of :meth:`state_error`."""
        ...

    def state_from_error(self, delta_x: RotErrState, state_ref: RotState) -> RotState:
        """Reconstruct full state from 6D error coordinates and reference."""
        ...

    def affine_error_dynamics_step(
        self,
        x_nom_k: RotState,
        x_nom_kp1: RotState,
        u_nom_k: FloatArray,
    ) -> AffineErrorDynamicsStep:
        """Return one-step affine error dynamics used by MPC constraints."""
        ...

    def affine_error_theta_bc(self, x_nom_k: RotState) -> tuple[FloatArray, FloatArray]:
        """
        The outputs should be used in a linear inequality of the form

        ``prog.AddLinearConstraint(2.0 * alpha_k.dot(theta_k) - s[k] <= b_k)``

        where ``s[k]`` is an optional slack variable.
        """
        ...
