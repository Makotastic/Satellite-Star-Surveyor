"""Dynamics module for spacecraft rigid body motion."""

from .quaternion import (
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from .rigid_body import SpacecraftDynamics
from .disturbances import DisturbanceModel

__all__ = [
    "quaternion_multiply",
    "quaternion_conjugate",
    "quaternion_normalize",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "SpacecraftDynamics",
    "DisturbanceModel",
]