"""Controllers module for spacecraft attitude control."""

from .lqr import LQRController
from .mpc_nominal_drake import NominalMPC
# from .mpc_learning_augmented import LearningAugmentedMPC
from .error_state_mapping import ErrorStateMappingService

__all__ = [
    "LQRController",
    "NominalMPC",
    # "LearningAugmentedMPC",
    "ErrorStateMappingService",
]
