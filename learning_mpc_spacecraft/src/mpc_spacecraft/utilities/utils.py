from collections.abc import Sequence
from typing import TypeAlias
import numpy as np
import quaternion
from numpy.typing import NDArray

# A type alias for a float64 vector with exactly 3 elements.
# Note: NumPy typing cannot fully encode shape constraints at runtime.
Vec3: TypeAlias = NDArray[np.float64]

FloatArray: TypeAlias = NDArray[np.float64]

RotErrState: TypeAlias = FloatArray
RotState: TypeAlias = FloatArray

Quat: TypeAlias = quaternion.quaternion  # type: ignore[attr-defined]

# Unit conversions
ARCSEC_TO_RAD = np.pi / 648000.0
DEG_TO_RAD = np.pi / 180.0
G_STD = 9.80665
HOUR_TO_SEC = 3600.0

I3 = np.eye(3)
z3 = np.zeros((3, 3))


def as_vec3(v: Sequence[float] | FloatArray) -> Vec3:
    """Return a validated 3-element float64 vector.

    Raises:
        ValueError: If the input cannot be coerced to shape (3,) or contains
            non-finite values.
    """
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Vector contains NaN or infinite values.")
    return arr


def skew(v: Sequence[float] | FloatArray) -> FloatArray:
    """Return the 3x3 skew-symmetric matrix of a 3D vector."""
    vec = as_vec3(v)
    mat = np.zeros((3, 3), dtype=np.float64)
    mat[0, 1] = -vec[2]
    mat[0, 2] = vec[1]
    mat[1, 0] = vec[2]
    mat[1, 2] = -vec[0]
    mat[2, 0] = -vec[1]
    mat[2, 1] = vec[0]
    return mat


def jacob_r_lie(theta: Vec3) -> FloatArray:
    skew_theta = skew(theta)
    return I3 - (1 / 2) * skew_theta + (1 / 6) * (skew_theta @ skew_theta)


def jacob_r_lie_inv(theta: Vec3) -> FloatArray:
    skew_theta = skew(theta)
    return I3 + (1 / 2) * skew_theta + (1 / 12) * (skew_theta @ skew_theta)


def expm_so3(theta: Vec3) -> FloatArray:
    """Exponential map for SO(3) using Rodrigues' formula.

    Args:
        theta: Rotation vector (axis * angle in radians).

    Returns:
        Rotation matrix in SO(3).
    """
    vec = as_vec3(theta)
    angle_sq = float(vec @ vec)
    angle = float(np.sqrt(angle_sq))
    skew_theta = skew(vec)

    if angle < 1e-8:
        # Series expansion for small angles.
        a = 1.0 - angle_sq / 6.0 + (angle_sq * angle_sq) / 120.0
        b = 0.5 - angle_sq / 24.0 + (angle_sq * angle_sq) / 720.0
    else:
        a = np.sin(angle) / angle
        b = (1.0 - np.cos(angle)) / angle_sq

    return I3 + a * skew_theta + b * (skew_theta @ skew_theta)


def logm_so3(R: FloatArray) -> Vec3:
    """Logarithm map for SO(3).

    Args:
        R: Rotation matrix in SO(3).

    Returns:
        Rotation vector (axis * angle) with principal angle in [0, pi].
    """
    cos_angle = (np.trace(R) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))

    if angle < 1e-8:
        # For very small angles, use first-order approximation.
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64,
        )

    if np.pi - angle < 1e-6:
        # Near pi, use diagonal elements to extract axis robustly.
        axis = np.empty(3, dtype=np.float64)
        axis[0] = np.sqrt(max(0.0, (R[0, 0] + 1.0) * 0.5))
        axis[1] = np.sqrt(max(0.0, (R[1, 1] + 1.0) * 0.5))
        axis[2] = np.sqrt(max(0.0, (R[2, 2] + 1.0) * 0.5))

        # Choose signs based on off-diagonal terms.
        if R[2, 1] - R[1, 2] < 0.0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0.0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0.0:
            axis[2] = -axis[2]

        norm = np.linalg.norm(axis)
        if norm > 0.0:
            axis /= norm
        return axis * angle

    factor = angle / (2.0 * np.sin(angle))
    return factor * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
