from collections.abc import Sequence
from typing import TypeAlias
import numpy as np
import quaternion
from numpy.typing import NDArray
from .array_view_generic import ArrayView

Vec3: TypeAlias = NDArray[np.float64]
FloatArray: TypeAlias = NDArray[np.float64]
Quat: TypeAlias = quaternion.quaternion

class TranslationState(ArrayView):
    position: FloatArray
    velocity: FloatArray

    __fields__ = [
        ("position", 3),
        ("velocity", 3),
    ]


class RotationState(ArrayView):
    quat: FloatArray
    omega: FloatArray

    __fields__ = [
        ("quat", 4),
        ("omega", 3),
    ]

    __defaults__ = {
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }

class RotationErrorState(ArrayView):
    error_angle: FloatArray
    omega: FloatArray

    __fields__ = [
        ("error_angle", 3),
        ("omega", 3),
    ]


class TranslationControl(ArrayView):
    acceleration: FloatArray

    __fields__ = [
        ("acceleration", 3),
    ]


class RotationControl(ArrayView):
    torque: FloatArray

    __fields__ = [
        ("torque", 3),
    ]


class RigidBodyControl(ArrayView):
    translation: TranslationControl
    rotation: RotationControl
    acceleration: FloatArray
    torque: FloatArray

    __fields__ = [
        ("translation", TranslationControl),
        ("rotation", RotationControl),
    ]

    __aliases__ = {
        "acceleration": "translation.acceleration",
        "torque": "rotation.torque",
    }


class RigidBodyState(ArrayView):
    translation: TranslationState
    rotation: RotationState
    position: FloatArray
    velocity: FloatArray
    quat: FloatArray
    omega: FloatArray

    __fields__ = [
        ("translation", TranslationState),
        ("rotation", RotationState),
    ]

    __aliases__ = {
        "position": "translation.position",
        "velocity": "translation.velocity",
        "quat": "rotation.quat",
        "omega": "rotation.omega",
    }


class SensorBiasState(ArrayView):
    accel: FloatArray
    gyro: FloatArray

    __fields__ = [
        ("accel", 3),
        ("gyro", 3),
    ]

class SensorRigidBodyState(ArrayView):
    position: FloatArray
    velocity: FloatArray
    quat: FloatArray
    omega: FloatArray
    translation: TranslationState
    rotation: RotationState
    gyro_bias: FloatArray
    accel_bias: FloatArray
    rigid_body: FloatArray

    __fields__ = [
        ("rigid_body", RigidBodyState),
        ("sensor_bias", SensorBiasState),
    ]

    __aliases__ = {
        "position": "rigid_body.position",
        "velocity": "rigid_body.velocity",
        "quat": "rigid_body.quat",
        "omega": "rigid_body.omega",
        "gyro_bias": "sensor_bias.gyro",
        "accel_bias": "sensor_bias.accel",
        "translation": "rigid_body.translation",
        "rotation": "rigid_body.rotation",
    }

class MeasuredState(ArrayView):
    quat: FloatArray
    omega: FloatArray
    position: FloatArray
    velocity: FloatArray
    translation: TranslationState
    rotation: RotationState
    inertial_accel: FloatArray

    __fields__ = [
        ("rigid_body", RigidBodyState),
        ("inertial_accel", 3)
    ]

    __aliases__ = {
        "position": "rigid_body.position",
        "velocity": "rigid_body.velocity",
        "quat": "rigid_body.quat",
        "omega": "rigid_body.omega",
        "translation": "rigid_body.translation",
        "rotation": "rigid_body.rotation",
    }


class FullSimState(ArrayView):
    sensor_rigid_body: SensorRigidBodyState
    inertial_accel: FloatArray
    position: FloatArray
    velocity: FloatArray
    quat: FloatArray
    omega: FloatArray
    gyro_bias: FloatArray
    accel_bias: FloatArray
    translation: TranslationState
    rotation: RotationState
    rigid_body: RigidBodyState

    __fields__ = [
        ("sensor_rigid_body", SensorRigidBodyState),
        ("inertial_accel", 3)
    ]

    __aliases__ = {
        "position": "sensor_rigid_body.position",
        "velocity": "sensor_rigid_body.velocity",
        "quat": "sensor_rigid_body.quat",
        "omega": "sensor_rigid_body.omega",
        "gyro_bias": "sensor_rigid_body.gyro_bias",
        "accel_bias": "sensor_rigid_body.accel_bias",
        "translation": "sensor_rigid_body.translation",
        "rotation": "sensor_rigid_body.rotation",
        "rigid_body": "sensor_rigid_body.rigid_body",
    }


# Unit conversions
ARCSEC_TO_RAD = np.pi / 648000.0
DEG_TO_RAD = np.pi / 180.0
G_STD = 9.80665
HOUR_TO_SEC = 3600.0

I3 = np.eye(3)
z3 = np.zeros((3, 3))

BODY_FORWARD_VEC3 = np.array([1.0, 0.0, 0.0])

R_EARTH_M = 6.378e6
M_EARTH = 5.972e24  # kg
G_CONST = 6.674e-11


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


def unit(v: FloatArray, axis=-1):
    n = np.linalg.norm(v, axis=axis)
    if np.any(n == 0):
        raise ValueError("Norm is zero")
    return v / n


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


def unskew(mat: FloatArray) -> Vec3:
    """Return the 3D vector associated with a 3x3 skew-symmetric matrix.

    This is the inverse operation of `skew` for valid skew-symmetric inputs.

    Raises:
        ValueError: If the input is not a finite 3x3 skew-symmetric matrix.
    """
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected shape (3, 3), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Matrix contains NaN or infinite values.")
    if not np.allclose(arr + arr.T, 0.0, atol=1e-12):
        raise ValueError("Expected a skew-symmetric matrix.")

    return np.array([arr[2, 1], arr[0, 2], arr[1, 0]], dtype=np.float64)


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


def quat_rotate_vector_a_to_b(body_forward_vec: Vec3, goal_vec: Vec3) -> Quat:
    goal_u = unit(np.asarray(goal_vec, dtype=float))
    body_fwd_u = unit(np.asarray(body_forward_vec, dtype=float))

    dot = float(np.clip(np.dot(body_fwd_u, goal_u), -1.0, 1.0))

    # Aligned: identity rotation.
    if dot > 1.0 - 1e-8:
        return quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

    # Anti-parallel: deterministic 180deg about a fixed orthogonal axis.
    if dot < -1.0 + 1e-8:
        basis = (
            np.array([1.0, 0.0, 0.0])
            if abs(body_fwd_u[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        axis = unit(np.cross(body_fwd_u, basis))
        return quaternion.from_rotation_vector(axis * np.pi)

    axis = unit(np.cross(body_fwd_u, goal_u))
    theta = np.arccos(dot)
    return quaternion.from_rotation_vector(axis * theta)
