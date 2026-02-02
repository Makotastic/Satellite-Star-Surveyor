from collections.abc import Sequence
from typing import TypeAlias
import numpy as np
import quaternion

from numpy.typing import NDArray

# A type alias for a float64 vector with exactly 3 elements.
# Note: NumPy typing cannot fully encode shape constraints at runtime.
Vec3: TypeAlias = NDArray[np.float64]

FloatArray: TypeAlias = NDArray[np.float64]

Quat: TypeAlias = quaternion.quaternion  # type: ignore[attr-defined]

# Unit conversions
ARCSEC_TO_RAD = np.pi / 648000.0
DEG_TO_RAD = np.pi / 180.0
G_STD = 9.80665
HOUR_TO_SEC = 3600.0


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
