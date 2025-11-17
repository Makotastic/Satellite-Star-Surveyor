"""Quaternion utilities for spacecraft attitude representation."""

import numpy as np
from typing import Tuple


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.
    
    Quaternion format: [w, x, y, z] where w is the scalar part.
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        
    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute the conjugate of a quaternion.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion to unit length.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Normalized quaternion
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    q = quaternion_normalize(q)
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return quaternion_normalize(np.array([w, x, y, z]))


def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Compute the time derivative of a quaternion given angular velocity.
    
    Args:
        q: Quaternion [w, x, y, z]
        omega: Angular velocity vector [wx, wy, wz] in body frame
        
    Returns:
        Quaternion derivative [dw/dt, dx/dt, dy/dt, dz/dt]
    """
    # Omega matrix for quaternion kinematics
    Omega = np.array([
        [0, -omega[0], -omega[1], -omega[2]],
        [omega[0], 0, omega[2], -omega[1]],
        [omega[1], -omega[2], 0, omega[0]],
        [omega[2], omega[1], -omega[0], 0]
    ])
    
    return 0.5 * Omega @ q


def quaternion_error(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """
    Compute the quaternion error between current and reference quaternions.
    
    Args:
        q: Current quaternion [w, x, y, z]
        q_ref: Reference quaternion [w, x, y, z]
        
    Returns:
        Error quaternion [w, x, y, z]
    """
    q_ref_conj = quaternion_conjugate(q_ref)
    q_error = quaternion_multiply(q_ref_conj, q)
    return q_error