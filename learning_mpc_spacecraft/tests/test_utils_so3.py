import numpy as np
import pytest

from src.mpc_spacecraft.utilities.utils import expm_so3, logm_so3, skew


@pytest.mark.unit
def test_expm_so3_identity():
    """Zero rotation vector should map to identity rotation matrix."""
    R = expm_so3(np.zeros(3))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


@pytest.mark.unit
def test_logm_so3_identity():
    """Identity rotation matrix should map to zero rotation vector."""
    theta = logm_so3(np.eye(3))
    np.testing.assert_allclose(theta, np.zeros(3), atol=1e-12)


@pytest.mark.unit
def test_expm_so3_small_angle_matches_series():
    """Small-angle expm should match first-order series: R ≈ I + [theta]_x."""
    theta = np.array([1e-6, -2e-6, 1.5e-6])
    R = expm_so3(theta)
    R_expected = np.eye(3) + skew(theta)
    np.testing.assert_allclose(R, R_expected, rtol=0, atol=1e-10)


@pytest.mark.unit
def test_logm_expm_roundtrip_random_axis():
    """logm(expm(theta)) should recover theta (principal angle) for moderate angles."""
    axis = np.array([0.3, -0.4, 0.5])
    axis = axis / np.linalg.norm(axis)
    angle = 1.2  # rad, comfortably within (0, pi)
    theta = axis * angle
    R = expm_so3(theta)
    theta_rec = logm_so3(R)
    np.testing.assert_allclose(theta_rec, theta, atol=1e-9, rtol=1e-9)


@pytest.mark.unit
def test_logm_so3_pi_rotation_axis():
    """logm should handle near-pi rotations and return principal axis*angle."""
    axis = np.array([1.0, 2.0, -1.0])
    axis = axis / np.linalg.norm(axis)
    angle = np.pi
    R = expm_so3(axis * angle)
    theta = logm_so3(R)
    # For pi rotations, axis sign is ambiguous: check magnitude and axis alignment.
    assert np.isclose(np.linalg.norm(theta), np.pi, atol=1e-6)
    axis_rec = theta / np.linalg.norm(theta)
    assert np.isclose(abs(np.dot(axis_rec, axis)), 1.0, atol=1e-6)
