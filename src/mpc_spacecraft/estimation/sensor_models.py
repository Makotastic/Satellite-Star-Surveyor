"""Sensor noise models and immutable configuration containers."""

from dataclasses import dataclass, field

import numpy as np

from mpc_spacecraft.utilities.utils import ARCSEC_TO_RAD, DEG_TO_RAD, G_STD, HOUR_TO_SEC

MEKF_ERROR_STATE_SIZE = 15

MEKF_IDX_R = slice(0, 3)
MEKF_IDX_V = slice(3, 6)
MEKF_IDX_THETA = slice(6, 9)
MEKF_IDX_BG = slice(9, 12)
MEKF_IDX_BA = slice(12, 15)


def make_mekf_initial_covariance(
    position_sigma: float | np.ndarray = 1.0,
    velocity_sigma: float | np.ndarray = 0.1,
    attitude_sigma: float | np.ndarray = 1.0 * DEG_TO_RAD,
    gyro_bias_sigma: float | np.ndarray = (0.02 * DEG_TO_RAD) / HOUR_TO_SEC,
    accel_bias_sigma: float | np.ndarray = 50e-6 * G_STD,
) -> np.ndarray:
    """Build a 15x15 MEKF initial covariance from 1-sigma uncertainties.

    The covariance matches the MEKF error-state ordering:
    ``dx = [dr, dv, dtheta, dbg, dba].T``.

    Args:
        position_sigma: Initial position 1-sigma uncertainty in m.
        velocity_sigma: Initial velocity 1-sigma uncertainty in m/s.
        attitude_sigma: Initial attitude-error 1-sigma uncertainty in rad.
        gyro_bias_sigma: Initial gyro-bias 1-sigma uncertainty in rad/s.
        accel_bias_sigma: Initial accelerometer-bias 1-sigma uncertainty in m/s^2.

    Scalars apply the same uncertainty to all three axes; arrays can specify
    per-axis values.
    """
    P = np.zeros((MEKF_ERROR_STATE_SIZE, MEKF_ERROR_STATE_SIZE))
    P[MEKF_IDX_R, MEKF_IDX_R] = np.diag(np.broadcast_to(position_sigma, 3) ** 2)
    P[MEKF_IDX_V, MEKF_IDX_V] = np.diag(np.broadcast_to(velocity_sigma, 3) ** 2)
    P[MEKF_IDX_THETA, MEKF_IDX_THETA] = np.diag(np.broadcast_to(attitude_sigma, 3) ** 2)
    P[MEKF_IDX_BG, MEKF_IDX_BG] = np.diag(np.broadcast_to(gyro_bias_sigma, 3) ** 2)
    P[MEKF_IDX_BA, MEKF_IDX_BA] = np.diag(np.broadcast_to(accel_bias_sigma, 3) ** 2)
    return P

# GNSS measurement noise (1-sigma)
_sigma_gnss_pos = 0.3  # m
_sigma_gnss_vel = 0.03  # m/s


@dataclass(frozen=True)
class GNSSConfig:
    R: np.ndarray = field(
        default_factory=lambda: np.diag(
            [
                _sigma_gnss_pos**2,
                _sigma_gnss_pos**2,
                _sigma_gnss_pos**2,
                _sigma_gnss_vel**2,
                _sigma_gnss_vel**2,
                _sigma_gnss_vel**2,
            ]
        )
    )


# Star tracker measurement noise (1-sigma)
_sigma_st = 5.0 * ARCSEC_TO_RAD  # 5 arcsec in radians


@dataclass(frozen=True)
class StarTrackerConfig:
    R: np.ndarray = field(default_factory=lambda: np.eye(3) * (_sigma_st**2))


# Accelerometer noise (1-sigma)
_sigma_a = 20e-6 * G_STD  # m/s^2/sqrt(Hz)
_sigma_ba = 50e-6 * G_STD  # m/s^2 (1-sigma bias)


@dataclass(frozen=True)
class IMUConfig:
    sigma_a2: np.ndarray = field(default_factory=lambda: np.eye(3) * (_sigma_a**2))
    sigma_ba2: np.ndarray = field(
        default_factory=lambda: np.eye(3) * ((_sigma_ba**2) / HOUR_TO_SEC)
    )


# Gyro noise (1-sigma)
_sigma_g = (0.005 * DEG_TO_RAD) / np.sqrt(HOUR_TO_SEC)  # rad/sqrt(s)
_sigma_bg = (0.02 * DEG_TO_RAD) / HOUR_TO_SEC  # rad/s (1-sigma bias)


@dataclass(frozen=True)
class GyroConfig:
    sigma_g2: np.ndarray = field(default_factory=lambda: np.eye(3) * (_sigma_g**2))
    sigma_bg2: np.ndarray = field(
        default_factory=lambda: np.eye(3) * ((_sigma_bg**2) / HOUR_TO_SEC)
    )
