"""Sensor noise models and immutable configuration containers."""

from dataclasses import dataclass, field

import numpy as np

from mpc_spacecraft.utilities.utils import ARCSEC_TO_RAD, DEG_TO_RAD, G_STD, HOUR_TO_SEC

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
