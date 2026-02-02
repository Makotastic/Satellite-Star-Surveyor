"""Disturbance models for spacecraft dynamics."""

import numpy as np
from numpy.linalg import norm

from mpc_spacecraft.utilities.utils import FloatArray, Vec3


M_EARTH = 5.972e24  # kg
G_CONST = 6.674e-11


class DisturbanceModel:
    """
    Model for external disturbances acting on the spacecraft.

    Supports various disturbance types:
    - Constant bias
    - Random noise
    - Sinusoidal disturbances
    - Atmospheric drag (simplified)
    """

    def __init__(
        self,
        bias: np.ndarray | None = None,
        noise_std: float = 0.0,
        sinusoidal_amplitude: float = 0.0,
        sinusoidal_frequency: float = 0.0,
        seed: int | None = None,
    ):
        """
        Initialize disturbance model.

        Args:
            bias: Constant bias torque [tau_x, tau_y, tau_z] (N*m)
            noise_std: Standard deviation of random noise (N*m)
            sinusoidal_amplitude: Amplitude of sinusoidal disturbance (N*m)
            sinusoidal_frequency: Frequency of sinusoidal disturbance (rad/s)
            seed: Random seed for reproducibility
        """
        self.bias = bias if bias is not None else np.zeros(3)
        self.noise_std = noise_std
        self.sinusoidal_amplitude = sinusoidal_amplitude
        self.sinusoidal_frequency = sinusoidal_frequency

        if seed is not None:
            np.random.seed(seed)

    def get_disturbance(self, time: float) -> FloatArray:
        """
        Compute total disturbance torque at given time.

        Args:
            time: Current simulation time (s)

        Returns:
            Disturbance torque [tau_d_x, tau_d_y, tau_d_z] (N*m)
        """
        disturbance = self.bias.copy()

        # Add random noise
        if self.noise_std > 0:
            disturbance += np.random.normal(0, self.noise_std, 3)

        # Add sinusoidal component
        if self.sinusoidal_amplitude > 0:
            phase = self.sinusoidal_frequency * time
            sinusoidal = self.sinusoidal_amplitude * np.array(
                [
                    np.sin(phase),
                    np.sin(phase + 2 * np.pi / 3),
                    np.sin(phase + 4 * np.pi / 3),
                ]
            )
            disturbance += sinusoidal

        return disturbance


def get_atmospheric_drag(
    velocity: FloatArray,
    altitude: float,
    drag_coefficient: float = 2.2,
    reference_area: float = 1.0,
) -> FloatArray:
    """
    Compute simplified atmospheric drag torque.

    Args:
        velocity: Velocity vector in body frame (m/s)
        altitude: Altitude above surface (m)
        drag_coefficient: Drag coefficient (dimensionless)
        reference_area: Reference area (m^2)

    Returns:
        Drag torque [tau_d_x, tau_d_y, tau_d_z] (N*m)
    """
    # Simplified atmospheric density model (exponential)
    rho_0 = 1.225  # kg/m^3 at sea level
    H = 8500  # Scale height (m)
    rho = rho_0 * np.exp(-altitude / H)

    # Drag force
    v_mag = np.linalg.norm(velocity)
    if v_mag < 1e-6:
        return np.zeros(3)

    drag_force = 0.5 * rho * drag_coefficient * reference_area * v_mag**2

    # Simplified torque (assuming center of pressure offset)
    # In reality, this would depend on spacecraft geometry
    lever_arm = 0.1  # m
    drag_torque = lever_arm * drag_force * (velocity / v_mag)

    return drag_torque


def jacobian_gravity(pos: Vec3) -> FloatArray:
    """
    Compute the Jacobian of the central-gravity acceleration w.r.t. position.

    Args:
        pos: Inertial position vector (m).

    Returns:
        3x3 Jacobian matrix of gravitational acceleration (1/s^2).
    """
    r_vec = pos - np.zeros(3)
    return -(G_CONST * M_EARTH) * (
        (1 / norm(r_vec) ** 3) * np.eye(3)
        - (3 / norm(r_vec) ** 5) * np.outer(r_vec, r_vec)
    )


def gravity(pos: Vec3) -> Vec3:
    """
    Compute central-gravity acceleration at a position.

    Args:
        pos: Inertial position vector (m).

    Returns:
        Gravitational acceleration vector (m/s^2).
    """
    r_vec = pos - np.zeros(3)
    return -((G_CONST * M_EARTH) / (norm(r_vec) ** 3)) * r_vec
