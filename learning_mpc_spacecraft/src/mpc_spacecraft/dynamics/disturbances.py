"""Disturbance models for spacecraft dynamics."""

import numpy as np
from typing import Optional


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
        bias: Optional[np.ndarray] = None,
        noise_std: float = 0.0,
        sinusoidal_amplitude: float = 0.0,
        sinusoidal_frequency: float = 0.0,
        seed: Optional[int] = None
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
    
    def get_disturbance(self, time: float) -> np.ndarray:
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
            sinusoidal = self.sinusoidal_amplitude * np.array([
                np.sin(phase),
                np.sin(phase + 2*np.pi/3),
                np.sin(phase + 4*np.pi/3)
            ])
            disturbance += sinusoidal
        
        return disturbance
    
    def get_atmospheric_drag(
        self,
        velocity: np.ndarray,
        altitude: float,
        drag_coefficient: float = 2.2,
        reference_area: float = 1.0
    ) -> np.ndarray:
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
    
    def get_gravity_gradient(
        self,
        quaternion: np.ndarray,
        orbital_radius: float,
        mu: float = 3.986e14  # Earth's gravitational parameter (m^3/s^2)
    ) -> np.ndarray:
        """
        Compute gravity gradient torque.
        
        Args:
            quaternion: Spacecraft attitude quaternion [w, x, y, z]
            orbital_radius: Distance from central body (m)
            mu: Gravitational parameter of central body (m^3/s^2)
            
        Returns:
            Gravity gradient torque [tau_g_x, tau_g_y, tau_g_z] (N*m)
        """
        # This is a simplified model
        # Full implementation would require inertia matrix and nadir vector
        
        # Orbital angular velocity
        n = np.sqrt(mu / orbital_radius**3)
        
        # Simplified gravity gradient torque
        # Actual implementation would use: tau_g = 3*n^2 * (z_nadir x I * z_nadir)
        # For now, return zeros as placeholder
        return np.zeros(3)
    
    def get_solar_radiation_pressure(
        self,
        sun_direction: np.ndarray,
        solar_constant: float = 1361.0,  # W/m^2
        reflectivity: float = 0.5,
        area: float = 1.0
    ) -> np.ndarray:
        """
        Compute solar radiation pressure torque.
        
        Args:
            sun_direction: Unit vector pointing to sun in body frame
            solar_constant: Solar irradiance (W/m^2)
            reflectivity: Surface reflectivity (0-1)
            area: Exposed area (m^2)
            
        Returns:
            Solar pressure torque [tau_s_x, tau_s_y, tau_s_z] (N*m)
        """
        c = 299792458  # Speed of light (m/s)
        
        # Solar pressure force
        pressure = solar_constant / c
        force = pressure * area * (1 + reflectivity)
        
        # Simplified torque (assuming center of pressure offset)
        lever_arm = 0.1  # m
        torque = lever_arm * force * sun_direction
        
        return torque