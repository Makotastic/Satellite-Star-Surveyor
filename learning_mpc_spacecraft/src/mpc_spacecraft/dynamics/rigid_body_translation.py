
from mpc_spacecraft.utilities.utils import (
    FloatArray,
    TransState,
    Vec3,
    I3,
    z3,
    IDX_POS,
    IDX_VEL,
)

import numpy as np

class TranslationalDynamics:

    def __init__(self, mass: float, dt: float):

        self.mass = mass
        self.dt = dt

    def continuous_dynamics(
            self,
            state: TransState,
            acceleration: Vec3,
        ) -> FloatArray:
            
            state_dt = np.zeros(6)
            
            state_dt[IDX_POS] = state[IDX_VEL]
            state_dt[IDX_VEL] = acceleration
            
            return state_dt

    def discrete_dynamics_rk4(
        self,
        state: TransState,
        acceleration: Vec3, 
    ) -> TransState:
        """
        Compute discrete-time dynamics using RK4 integration.

        Args:
            state: Current state
            acceleration: Applied acceleration

        Returns:
            Next state
        """
        # RK4 integration
        k1 = self.continuous_dynamics(state, acceleration)
        k2 = self.continuous_dynamics(state + 0.5 * self.dt * k1, acceleration)
        k3 = self.continuous_dynamics(state + 0.5 * self.dt * k2, acceleration)
        k4 = self.continuous_dynamics(state + self.dt * k3, acceleration)

        next_state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return next_state