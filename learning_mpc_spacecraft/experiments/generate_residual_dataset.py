import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import quaternion
from src.mpc_spacecraft.dynamics.rigid_body import SpacecraftDynamics
from src.mpc_spacecraft.dynamics.disturbances import DisturbanceModel
from tqdm import tqdm

# Define parameters
I_nominal = np.array([
    [90.0,  -5.0,   3.0],
    [-5.0, 110.0,  -4.0],
    [ 3.0,  -4.0, 100.0],
])  # kg*m^2

I_real = np.array([
    [120.0, -18.0,   9.0],
    [-18.0,  75.0, -12.0],
    [  9.0, -12.0, 135.0],
])

dt = 0.5
nominal_dynamics = SpacecraftDynamics(I_nominal, dt=dt)
real_dynamics = SpacecraftDynamics(I_real, dt=dt)

disturbance_model = DisturbanceModel(
    bias=np.array([0.05, -0.045, 0.03]),  # constant environmental torque bias [N·m]
    noise_std=0.003,                      # small white noise
    sinusoidal_amplitude=0.04,           # pretty noticeable sinusoid component
    sinusoidal_frequency=0.05,           # low frequency: 0.05 Hz (period = 20s)
    seed=42,
)

u_min = np.array([-0.10, -0.10, -0.10])  # N·m
u_max = np.array([+0.10, +0.10, +0.10])  # N·m

N = 100000  # Number of samples
np.random.seed(42)

states = []
controls = []
next_states = []
residuals = []

for _ in tqdm(range(N), desc="Generating samples"):
    # Generate random state: random unit quaternion + angular velocity in [-0.5, 0.5]
    # Random axis
    axis = np.random.randn(3)
    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
    else:
        axis = np.array([1.0, 0.0, 0.0])
    
    # Random angle in [0, 2*pi] for full coverage
    angle = np.random.uniform(0, 2 * np.pi)
    
    # Quaternion from axis-angle
    q = np.array([
        np.cos(angle / 2),
        axis[0] * np.sin(angle / 2),
        axis[1] * np.sin(angle / 2),
        axis[2] * np.sin(angle / 2)
    ])
    
    # Normalize quaternion
    q_norm = np.quaternion(*q).normalized()
    q = quaternion.as_float_array(q_norm)
    
    # Random angular velocity
    omega = np.random.uniform(-0.5, 0.5, 3)
    
    state = np.concatenate([q, omega])
    
    # Random control within limits
    control = np.random.uniform(u_min, u_max, 3)
    
    # Compute nominal next state (no disturbance)
    nominal_next = nominal_dynamics.discrete_dynamics_rk4(state, control)
    
    # Compute real next state with disturbance (random time for variety)
    t = np.random.uniform(0, 10000)  # Long time span to cover sinusoidal cycles
    disturbance = disturbance_model.get_disturbance(t)
    real_next = real_dynamics.discrete_dynamics_rk4(state, control, disturbance)
    
    # Residual: real_next - nominal_next
    res = real_next - nominal_next
    
    # Collect
    states.append(state)
    controls.append(control)
    next_states.append(real_next)
    residuals.append(res)

# Convert to numpy arrays
states = np.array(states)
controls = np.array(controls)
next_states = np.array(next_states)
residuals = np.array(residuals)

# Save to file in experiments/
output_path = 'experiments/datasets/residual_dataset_full.npz'
np.savez(output_path, states=states, controls=controls, next_states=next_states, residuals=residuals)

print(f"Generated {N} samples and saved to {output_path}")
print(f"Shapes: states {states.shape}, controls {controls.shape}, next_states {next_states.shape}, residuals {residuals.shape}")