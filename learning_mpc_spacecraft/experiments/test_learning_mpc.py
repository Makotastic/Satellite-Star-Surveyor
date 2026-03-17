import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import quaternion
import matplotlib.pyplot as plt
from tqdm import tqdm

from mpc_spacecraft.dynamics.rigid_body import SpacecraftDynamics
from src.mpc_spacecraft.dynamics.disturbances import DisturbanceModel
from src.mpc_spacecraft.learning.residual_model import ResidualDynamicsModel
from src.mpc_spacecraft.learning.dataset import DynamicsDataset
from src.mpc_spacecraft.controllers.mpc_learning_augmented import LearningAugmentedMPC

# Load configurations from generate_residual_dataset.py
I_nominal = np.array(
    [
        [90.0, -5.0, 3.0],
        [-5.0, 110.0, -4.0],
        [3.0, -4.0, 100.0],
    ]
)  # kg*m^2

I_real = np.array(
    [
        [120.0, -18.0, 9.0],
        [-18.0, 75.0, -12.0],
        [9.0, -12.0, 135.0],
    ]
)  # kg*m^2

dt = 0.1
u_min = np.array([-0.10, -0.10, -0.10])  # N·m
u_max = np.array([+0.10, +0.10, +0.10])  # N·m

# Disturbance parameters
disturbance_params = {
    "bias": np.array([0.02, -0.015, 0.01]),
    "noise_std": 0.003,
    "sinusoidal_amplitude": 0.04,
    "sinusoidal_frequency": 0.05,
    "seed": 42,
}

# Load dataset for normalization stats
data_path = "experiments/datasets/residual_dataset_stress.npz"
data = np.load(data_path)
states = data["states"]
controls = data["controls"]
next_states = data["next_states"]
residuals = data["residuals"]

dataset = DynamicsDataset(
    states=states,
    controls=controls,
    next_states=next_states,
    residuals=residuals,
    normalize=True,  # Compute normalization stats
)

# Load trained residual model
model_path = "experiments/models/residual_model_stress.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
residual_model = ResidualDynamicsModel.load(model_path, device=device)
print(f"Loaded residual model from {model_path} on {device}")

# Create dynamics models
nominal_dynamics = SpacecraftDynamics(I_nominal, dt=dt)
real_dynamics = SpacecraftDynamics(I_real, dt=dt)

disturbance_model = DisturbanceModel(**disturbance_params)

# MPC parameters
horizon = 15
Q = np.diag([10.0] * 3 + [1.0] * 3)  # Attitude error higher weight
R = 0.1 * np.eye(3)
Q_terminal = 10 * Q
max_sqp_iters = 2
residual_scale = 1.0

# Create LearningAugmentedMPC
mpc = LearningAugmentedMPC(
    horizon=horizon,
    dynamics=nominal_dynamics,
    Q=Q,
    R=R,
    residual_model=residual_model,
    normalizer=dataset,
    Q_terminal=Q_terminal,
    u_min=u_min,
    u_max=u_max,
    max_sqp_iters=max_sqp_iters,
    residual_scale=residual_scale,
    device=device,
)

# Simulation setup
num_steps = 200
t_start = 0.0

# Initial state: random quaternion with 45 deg error + small omega
# np.random.seed(42)
axis = np.random.randn(3)
if np.linalg.norm(axis) > 1e-6:
    axis /= np.linalg.norm(axis)
angle = np.pi / 6
q_init = np.array([np.cos(angle / 2), *axis * np.sin(angle / 2)])
omega_init = np.zeros(3)  # np.random.uniform(-0.03, 0.03, 3)
state = np.concatenate([q_init, omega_init])

# Goal state: identity quaternion + zero omega
goal_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Storage
states_history = [state.copy()]
controls_history = []
times = [t_start]

# Closed-loop simulation
state = state.copy()
for step in tqdm(range(num_steps), desc="Simulating Learning-Augmented MPC"):
    t = t_start + step * dt

    # Get control from MPC (goal regulation)
    control = mpc.get_first_control(state, x_goal=goal_state)

    # Simulate with real dynamics + disturbance
    disturbance = disturbance_model.get_disturbance(t)
    next_state = real_dynamics.discrete_dynamics_rk4_rotation(
        state, control, disturbance
    )

    # Store
    states_history.append(next_state.copy())
    controls_history.append(control.copy())
    times.append(t + dt)
    state = next_state

# Convert to arrays
states_history = np.array(states_history)
controls_history = np.array(controls_history)
times = np.array(times)  # Full times matching states

# Compute errors
angle_errors = []
velocity_errors = []
for s in states_history:
    # Angle error
    q = quaternion.as_quat_array(s[:4])
    q_goal = quaternion.as_quat_array(goal_state[:4])
    q_error = q_goal.conjugate() * q
    angle_error = 2 * np.arccos(np.clip(np.abs(q_error.w), -1.0, 1.0))
    angle_errors.append(angle_error)

    # Velocity error (norm of omega)
    omega_error = np.linalg.norm(s[4:])
    velocity_errors.append(omega_error)

angle_errors = np.array(angle_errors)
velocity_errors = np.array(velocity_errors)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(times, np.rad2deg(angle_errors), "b-", linewidth=2, label="Angle Error")
ax1.set_ylabel("Angle Error (degrees)")
ax1.set_xlabel("Time (s)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(times, velocity_errors, "r-", linewidth=2, label="||ω|| Error")
ax2.set_ylabel("Angular Velocity Magnitude (rad/s)")
ax2.set_xlabel("Time (s)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("Learning-Augmented MPC: Error Convergence Test")
plt.tight_layout()
plt.savefig("experiments/learning_mpc_convergence.png", dpi=150, bbox_inches="tight")
plt.show()

# Convergence check
final_angle = np.rad2deg(angle_errors[-1])
final_vel = velocity_errors[-1]
converged = final_angle < 5.0 and final_vel < 0.01
print(f"Final angle error: {final_angle:.2f}°")
print(f"Final velocity error: {final_vel:.4f} rad/s")
print(f"Converged: {'Yes' if converged else 'No'}")

# Save data
np.savez(
    "experiments/learning_mpc_results.npz",
    times=times,
    states=states_history,
    controls=controls_history,
    angle_errors=angle_errors,
    velocity_errors=velocity_errors,
)
print("Results saved to experiments/learning_mpc_results.npz")
