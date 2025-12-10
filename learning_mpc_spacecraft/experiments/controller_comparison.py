import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import quaternion
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.mpc_spacecraft.dynamics.rigid_body import SpacecraftDynamics
from src.mpc_spacecraft.dynamics.disturbances import DisturbanceModel
from src.mpc_spacecraft.learning.residual_model import ResidualDynamicsModel
from src.mpc_spacecraft.learning.dataset import DynamicsDataset
from src.mpc_spacecraft.controllers.lqr import LQRController
from src.mpc_spacecraft.controllers.mpc_nominal_drake import NominalMPC
from src.mpc_spacecraft.controllers.mpc_learning_augmented import LearningAugmentedMPC
from src.mpc_spacecraft.analysis import (
    compute_tracking_error, compute_control_effort, compute_stability_metrics,
    compute_quaternion_trajectory_errors, compute_velocity_errors, compute_performance_index,
    compare_controllers, plot_quaternion_trajectory, plot_angular_velocity,
    plot_control_inputs, plot_performance_comparison, plot_error_comparison, plot_all_comparisons
)

# Toggles for controllers
run_lqr = True
run_nominal_mpc = True
run_learning_mpc = True

# Shared configurations
I_nominal = np.array([
    [90.0,  -5.0,   3.0],
    [-5.0, 110.0,  -4.0],
    [ 3.0,  -4.0, 100.0],
])  # kg*m^2

I_real_full = np.array([
    [120.0, -18.0,   9.0],
    [-18.0,  75.0, -12.0],
    [  9.0, -12.0, 135.0],
]) # kg*m^2 full%

I_real_5 = np.array([
    [ 91.5,  -5.65,   3.30],
    [ -5.65, 108.25, -4.40],
    [  3.30, -4.40, 101.75],
])

I_real_15 = np.array([
    [94.5 , -6.95,  3.90],
    [-6.95,104.75, -5.20],
    [ 3.90, -5.20,105.25],
])

I_real = np.array([
    [ 97.5 ,  -8.25,   4.50],
    [ -8.25, 101.25,  -6.00],
    [  4.50,  -6.00, 108.75],
])



dt = 0.5
u_min = np.array([-0.10, -0.10, -0.10])  # N·m
u_max = np.array([+0.10, +0.10, +0.10])  # N·m

disturbance_params = {
    'bias': np.array([0.05, -0.045, 0.03]),
    'noise_std': 0.003,
    'sinusoidal_amplitude': 0.04,
    'sinusoidal_frequency': 0.05,
    'seed': 42,
}

# Load dataset for normalization (learning MPC)
data_path = 'experiments/datasets/residual_dataset_25.npz'
data = np.load(data_path)
states = data['states']
controls = data['controls']
next_states = data['next_states']
residuals = data['residuals']

dataset = DynamicsDataset(
    states=states,
    controls=controls,
    next_states=next_states,
    residuals=residuals,
    normalize=True
)

# Load residual model for learning MPC
model_path = 'experiments/models/residual_model_25.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
residual_model = ResidualDynamicsModel.load(model_path, device=device)

# Create dynamics and disturbance
nominal_dynamics = SpacecraftDynamics(I_nominal, dt=dt)
real_dynamics = SpacecraftDynamics(I_real, dt=dt)
disturbance_model = DisturbanceModel(**disturbance_params)

# Shared parameters
num_steps = 700
t_start = 0.0

# Initial state: 45° quaternion error + small omega
# np.random.seed(42)
axis = np.random.randn(3)
axis /= np.linalg.norm(axis)
angle = np.pi / 4  # 45 degrees
q_init = np.array([
    np.cos(angle / 2),
    axis[0] * np.sin(angle / 2),
    axis[1] * np.sin(angle / 2),
    axis[2] * np.sin(angle / 2)
])
omega_init = np.random.uniform(-0.02, 0.02, 3)
initial_state = np.concatenate([q_init, omega_init])

# Goal state: identity quaternion + zero omega
goal_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# MPC parameters (shared for nominal and learning)
horizon = 30
Q = np.diag([10.0] * 3 + [1.0] * 3)  # Higher weight on attitude error
R = 0.1 * np.eye(3)
Q_terminal = 10 * Q
max_sqp_iters = 2
residual_scale = 1.0

# Create controllers if toggled
controllers = {}
results = {}

if run_lqr:
    # Linearize nominal dynamics around goal for LQR (error dynamics)
    zero_control = np.zeros(3)
    A, B = nominal_dynamics.linearize(goal_state, zero_control)
    Ad, Bd = nominal_dynamics.discretize_linear_system(A, B)
    lqr = LQRController(Ad, Bd, Q, R, discrete=True)
    controllers['lqr'] = lqr
    print("Created LQR controller")

if run_nominal_mpc:
    nominal_mpc = NominalMPC(
        horizon=horizon,
        dynamics=nominal_dynamics,
        Q=Q,
        R=R,
        Q_terminal=Q_terminal,
        u_min=u_min,
        u_max=u_max,
        max_sqp_iters=max_sqp_iters
    )
    controllers['nominal_mpc'] = nominal_mpc
    print("Created Nominal MPC controller")

if run_learning_mpc:
    learning_mpc = LearningAugmentedMPC(
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
        device=device
    )
    controllers['learning_mpc'] = learning_mpc
    print("Created Learning-Augmented MPC controller")

def simulate_closed_loop(controller, initial_state, goal_state, num_steps, dynamics, disturbance_model, dt, controller_name):
    """
    Simulate closed-loop with given controller.
    
    Args:
        controller: Controller instance
        initial_state: Initial full state [7]
        goal_state: Goal full state [7]
        num_steps: Number of simulation steps
        dynamics: Real dynamics for simulation
        disturbance_model: Disturbance model
        dt: Timestep
        controller_name: Name for logging
        
    Returns:
        dict with 'times', 'states', 'controls'
    """
    states = [initial_state.copy()]
    controls = []
    times = [0.0]
    state = initial_state.copy()
    
    for step in tqdm(range(num_steps), desc=f"Simulating {controller_name}"):
        t = step * dt
        
        # Compute control
        if controller_name == 'lqr':
            error_state = nominal_dynamics.state_error(state, goal_state)  # Use nominal for linearization
            control = controller.compute_control(error_state)
            control = np.clip(control, u_min, u_max)
        else:
            control = controller.get_first_control(state, x_goal=goal_state)
        
        # Simulate with real dynamics + disturbance
        disturbance = disturbance_model.get_disturbance(t)
        next_state = dynamics.discrete_dynamics_rk4(state, control, disturbance)
        
        states.append(next_state.copy())
        controls.append(control.copy())
        times.append(t + dt)
        state = next_state
    
    return {
        'times': np.array(times),
        'states': np.array(states),
        'controls': np.array(controls)
    }

# Run simulations
for name, controller in controllers.items():
    if name in ['lqr', 'nominal_mpc', 'learning_mpc']:
        sim_data = simulate_closed_loop(
            controller, initial_state, goal_state, num_steps, 
            real_dynamics, disturbance_model, dt, name
        )
        results[name] = sim_data
        print(f"Completed simulation for {name}")

# Create constant reference trajectory
ref_states = np.tile(goal_state, (num_steps + 1, 1))

# Compute metrics for each controller
comparison_metrics = {}
for name, data in results.items():
    states = data['states']
    controls = data['controls']
    times = data['times']
    
    # Tracking error (for full state, but quat needs special handling)
    tracking = compute_tracking_error(states, ref_states)
    
    # Control effort
    effort = compute_control_effort(controls, dt)
    
    # Stability
    stability = compute_stability_metrics(states, equilibrium=goal_state)
    
    # Custom errors
    quat_errors = compute_quaternion_trajectory_errors(states, goal_state[:4])
    vel_errors = compute_velocity_errors(states)
    
    # Add to data for plotting
    data['quaternion_errors'] = quat_errors
    data['velocity_errors'] = vel_errors
    data['references'] = ref_states

    
    # Compile metrics
    metrics = {**tracking, **effort, **stability}
    metrics['quat_rmse'] = np.sqrt(np.mean(quat_errors**2))
    metrics['vel_rmse'] = np.sqrt(np.mean(vel_errors**2))
    
    comparison_metrics[name] = metrics

# Overall comparison
overall_comparison = compare_controllers(results, dt)
print("Comparison metrics computed")

# Generate plots
# 2. Angular velocity
# plot_angular_velocity(
#     results['lqr']['times'], results['lqr']['states'][:, 4:],
#     references=ref_states[:, 4:],
#     title="Angular Velocity Comparison",
#     save_path='experiments/omega_comparison.png'
# )

# 3. Control inputs (example for LQR, but can loop for all)
# plot_control_inputs(
#     results['lqr']['times'][:-1], results['lqr']['controls'],
#     limits=(u_min, u_max),
#     title="Control Inputs Comparison (LQR)",
#     save_path='experiments/controls_lqr.png'
# )

# Define consistent color scheme
colors = {
    'lqr': "#ce0f0f",  # blue
    'nominal_mpc': "#29a76c",  # green
    'learning_mpc': "#2462E9"  # orange
}

# Combined comparison plot
plot_all_comparisons(
    results,
    comparison_metrics,
    colors,
    metric_names=['rmse_total', 'quat_rmse', 'vel_rmse'],
    limits=(u_min, u_max),
    title="Controller Comparison Dashboard",
    save_path='experiments/combined_controller_comparison_25_45degree.png'
)

# Save all results
all_results = {name: {**results[name], 'metrics': comparison_metrics[name]} for name in results}
np.savez('experiments/controller_comparison_results_25_45degree.npz', **all_results)
print("All plots and results saved to experiments/")
# plt.show()
print("Comparison complete. Check the combined plot for analysis.")