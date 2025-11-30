import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mpc_spacecraft.dynamics.rigid_body import SpacecraftDynamics
from mpc_spacecraft.controllers.lqr import LQRController
from mpc_spacecraft.controllers.mpc_nominal_drake import NominalMPC
from mpc_spacecraft.simulation.simulate_closed_loop import simulate_closed_loop, compare_controllers, SimulationLogger
from mpc_spacecraft.simulation.scenarios import ScenarioGenerator
from mpc_spacecraft.config.default_config import SpacecraftConfig, MPCConfig, LQRConfig, SimulationConfig
from mpc_spacecraft.analysis.plotting import plot_state_trajectory, plot_control_inputs, plot_angular_velocity
import matplotlib.pyplot as plt

# Step 1: Environment setup assumed (Docker/VSCode devcontainer)

# Step 2: Imports (done above)

# Step 3: Initialize spacecraft dynamics
config = SpacecraftConfig()
dynamics = SpacecraftDynamics(inertia=config.INERTIA, dt=config.DT)

# Step 4: Set up LQR controller
# Linearize around equilibrium (identity quaternion, zero angular velocity/control)
equilibrium_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
A_cont, B_cont = dynamics.linearize(equilibrium_state, np.zeros(3))
# Discretize for LQR (assuming discrete-time)
A_disc, B_disc = dynamics.discretize_linear_system(A_cont, B_cont)
lqr_config = LQRConfig()
lqr_controller = LQRController(A_disc, B_disc, lqr_config.Q, lqr_config.R, discrete=True)

# Step 5: Set up MPC controller
mpc_config = MPCConfig()
def dynamics_func(x, u):
    return dynamics.discrete_dynamics_rk4(x, u)
def linearizer_func(x_ref, u_ref):
    A_cont, B_cont = dynamics.linearize(x_ref, u_ref)
    A_disc, B_disc = dynamics.discretize_linear_system(A_cont, B_cont)
    return A_disc, B_disc
mpc_controller = NominalMPC(
    dynamics_func=dynamics_func,
    linearizer_func=linearizer_func,
    horizon=mpc_config.HORIZON,
    dt=config.DT,
    state_dim=7,
    control_dim=3,
    Q=mpc_config.Q,
    R=mpc_config.R,
    Q_terminal=mpc_config.Q_TERMINAL,
    u_min=np.full(3, mpc_config.U_MIN),
    u_max=np.full(3, mpc_config.U_MAX)
)

# Step 6: Define rest-to-rest scenario
sim_config = SimulationConfig()
duration = 10.0  # Short duration for testing
dt = sim_config.DT
q_initial = np.array([np.cos(np.pi/6), 0, 0, np.sin(np.pi/6)])  # 30 degree rotation around z
q_target = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
reference_trajectory = ScenarioGenerator().rest_to_rest(duration, dt, q_target)

# Step 7: Generate initial state
x0 = np.concatenate([q_initial, np.zeros(3)])

# Step 8 & 9: Run individual simulations
print("Running LQR simulation...")
lqr_logger = simulate_closed_loop(
    dynamics=dynamics,
    controller=lqr_controller,
    x0=x0,
    duration=duration,
    dt=dt,
    reference_trajectory=reference_trajectory,
    verbose=True
)

print("Running MPC simulation...")
mpc_logger = simulate_closed_loop(
    dynamics=dynamics,
    controller=mpc_controller,
    x0=x0,
    duration=duration,
    dt=dt,
    reference_trajectory=reference_trajectory,
    verbose=True
)

# Step 10: Compare controllers
controllers = {
    'LQR': lqr_controller,
    'MPC': mpc_controller
}
comparison_results = compare_controllers(
    dynamics=dynamics,
    controllers=controllers,
    x0=x0,
    duration=duration,
    dt=dt,
    reference_trajectory=reference_trajectory,
    verbose=True
)

# Step 11: Save simulation data
lqr_data = lqr_logger.to_arrays()
np.savez('experiments/lqr_simulation_data.npz', **lqr_data)

mpc_data = mpc_logger.to_arrays()
np.savez('experiments/mpc_simulation_data.npz', **mpc_data)

comparison_data = {name: logger.to_arrays() for name, logger in comparison_results.items()}
for name, data in comparison_data.items():
    np.savez(f'experiments/{name.lower()}_comparison_data.npz', **data)

print("Simulation data saved to experiments/ directory as .npz files.")

# Step 12: Generate plots (optional, for validation)
# LQR plots
fig1 = plot_state_trajectory(lqr_data['times'], lqr_data['states'], lqr_data['references'], title="LQR State Trajectory")
fig1.savefig('experiments/lqr_state_trajectory.png')
plt.close(fig1)

fig2 = plot_control_inputs(lqr_data['times'], lqr_data['controls'], title="LQR Controls")
fig2.savefig('experiments/lqr_controls.png')
plt.close(fig2)

fig3 = plot_angular_velocity(lqr_data['times'], lqr_data['states'][:, 4:], lqr_data['references'][:, 4:], title="LQR Angular Velocity")
fig3.savefig('experiments/lqr_angular_velocity.png')
plt.close(fig3)

# MPC plots (similar)
fig4 = plot_state_trajectory(mpc_data['times'], mpc_data['states'], mpc_data['references'], title="MPC State Trajectory")
fig4.savefig('experiments/mpc_state_trajectory.png')
plt.close(fig4)

fig5 = plot_control_inputs(mpc_data['times'], mpc_data['controls'], title="MPC Controls")
fig5.savefig('experiments/mpc_controls.png')
plt.close(fig5)

fig6 = plot_angular_velocity(mpc_data['times'], mpc_data['states'][:, 4:], mpc_data['references'][:, 4:], title="MPC Angular Velocity")
fig6.savefig('experiments/mpc_angular_velocity.png')
plt.close(fig6)

# Comparison plot example (first state component)
fig7, ax = plt.subplots(figsize=(10, 6))
for name, data in comparison_data.items():
    ax.plot(data['times'], data['states'][:, 0], label=f'{name} q_w')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Quaternion w')
ax.legend()
ax.grid(True)
fig7.suptitle('Controller Comparison: Quaternion w')
fig7.savefig('experiments/comparison_quaternion_w.png')
plt.close(fig7)

print("Plots generated in experiments/ directory for validation.")

# Step 13: Validation complete - data ready for custom plotting
print("Setup complete. Load .npz files with np.load() for custom analysis and plotting.")