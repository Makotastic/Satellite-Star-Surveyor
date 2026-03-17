"""Closed-loop simulation utilities."""

from dataclasses import dataclass, field
import pickle
from typing import Callable

import numpy as np

from mpc_spacecraft.utilities.utils import FloatArray


@dataclass
class SimulationLogger:
    """Logger for simulation data."""

    times: list[float] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    controls: list[np.ndarray] = field(default_factory=list)
    references: list[np.ndarray] = field(default_factory=list)
    residuals: list[np.ndarray] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)

    def log_step(
        self,
        time: float,
        state: np.ndarray,
        control: np.ndarray,
        reference: np.ndarray | None = None,
        residual: np.ndarray | None = None,
        cost: float | None = None,
    ):
        """Log a single simulation step."""
        self.times.append(time)
        self.states.append(state.copy())
        self.controls.append(control.copy())

        if reference is not None:
            self.references.append(reference.copy())
        if residual is not None:
            self.residuals.append(residual.copy())
        if cost is not None:
            self.costs.append(cost)

    def to_arrays(self) -> dict[str, FloatArray]:
        """Convert logged data to numpy arrays."""
        data = {
            "times": np.array(self.times),
            "states": np.array(self.states),
            "controls": np.array(self.controls),
        }

        if self.references:
            data["references"] = np.array(self.references)
        if self.residuals:
            data["residuals"] = np.array(self.residuals)
        if self.costs:
            data["costs"] = np.array(self.costs)

        return data

    def save(self, filepath: str):
        """Save logged data to file."""
        data = self.to_arrays()

        if filepath.endswith(".npz"):
            np.savez(filepath, **data)
        elif filepath.endswith(".pkl"):
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SimulationLogger":
        """Load logged data from file."""
        if filepath.endswith(".npz"):
            data = np.load(filepath)
        elif filepath.endswith(".pkl"):
            with open(filepath, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        logger = cls()
        logger.times = data["times"].tolist()
        logger.states = data["states"].tolist()
        logger.controls = data["controls"].tolist()

        if "references" in data:
            logger.references = data["references"].tolist()
        if "residuals" in data:
            logger.residuals = data["residuals"].tolist()
        if "costs" in data:
            logger.costs = data["costs"].tolist()

        return logger


def simulate_closed_loop(
    dynamics,
    controller,
    x0: FloatArray,
    duration: float,
    dt: float,
    reference_trajectory: Callable | None = None,
    disturbance_model: object | None = None,
    logger: SimulationLogger | None = None,
    verbose: bool = True,
) -> SimulationLogger:
    """
    Run closed-loop simulation.

    Args:
        dynamics: Dynamics object with discrete_dynamics method
        controller: Controller object with compute_control or get_first_control method
        x0: Initial state
        duration: Simulation duration (s)
        dt: Timestep (s)
        reference_trajectory: Function t -> x_ref (optional)
        disturbance_model: Disturbance model object (optional)
        logger: SimulationLogger to use (creates new one if None)
        verbose: Whether to print progress

    Returns:
        SimulationLogger with recorded data
    """
    if logger is None:
        logger = SimulationLogger()

    # Initialize
    state = x0.copy()
    time = 0.0
    num_steps = int(duration / dt)

    if verbose:
        print(f"Running simulation for {duration}s ({num_steps} steps)")

    # Simulation loop
    for step in range(num_steps):
        # Get reference
        if reference_trajectory is not None:
            x_ref = reference_trajectory(time)
        else:
            x_ref = np.array([1, 0, 0, 0, 0, 0, 0])

        # Compute control
        if hasattr(controller, "get_first_control"):
            # MPC controller
            control = controller.get_first_control(state, x_ref)
        elif hasattr(controller, "compute_control"):
            # LQR or other feedback controller
            control = controller.compute_control(state, x_ref)
        else:
            raise ValueError(
                "Controller must have get_first_control or compute_control method"
            )

        # Get disturbance
        disturbance = None
        if disturbance_model is not None:
            disturbance = disturbance_model.get_disturbance(time)

        # Log current step
        logger.log_step(time, state, control, x_ref)

        # Propagate dynamics
        if hasattr(dynamics, "discrete_dynamics_rk4_rotation"):
            state = dynamics.discrete_dynamics_rk4_rotation(state, control, disturbance)
        else:
            raise ValueError("Dynamics must have discrete_dynamics method")

        # Update time
        time += dt

        if verbose and (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{num_steps} (t={time:.2f}s)")

    if verbose:
        print("Simulation complete")

    return logger


def compare_controllers(
    dynamics,
    controllers: dict[str, object],
    x0: np.ndarray,
    duration: float,
    dt: float,
    reference_trajectory: Callable | None = None,
    disturbance_model: object | None = None,
    verbose: bool = True,
) -> dict[str, SimulationLogger]:
    """
    Compare multiple controllers on the same scenario.

    Args:
        dynamics: Dynamics object
        controllers: Dictionary of {name: controller}
        x0: Initial state
        duration: Simulation duration
        dt: Timestep
        reference_trajectory: Reference trajectory function
        disturbance_model: Disturbance model
        verbose: Whether to print progress

    Returns:
        Dictionary of {name: SimulationLogger}
    """
    results = {}

    for name, controller in controllers.items():
        if verbose:
            print(f"\n=== Simulating {name} ===")

        logger = simulate_closed_loop(
            dynamics=dynamics,
            controller=controller,
            x0=x0,
            duration=duration,
            dt=dt,
            reference_trajectory=reference_trajectory,
            disturbance_model=disturbance_model,
            verbose=verbose,
        )

        results[name] = logger

    return results
