"""Simulation module for closed-loop control."""

from .simulate_closed_loop import simulate_closed_loop, SimulationLogger
from .scenarios import ScenarioGenerator, create_reference_trajectory

__all__ = [
    "simulate_closed_loop",
    "SimulationLogger",
    "ScenarioGenerator",
    "create_reference_trajectory",
]