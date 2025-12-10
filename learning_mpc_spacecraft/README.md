# Learning-Augmented MPC for Spacecraft

This project implements a learning-augmented Model Predictive Control (MPC) system, a baseline LQR controller, and real-time visualization for a spacecraft/lander dynamic system.

## Quick Start

**See [`QUICKSTART.md`](QUICKSTART.md) for detailed setup instructions.**

### Using Docker Compose (Recommended)

```bash
cd learning_mpc_spacecraft
docker-compose up -d
docker-compose exec mpc_spacecraft bash
pip install -e .
```

### Using VSCode DevContainer

1. Open the project in VSCode
2. Press `F1` → "Dev Containers: Reopen in Container"
3. Wait for the container to build

## Overview

This project focuses on developing advanced control strategies for spacecraft, combining traditional control methods with machine learning techniques. The core idea is to use a learning module to augment the MPC controller, improving its performance and robustness. Real-time visualization will provide insights into the spacecraft's behavior.

## Technology Stack

- **Core Numerics:** `numpy`, `scipy`, custom rigid-body dynamics, manual discretization.
- **Optimization:** `pydrake.solvers.MathematicalProgram` with `OsqpSolver`, `SnoptSolver`, or `IpoptSolver`.
- **Learning Module:** `pytorch` for training residual dynamics models.
- **Visualization:** `meshcat` via `pydrake.visualization` (primary), `matplotlib.animation`, `pyvista`, `vispy` (optional).

## Codebase Layout

```
learning_mpc_spacecraft/
├─ README.md
├─ pyproject.toml / setup.cfg / requirements.txt
├─ src/
│  └─ mpc_spacecraft/
│     ├─ config/
│     ├─ dynamics/
│     ├─ controllers/
│     ├─ learning/
│     ├─ simulation/
│     ├─ visualization/
│     ├─ analysis/
│     └─ scripts/
├─ tests/
├─ notebooks/
├─ experiments/
└─ reports/
```

## Module Breakdown

### Dynamics
- `rigid_body.py`
- `quaternion.py`
- `disturbances.py`
Includes quaternion attitude dynamics, angular velocity dynamics, disturbance injection, discretization, and linearization utilities.

### Controllers
- **Base MPC:** `controllers/mpc_nominal_drake.py` (manual implementation using Drake backend solvers)
- **Learning-Augmented MPC:** `controllers/mpc_learning_augmented.py` (integrates PyTorch residual model)
- **LQR Controller:** `controllers/lqr.py` (computes continuous/discrete-time LQR gain)

### Learning Module
- `learning/residual_model.py`
- `learning/train_residual.py`
- `learning/dataset.py`
Handles dataset construction, MLP residual model training, model saving/loading, and prediction error evaluation.

### Simulation
- `simulate_closed_loop.py`
- `scenarios.py`
Provides a unified loop for LQR, MPC, and learning-MPC simulations, with logging capabilities for states, controls, and reference trajectories.

### Real-Time Visualization
- `visualization/meshcat_viz.py`
Features a Meshcat scene graph, lander mesh, moon/Mars sphere, and real-time pose updates for browser-based visualization.

### Analysis & Reporting
- `analysis/metrics.py`
- `analysis/plotting.py`
- `analysis/stability.py`
Tools for comparing controllers, computing closed-loop stability metrics, and generating plots for reports.

## Timeline

- **Week 1:** Baseline Dynamics + Manual MPC
- **Week 2:** Learning Module + LQR
- **Week 3:** Visualization + Analysis
- **Week 4:** Experiments + Final Report

## Deliverables

- Full manually implemented MPC controller
- Residual learning module
- Baseline LQR controller
- Real-time 3D visualization via Meshcat
- Comparison experiments
- Full reproducible codebase
- Final report