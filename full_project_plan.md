# Project Plan: Learning-Augmented MPC + LQR + Drake + Real-Time Visualization

## Overview
This document outlines the full plan for implementing a **learning-augmented Model Predictive Control (MPC)** system, a **baseline LQR controller**, and **real-time visualization** for a spacecraft/lander dynamic system.  
All MPC formulations will be implemented **manually**, i.e., **you will build the optimization program from scratch** (constraints, cost, dynamics), and only use **pydrake as a solver backend**, not as a high-level MPC builder.

---

# 1. Technology Stack

## Core Numerics
- `numpy`, `scipy`
- Custom rigid-body dynamics (quaternions, rotational kinematics, torques)
- Manual discretization (RK4 or midpoint)

## Optimization (Backend Only)
- `pydrake.solvers.MathematicalProgram`
- Solvers: `OsqpSolver`, `SnoptSolver`, or `IpoptSolver`

⚠️ **You will implement the MPC formulation yourself:**  
- Decision variables  
- Constraint matrices  
- Cost matrices  
- Dynamics constraints  
- Terminal cost (optional)  
- Reference handling

## Learning Module
- `pytorch`
- Train residual dynamics model  
- Integrate learned model inside MPC prediction

## Visualization
**Primary:**  
- `meshcat` via `pydrake.visualization`

**Optional:**  
- `matplotlib.animation`
- `pyvista` / `vispy`

---

# 2. Codebase Layout

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

---

# 3. Module Breakdown

## 3.1 Dynamics
- `rigid_body.py`
- `quaternion.py`
- `disturbances.py`

Includes:
- Quaternion attitude dynamics  
- Angular velocity dynamics  
- Disturbance injection  
- Discretization utilities  
- Linearization utilities (A, B matrices)

---

# 3.2 Controllers

### 3.2.1 Base MPC (manual implementation)
File: `controllers/mpc_nominal_drake.py`

You will manually implement:
- Decision variables: `x[0:N+1]`, `u[0:N]`
- Quadratic cost:
  - Tracking error  
  - Control effort  
  - Terminal cost (optional)
- Linear or nonlinear constraints:
  - Dynamics  
  - Input constraints  
  - State constraints (if any)
- Solve using Drake backend solvers

### 3.2.2 Learning-Augmented MPC
File: `controllers/mpc_learning_augmented.py`

- Load residual model (PyTorch)  
- Prediction dynamics:  
  ```
  x_{k+1} = f_model(x_k, u_k) + f_residual(x_k, u_k)
  ```
- Solve MPC with hybrid dynamics

### 3.2.3 LQR Controller
File: `controllers/lqr.py`

- Compute continuous/discrete-time LQR gain  
- Implement:
  ```
  u = -K (x - x_ref)
  ```

---

# 3.3 Learning Module
Files:
- `learning/residual_model.py`
- `learning/train_residual.py`
- `learning/dataset.py`

Responsibilities:
- Construct dataset from simulation logs  
- Train MLP residual model  
- Save/load model weights  
- Evaluate prediction error  

---

# 3.4 Simulation
Files:
- `simulate_closed_loop.py`
- `scenarios.py`

Features:
- Unified loop for LQR, MPC, and learning-MPC  
- Logging of:
  - states  
  - controls  
  - reference trajectory  
  - residual model predictions (optional)

---

# 3.5 Real-Time Visualization
File: `visualization/meshcat_viz.py`

Features:
- Meshcat scene graph  
- Lander mesh  
- Moon/Mars sphere  
- Update pose each simulation step  
- Browser-based visualization

---

# 3.6 Analysis & Reporting
Files:
- `analysis/metrics.py`
- `analysis/plotting.py`
- `analysis/stability.py`

Features:
- Compare controllers  
- Compute closed-loop stability metrics  
- Generate plots for report  

---

# 4. Updated Timeline

## Week 1 – Baseline Dynamics + Manual MPC
- Implement rigid-body attitude dynamics  
- Build custom MPC formulation using Drake solver  
- Run baseline MPC simulations  
- Basic logging + reference trajectory generator

## Week 2 – Learning Module + LQR
- Implement residual model and training pipeline  
- Add LQR controller  
- Add learning-augmented MPC  
- Generate training datasets  
- Compare LQR vs MPC  

## Week 3 – Visualization + Analysis
- Implement Meshcat real-time visualization  
- Integrate with simulation loop  
- Add metrics & plotting  
- Clean API for launching experiments  

## Week 4 – Experiments + Final Report
- Run multiple scenarios  
- Collect performance metrics  
- Create figures  
- Write final analysis  
- Produce demo visualizations (video or real-time playback)

---

# 5. Deliverables Summary

- **Full manually implemented MPC controller**
- **Residual learning module**
- **Baseline LQR controller**
- **Real-time 3D visualization via Meshcat**
- **Comparison experiments**
- **Full reproducible codebase**
- **Final report**
