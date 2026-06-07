# Upgrading the Learning-Augmented MPC Spacecraft Attitude Control Project

## Overview

This document outlines how to extend the existing **Learning-Augmented
Model Predictive Control (LMPC)** spacecraft attitude control project
into a **full autonomous spacecraft simulation and guidance system**.

The goal is to transform the project from a standalone controller
experiment into a **complete autonomy stack** that demonstrates
capabilities valued by robotics and aerospace employers such as SpaceX,
Anduril, Shield AI, and Tesla.

Key focus areas:

-   State estimation
-   Robust nonlinear control
-   Learning-augmented dynamics
-   Mission-level autonomy
-   Realistic spacecraft actuator and sensor modeling
-   Real-time constraints

------------------------------------------------------------------------

# Current Project Capabilities

The current project includes:

-   Rigid-body spacecraft attitude dynamics
-   Quaternion-based orientation representation
-   LQR baseline controller
-   Linearized MPC controller
-   Learning-Augmented MPC (LMPC)
-   Residual dynamics learning using an MLP
-   Simulation with disturbances and model mismatch

Core technical themes already demonstrated:

-   Nonlinear rigid body dynamics
-   MPC trajectory optimization
-   ML-augmented control
-   Simulation-based evaluation
-   Model mismatch robustness analysis

------------------------------------------------------------------------

# Upgrade Path 1 --- Full State Estimation Loop (MEKF)

## Objective

Integrate a **Multiplicative Extended Kalman Filter (MEKF)** into the
control loop so the controller operates on **estimated state rather than
ground truth**.

## Add Realistic Sensors

Simulate realistic spacecraft sensors:

-   Gyroscope (bias + noise)
-   Star tracker
-   Sun sensor
-   Magnetometer

Example measurement models:

Gyroscope: - Angular velocity measurement with bias drift

Star tracker: - Quaternion or vector observation of star directions

Sun sensor: - Direction vector toward sun

Magnetometer: - Earth magnetic field vector

## Implementation Steps

1.  Implement sensor measurement models
2.  Add sensor noise and bias drift
3.  Implement MEKF propagation and update
4.  Feed estimated state to MPC/LMPC

## Demonstration Output

Plots comparing:

-   True attitude
-   Estimated attitude
-   Control performance degradation due to estimation noise

## Why This Matters

Real spacecraft **never have perfect state information**.

This upgrade demonstrates:

-   Sensor fusion
-   Flight software realism
-   Closed-loop estimation and control

------------------------------------------------------------------------

# Upgrade Path 2 --- Autonomous Star-Pointing Mission Layer

## Objective

Add a **mission planning layer** that commands spacecraft attitude
targets for observing stars.

Instead of controlling to a fixed orientation, the spacecraft:

1.  Selects a target
2.  Slews to it
3.  Stabilizes within pointing tolerance
4.  Collects observation time
5.  Moves to the next target

## Components

### Star Catalog

Use a subset of a real star catalog (e.g., Hipparcos).

Store stars as inertial direction vectors.

### Orbit Simulation

Simulate a satellite orbit in LEO and propagate:

-   Position
-   Earth occlusion
-   Sun exclusion angles

### Mission Constraints

Examples:

-   Avoid pointing too close to the sun
-   Maximum slew rate
-   Momentum limits

### Objective

Maximize observation time over a list of targets.

## Deliverable

Visualization showing:

-   Satellite orbit
-   Current pointing vector
-   Target star direction
-   Attitude error bounds

## Why This Matters

Transforms the project from a **controller experiment** into a
**spacecraft autonomy system**.

------------------------------------------------------------------------

# Upgrade Path 3 --- Stabilize the Learning-Augmented MPC

## Current Issue

The LMPC becomes unstable under high model error due to **non-smooth
neural network residual predictions**.

This causes large derivatives when computing Jacobians.

## Improvements

### Replace Finite Differencing

Use **automatic differentiation** to compute Jacobians through the
neural network.

Benefits:

-   Reduced numerical noise
-   Faster computation
-   More accurate linearization

### Improve Neural Network Smoothness

Use smoother architectures:

-   Tanh activations instead of ReLU
-   Gradient regularization
-   Spectral normalization
-   Lipschitz constraints

### Physics-Informed Learning

Add regularization enforcing physical structure:

-   Symmetry constraints
-   Energy conservation priors

### Robust MPC

Add uncertainty handling:

-   Tube MPC
-   Constraint tightening
-   Residual uncertainty bounds

## Demonstration

Show that LMPC remains stable at **high model mismatch levels**.

## Why This Matters

Demonstrates deep understanding of **ML inside control loops** and
real-world robustness challenges.

------------------------------------------------------------------------

# Upgrade Path 4 --- Reaction Wheel and Momentum Management

## Objective

Replace direct torque inputs with **realistic spacecraft actuators**.

### Reaction Wheel Model

Include:

-   Wheel angular momentum
-   Motor torque limits
-   Wheel speed saturation

### Momentum Buildup

External disturbances cause momentum accumulation.

### Momentum Dumping

Use magnetorquers or thrusters to desaturate wheels.

## Implementation

Extend state vector:

-   Wheel angular velocities

Add constraints:

-   Wheel speed limits
-   Torque limits

## Deliverable

Plots showing:

-   Wheel momentum buildup
-   Desaturation events
-   Attitude stability

## Why This Matters

Demonstrates **actuator physics awareness** and real spacecraft
engineering.

------------------------------------------------------------------------

# Upgrade Path 5 --- Real-Time Control Constraints

## Objective

Demonstrate that the controller can operate under **embedded compute
constraints**.

## Implementation

Port controller to:

-   C++
-   Real-time capable libraries

Run benchmarks on:

-   Jetson
-   Embedded CPU
-   Real-time loop simulation

Measure:

-   MPC solve time
-   Memory usage
-   Control frequency

## Deliverable

Real-time performance charts.

## Why This Matters

Companies care about **systems that actually run on hardware**.

------------------------------------------------------------------------

# Upgrade Path 6 --- Fault Detection and Recovery

## Objective

Simulate spacecraft failures and implement **fault detection and
isolation (FDI)**.

## Failure Cases

Examples:

-   Reaction wheel failure
-   Sensor bias drift
-   Stuck actuator

## Detection Methods

-   Residual monitoring
-   Innovation statistics in Kalman filter
-   Control error thresholds

## Response

-   Disable failed actuator
-   Reconfigure controller
-   Safe mode pointing

## Deliverable

Simulation showing spacecraft recovering from failures.

## Why This Matters

Robust autonomy systems must tolerate hardware failures.

------------------------------------------------------------------------

# Suggested Final Project Architecture

Recommended system structure:

Mission Planner → Target Selection → Slew Command

State Estimation → MEKF

Control → MPC / LMPC

Actuation → Reaction Wheels

Simulation → Orbit propagation → Sensor models → Disturbances

------------------------------------------------------------------------

# Final Project Title

**Robust Autonomous Spacecraft Attitude Control and Targeting with
Learning-Augmented Model Predictive Control**

------------------------------------------------------------------------

# Suggested Demonstration

A final demo should show:

-   Satellite orbiting Earth
-   Autonomous star selection
-   Slewing between targets
-   Attitude estimation via MEKF
-   Robust LMPC control under model error
-   Reaction wheel momentum management
-   Fault recovery scenarios

------------------------------------------------------------------------

# Skills Demonstrated

This extended project showcases:

-   Nonlinear spacecraft dynamics
-   Model predictive control
-   Machine learning in control systems
-   Sensor fusion and estimation
-   Autonomous mission planning
-   Real-time robotics systems
-   Robust autonomy engineering
