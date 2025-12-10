# Codebase Summary: Learning-Augmented MPC for Spacecraft Attitude Control

This codebase focuses on spacecraft attitude control, primarily through Model Predictive Control (MPC), with a significant emphasis on integrating machine learning to enhance control performance by learning and compensating for unmodeled dynamics or disturbances.

## 1. `src/mpc_spacecraft/dynamics/rigid_body.py`
This file defines the core `SpacecraftDynamics` class, which models the rigid body motion of a spacecraft.
*   **State Representation:** Uses a 7D state vector `[quaternion (4D), angular_velocity (3D)]`.
*   **Control Input:** 3D torque vector.
*   **Dynamics:** Implements continuous-time dynamics (quaternion kinematics and Euler's equation) and discrete-time integration methods (RK4 and Euler).
*   **Error State:** Provides functions to map between full state and a 6D error state `[delta_theta, delta_omega]` (attitude error and angular velocity error), crucial for linearization.
*   **Linearization:** Can linearize the continuous-time error dynamics around a reference point to obtain `A` and `B` matrices, and discretize these linear systems.

## 2. `src/mpc_spacecraft/dynamics/disturbances.py`
This file introduces the `DisturbanceModel` class to simulate various external disturbances affecting the spacecraft.
*   **Disturbance Types:** Supports constant bias, random noise, and sinusoidal disturbances.
*   **Simplified Models:** Includes simplified models for atmospheric drag, gravity gradient torque (placeholder), and solar radiation pressure. These models are designed to introduce realistic perturbations for testing and data generation.

## 3. `src/mpc_spacecraft/controllers/lqr.py`
This file implements a foundational `LQRController` (Linear Quadratic Regulator).
*   **Functionality:** Computes optimal feedback gain `K` for linear systems to derive control inputs `u = -K(x - x_ref)`.
*   **Discrete/Continuous:** Supports both discrete-time and continuous-time LQR by solving the corresponding Algebraic Riccati Equation (DARE/CARE).
*   **Features:** Includes control saturation, closed-loop eigenvalue analysis for stability, and cost-to-go computation. It serves as a baseline linear controller.

## 4. `src/mpc_spacecraft/controllers/mpc_nominal_drake.py`
This file defines the `NominalMPC` class, a Model Predictive Controller using Drake's `MathematicalProgram` as the optimization backend.
*   **MPC Formulation:** Manually formulates constraints, costs, and dynamics within Drake.
*   **Trajectory Management:** Generates initial guesses and cost reference trajectories (for tracking or goal-only MPC). Uses quaternion slerp for attitude interpolation.
*   **Optimization:** Solves a Quadratic Program (QP) in error coordinates, potentially using Sequential Quadratic Programming (SQP) iterations for non-tracking problems.
*   **Warm-starting:** Reuses previous MPC solutions to warm-start the optimizer, improving efficiency.
*   **Linearization:** Relies on the `SpacecraftDynamics` class for linearizing the system around a nominal trajectory.

## 5. `src/mpc_spacecraft/learning/residual_model.py`
This file defines the `ResidualDynamicsModel` class, a neural network (MLP) designed to learn the residual dynamics.
*   **Purpose:** Models the difference between true system dynamics and a nominal model (`f_true - f_nominal`).
*   **Architecture:** Configurable MLP with hidden layers, activation functions (ReLU, Tanh, ELU), and dropout.
*   **Input/Output:** Takes concatenated `[state, control]` as input and outputs the state-dimension residual.
*   **Persistence:** Includes methods to save and load trained models.

## 6. `src/mpc_spacecraft/learning/dataset.py`
This file provides utilities for creating and managing datasets for training residual dynamics models.
*   **`DynamicsDataset` Class:** A PyTorch `Dataset` that stores `(state, control, next_state, residual)` tuples.
*   **Residual Computation:** Can compute residuals if a nominal dynamics function is provided.
*   **Normalization:** Implements input and output normalization based on training data statistics, applying these consistently across train, validation, and test sets.
*   **Data Loading:** The `create_dataset_from_logs()` function loads simulation logs (pickle or NPZ), splits them into train/validation/test sets, and prepares `DynamicsDataset` objects.

## 7. `src/mpc_spacecraft/learning/train_residual.py`
This file contains functions for training, evaluating, and cross-validating `ResidualDynamicsModel` instances.
*   **`train_residual_model()`:** Manages the training loop, including data loading, loss calculation (MSE), optimization (Adam), learning rate scheduling (ReduceLROnPlateau), and saving the best model.
*   **`evaluate_model()`:** Assesses model performance on a test set, reporting metrics like MSE, MAE, RMSE, and R-squared.
*   **`cross_validate()`:** Performs k-fold cross-validation to provide a robust estimate of model performance.

## 8. `src/mpc_spacecraft/controllers/mpc_learning_augmented.py`
This file integrates the learned residual dynamics into the MPC framework.
*   **`HybridSpacecraftDynamics` Class:** Extends `SpacecraftDynamics` to incorporate the `ResidualDynamicsModel`. The discrete dynamics become `x_{k+1} = f_nominal(x_k, u_k) + alpha * f_residual(x_k, u_k)`, where `alpha` is a trust factor.
*   **Hybrid Linearization:** Crucially, it linearizes the *hybrid discrete-time error dynamics* (nominal + residual) using finite differences, providing the `A_d` and `B_d` matrices for the MPC.
*   **`LearningAugmentedMPC` Class:** A subclass of `NominalMPC` that uses `HybridSpacecraftDynamics` as its underlying dynamics model. This means the MPC now optimizes based on the more accurate, learning-augmented dynamics, allowing it to adapt and perform better in the presence of unmodeled effects.

## Overall Architecture:

The project establishes a robust framework for spacecraft attitude control. It starts with fundamental rigid-body dynamics and disturbance modeling. It then builds a nominal MPC controller using Drake, capable of tracking trajectories or regulating to a goal state. The innovative aspect lies in the learning-augmented MPC, where a neural network learns the residual dynamics (the difference between the nominal model and reality). This learned residual is then integrated into the dynamics model used by the MPC, allowing the controller to adapt and perform better in the presence of unmodeled effects. The `dataset.py` and `train_residual.py` modules provide the necessary infrastructure for training and evaluating these residual models.