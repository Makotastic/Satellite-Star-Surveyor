# High-Performance Linearization Strategies for Attitude NMPC

This document expands on two advanced approaches for improving
performance and correctness in nonlinear MPC for spacecraft attitude
dynamics:

1.  Lie-group error-state formulation (SO(3) geometry)
2.  Analytic Jacobian of the discrete error map (chain-rule exact
    linearization)

------------------------------------------------------------------------

# 1. Lie-Group Error-State Formulation (SO(3))

## Core Idea

Instead of representing attitude errors using quaternion subtraction or
small-angle approximations directly, represent them using Lie algebra
coordinates:

R = R_ref \* exp(φ\^)

Where: - R ∈ SO(3) - φ ∈ ℝ³ is the rotation vector - φ\^ is the
skew-symmetric matrix - exp(φ\^) is Rodrigues' formula

The error state becomes:

e = \[ φ, δω \]

This keeps the error in the tangent space of SO(3), which naturally
handles moving reference frames.

------------------------------------------------------------------------

## Why This Helps

The key advantage:

-   Error propagation automatically accounts for moving tangent frames.
-   No ad-hoc bias corrections are required.
-   The geometry is exact.
-   Extremely stable numerically.

The attitude error propagation becomes:

φ\_{k+1} ≈ J_r(ω_nom dt) φ_k + dt δω_k

Where J_r is the right Jacobian of SO(3):

J_r(θ) = I - ((1 - cos\|\|θ\|\|)/\|\|θ\|\|²) θ\^ + ((\|\|θ\|\| -
sin\|\|θ\|\|)/\|\|θ\|\|³) θ\^²

For small angles:

J_r(θ) ≈ I - 1/2 θ\^

This properly accounts for the change in reference frame between time
steps.

------------------------------------------------------------------------

## Benefits

-   Fast (closed-form expressions)
-   No finite-difference noise
-   Handles moving references correctly
-   Used in flight software and high-end INS systems

## Downsides

-   Requires implementing SO(3) exp/log maps
-   Requires rewriting quaternion error handling

------------------------------------------------------------------------

# 2. Analytic Jacobian of the Discrete Error Map

## Core Idea

Instead of finite differencing the entire discrete error map, compute
Jacobians analytically using the chain rule.

You already have the exact nonlinear map:

dx\_{k+1} = e( f_d( from_error(dx_k, x_nom_k), u_nom_k + du_k ),
x_nom\_{k+1} )

Linearize this using chain rule.

------------------------------------------------------------------------

## Chain-Rule Structure

Let:

-   x_k = from_error(dx_k, x_nom_k)
-   x\_{k+1} = f_d(x_k, u_k)
-   dx\_{k+1} = e(x\_{k+1}, x_nom\_{k+1})

Then:

A_k = (∂e/∂x)\|*{k+1} · (∂f_d/∂x)\|*{k} · (∂x/∂e)\|\_{k}

B_k = (∂e/∂x)\|*{k+1} · (∂f_d/∂u)\|*{k}

This breaks the problem into three manageable Jacobians.

------------------------------------------------------------------------

## Required Components

1.  Dynamics Jacobians: F_x = ∂f_d/∂x F_u = ∂f_d/∂u

2.  Error map Jacobian: E_x = ∂e/∂x

3.  Reconstruction Jacobian: X_e = ∂x/∂e

Each has known closed-form expressions for quaternion error
representations.

------------------------------------------------------------------------

## Why This Is Powerful

-   No finite-difference noise
-   Exact discrete-time linearization
-   Works with moving references
-   High performance once implemented

------------------------------------------------------------------------

# Comparison

  Method                Speed     Accuracy   Effort
  --------------------- --------- ---------- --------
  Finite Difference     Slow      Noisy      Easy
  AutoDiff              Medium    Exact      Medium
  Lie-Group Analytic    Fastest   Exact      Hard
  Chain-Rule Analytic   Fastest   Exact      Hard

------------------------------------------------------------------------

# Recommended Path

Short term: - Implement analytic chain-rule Jacobians of the discrete
error map.

Long term: - Transition to Lie-group error-state formulation for maximum
robustness and geometric correctness.
