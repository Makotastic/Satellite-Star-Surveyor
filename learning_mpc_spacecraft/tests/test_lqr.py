import pytest
import numpy as np
import quaternion as qu
from src.mpc_spacecraft.controllers.lqr import LQRController
from src.mpc_spacecraft.dynamics.rigid_body import SpacecraftDynamics


@pytest.fixture
def inertia():
    """Diagonal inertia matrix."""
    return np.diag([1.0, 1.0, 1.0])


@pytest.fixture
def dt():
    """Timestep."""
    return 0.1


@pytest.fixture
def dynamics(inertia, dt):
    """Dynamics instance."""
    return SpacecraftDynamics(inertia, dt)


@pytest.fixture
def ref_state():
    """Reference state."""
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def small_error_state():
    """Small 6D error state for LQR."""
    # delta_theta = [0.01, 0.02, 0.03], delta_omega = [0.01, 0.0, 0.0]
    return np.array([0.01, 0.02, 0.03, 0.01, 0.0, 0.0])


@pytest.fixture
def Q():
    """State cost matrix for 6D error state."""
    return np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])


@pytest.fixture
def R():
    """Control cost matrix."""
    return np.diag([0.1, 0.1, 0.1])


@pytest.fixture
def A_B(dynamics, ref_state):
    """Linearized A, B at reference."""
    zero_u = np.zeros(3)
    return dynamics.dynamics_error_jacobian(ref_state, zero_u)


@pytest.fixture
def lqr_discrete(A_B, Q, R, dynamics):
    """Discrete LQR controller."""
    A, B = A_B
    Ad, Bd = dynamics.discretize_linear_system(A, B)
    return LQRController(Ad, Bd, Q, R, discrete=True)


@pytest.fixture
def lqr_continuous(A_B, Q, R):
    """Continuous LQR controller."""
    A, B = A_B
    return LQRController(A, B, Q, R, discrete=False)


@pytest.fixture
def lqr_limited(A_B, Q, R, dynamics):
    """Discrete LQR controller with saturation limits."""
    A, B = A_B
    Ad, Bd = dynamics.discretize_linear_system(A, B)
    u_min = np.array([-0.5, -0.5, -0.5])
    u_max = np.array([0.5, 0.5, 0.5])
    return LQRController(
        Ad,
        Bd,
        Q,
        R,
        u_min=u_min,
        u_max=u_max,
        discrete=True,
    )


@pytest.mark.unit
def test_lqr_gain_computation_discrete(lqr_discrete, A_B, dynamics):
    A, B = A_B
    Ad, Bd = dynamics.discretize_linear_system(A, B)
    K = lqr_discrete.K

    assert K.shape == (3, 6)

    # Controller actually uses attitude states somewhere
    assert np.any(np.abs(K[:, :3]) > 1e-6)

    # Controller uses rates too
    assert np.any(np.abs(K[:, 3:]) > 1e-6)

    # Closed-loop system is stable (discrete-time)
    eigvals = np.linalg.eigvals(Ad - Bd @ K)
    assert np.all(np.abs(eigvals) < 1.0)


@pytest.mark.unit
def test_lqr_gain_computation_continuous(lqr_continuous, A_B):
    """Test continuous LQR gain computation."""
    A, B = A_B
    K = lqr_continuous.K
    assert K.shape == (3, 6)

    # Controller actually uses attitude states somewhere
    assert np.any(np.abs(K[:, :3]) > 1e-6)

    # Controller uses rates too
    assert np.any(np.abs(K[:, 3:]) > 1e-6)

    # Closed-loop system is stable (discrete-time)
    eigvals = np.linalg.eigvals(A - B @ K)
    assert np.all(np.real(eigvals) < 0.0)


@pytest.mark.unit
def test_compute_control_zero_error_gives_zero(lqr_discrete, small_error_state):
    """Zero error should produce (approximately) zero control."""
    zero_error = np.zeros_like(small_error_state)

    u = lqr_discrete.compute_control(zero_error)

    assert u.shape == (3,)
    np.testing.assert_allclose(u, np.zeros(3), atol=1e-9)


@pytest.mark.unit
def test_compute_control_linearity_and_symmetry(lqr_discrete, small_error_state):
    """
    LQR is linear state feedback: u(x) = -K x.
    Check scaling and sign symmetry:
      - u(a x) ≈ a u(x)
      - u(-x)  ≈ -u(x)
    """
    x = small_error_state

    u = lqr_discrete.compute_control(x)
    u_scaled = lqr_discrete.compute_control(2.0 * x)
    u_neg = lqr_discrete.compute_control(-x)

    # Shape check
    assert u.shape == (3,)

    # Scaling: u(2x) ≈ 2 u(x)
    np.testing.assert_allclose(u_scaled, 2.0 * u, rtol=1e-6, atol=1e-9)

    # Symmetry: u(-x) ≈ -u(x)
    np.testing.assert_allclose(u_neg, -u, rtol=1e-6, atol=1e-9)


@pytest.mark.unit
def test_saturation(lqr_discrete, lqr_limited, small_error_state):
    """Test control saturation using clipped limits on the controller."""
    u_min = lqr_limited.u_min
    u_max = lqr_limited.u_max

    # Scale state to ensure unsaturated control exceeds bounds.
    u_unsat = lqr_discrete.compute_control(small_error_state)
    max_abs = np.max(np.abs(u_unsat))
    assert max_abs > 0.0
    scale = (np.max(u_max) / max_abs) * 2.0
    saturated_state = small_error_state * scale

    u_unsat_scaled = lqr_discrete.compute_control(saturated_state)
    u_sat = lqr_limited.compute_control(saturated_state)

    # Saturated output should equal clipped unsaturated command.
    np.testing.assert_allclose(
        u_sat, np.clip(u_unsat_scaled, u_min, u_max), rtol=1e-6, atol=1e-9
    )
    assert np.any(u_unsat_scaled > u_max) or np.any(u_unsat_scaled < u_min)
    assert np.all(u_sat <= u_max)
    assert np.all(u_sat >= u_min)


@pytest.mark.unit
def test_compute_control_with_ref_matches_error_definition(
    lqr_discrete, small_error_state
):
    """
    For compute_control(state, ref), implementation should use error = state - ref.
    Verify that explicitly.
    """
    ref_state = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0])

    # What the controller returns when given (state, ref)
    u_with_ref = lqr_discrete.compute_control(small_error_state, ref_state)

    # What it *should* be if it uses error = state - ref internally
    effective_error = small_error_state - ref_state
    u_manual = lqr_discrete.compute_control(effective_error)

    assert u_with_ref.shape == (3,)
    np.testing.assert_allclose(u_with_ref, u_manual, rtol=1e-6, atol=1e-9)


@pytest.mark.unit
def test_closed_loop_eigenvalues(lqr_discrete):
    """Test closed-loop eigenvalues computation."""
    evals = lqr_discrete.get_closed_loop_eigenvalues()
    assert evals.shape == (6,)
    # Real parts should be negative or abs <1 for stability


@pytest.mark.unit
def test_is_stable_discrete(lqr_discrete):
    """Test stability check for discrete LQR."""
    assert lqr_discrete.is_stable()  # Should be stable with positive Q,R


@pytest.mark.unit
def test_is_stable_continuous(lqr_continuous):
    """Test stability check for continuous LQR."""
    assert lqr_continuous.is_stable()  # Should be stable


@pytest.mark.unit
def test_cost_to_go(lqr_discrete, small_error_state):
    """Test cost-to-go computation."""
    cost = lqr_discrete.compute_cost_to_go(small_error_state)
    assert isinstance(cost, float)
    assert cost > 0  # Positive definite S


@pytest.mark.integration
def test_closed_loop_convergence(
    dynamics, lqr_discrete, ref_state, Q, R, small_error_state
):
    """Integration test: closed-loop simulation with LQR, verify convergence."""

    # Start from full state corresponding to small error
    initial_state = dynamics.state_from_error(small_error_state, ref_state)
    current_state = initial_state
    states = [current_state]

    for _ in range(100):  # 10 seconds
        # Compute error state
        error_state = dynamics.state_error(current_state, ref_state)
        # Compute control on error
        u = lqr_discrete.compute_control(error_state)
        # Apply to full dynamics
        next_state = dynamics.discrete_dynamics_rk4(current_state, u)
        states.append(next_state)
        current_state = next_state

        # Early stop if converged
        final_error = dynamics.state_error(current_state, ref_state)
        if np.linalg.norm(final_error) < 1e-4:
            break

    # Verify convergence: final error small
    final_error = dynamics.state_error(states[-1], ref_state)
    assert np.linalg.norm(final_error) < 1e-3
    # Quaternions normalized
    q_final = qu.quaternion(*states[-1][:4])
    assert np.isclose(q_final.norm(), 1.0, atol=1e-10)


@pytest.mark.integration
def test_lqr_with_disturbance(
    dynamics, lqr_discrete, ref_state, Q, R, small_error_state
):
    """Test LQR robustness to small disturbance in closed-loop."""

    initial_state = dynamics.state_from_error(small_error_state, ref_state)
    current_state = initial_state
    disturbance = np.array([0.05, 0.0, 0.0])  # Small constant disturbance

    for _ in range(50):
        error_state = dynamics.state_error(current_state, ref_state)
        u = lqr_discrete.compute_control(error_state)
        # Apply with disturbance
        next_state = dynamics.discrete_dynamics_rk4(current_state, u, disturbance)
        current_state = next_state

    final_error = dynamics.state_error(current_state, ref_state)
    # With disturbance, error may not go to zero, but should be bounded small
    assert np.linalg.norm(final_error) < 0.05  # Tolerant of small disturbance
