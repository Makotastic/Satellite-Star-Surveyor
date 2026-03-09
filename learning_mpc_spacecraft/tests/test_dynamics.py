import pytest
import numpy as np
import quaternion
from mpc_spacecraft.dynamics.rigid_body_rotation import SpacecraftDynamics


@pytest.fixture
def inertia():
    """Diagonal inertia matrix for testing."""
    return np.diag([1.0, 1.0, 1.0])


@pytest.fixture
def dt():
    """Discretization timestep."""
    return 0.1


@pytest.fixture
def dynamics(inertia, dt):
    """SpacecraftDynamics instance."""
    return SpacecraftDynamics(inertia, dt)


@pytest.fixture
def ref_state():
    """Reference state: identity quaternion, zero angular velocity."""
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def zero_control():
    """Zero control input."""
    return np.zeros(3)


@pytest.fixture
def small_error_state(ref_state):
    """Small perturbation around reference."""
    state = ref_state.copy()
    # Small rotation: delta_theta = [0.01, 0.02, 0.03]
    dq = np.quaternion(1.0, 0.005, 0.01, 0.015)  # 0.5 * delta_theta
    dq = dq.normalized()
    q_ref = np.quaternion(*ref_state[:4])
    q = q_ref * dq
    state[:4] = quaternion.as_float_array(q)
    state[4:] = np.array([0.01, 0.02, 0.0])  # small delta_omega
    return state


@pytest.mark.unit
def test_quaternion_normalization(dynamics, ref_state, zero_control):
    """Test quaternion normalization in dynamics."""
    # Unnormalized quaternion
    unnorm_q = np.array([1.5, 0.1, 0.2, 0.3])
    state_unnorm = np.concatenate([unnorm_q, ref_state[4:]])

    # Test with non-zero control
    non_zero_control = np.array([1.0, 0.0, 0.0])
    next_state_nonzero = dynamics.discrete_dynamics_rk4(state_unnorm, non_zero_control)
    q_next_nonzero = np.quaternion(*next_state_nonzero[:4])
    assert np.isclose(q_next_nonzero.norm(), 1.0, atol=1e-10)


@pytest.mark.unit
def test_continuous_dynamics_equilibrium(dynamics, ref_state, zero_control):
    """Test continuous dynamics at equilibrium (should be zero)."""
    state_dot = dynamics.continuous_dynamics(ref_state, zero_control)
    np.testing.assert_allclose(state_dot, np.zeros(7), atol=1e-10)


@pytest.mark.unit
def test_discrete_dynamics_consistency(dynamics, ref_state, zero_control):
    """Test RK4 vs Euler for small step at equilibrium."""
    rk4_next = dynamics.discrete_dynamics_rk4(ref_state, zero_control)
    euler_next = dynamics.discrete_dynamics_euler(ref_state, zero_control)
    np.testing.assert_allclose(rk4_next, euler_next, rtol=1e-6, atol=1e-8)


@pytest.mark.unit
def test_quaternion_error_small_angle(dynamics):
    """Test quaternion error for small angles."""
    q_ref = np.quaternion(1.0, 0.0, 0.0, 0.0)
    # Small rotation: delta_theta = [0.01, 0.0, 0.0]
    dq = np.quaternion(1.0, 0.005, 0.0, 0.0)
    q = q_ref * dq.normalized()
    delta_theta = dynamics.quaternion_error(q, q_ref)
    expected = np.array([0.01, 0.0, 0.0])
    np.testing.assert_allclose(delta_theta, expected, rtol=1e-4, atol=1e-8)
    # Check small angle: ||delta_theta|| < 0.1
    assert np.linalg.norm(delta_theta) < 0.1


@pytest.mark.unit
def test_state_error_and_from_error_roundtrip(dynamics, ref_state, small_error_state):
    """Test state_error and state_from_error roundtrip."""
    delta_x = dynamics.state_error(small_error_state, ref_state)
    reconstructed = dynamics.state_from_error(delta_x, ref_state)
    np.testing.assert_allclose(
        reconstructed[:4], small_error_state[:4], rtol=1e-3, atol=1e-8
    )
    np.testing.assert_allclose(
        reconstructed[4:], small_error_state[4:], rtol=1e-8, atol=1e-10
    )
    # Verify error coords linearity for small errors
    assert np.linalg.norm(delta_x[:3]) < 0.1  # small angle approx


@pytest.mark.unit
def test_linearize_around_equilibrium(dynamics, ref_state, zero_control):
    """Test linearization at equilibrium: A ≈ 0 for attitude (kinematics), -[I]^{-1} cross terms for omega; B = [I]^{-1}."""
    A, B = dynamics.linearize(ref_state, zero_control)
    # At equilibrium, A should have structure: top-left 0 (quaternion kinematics linearizes to 0 for small), etc.
    # For identity q, omega=0, A[:3,:3] ≈ 0, A[:3,3:] ≈ -[omega_skew] but omega=0 so 0, A[3:,:3] ≈ -[I]^{-1} [omega x I e_i] but simplified
    # Expect B ≈ inv(I) for omega part, 0 for quaternion
    expected_B = np.zeros((6, 3))
    expected_B[3:, :] = dynamics.inertia_inv  # torque affects omega_dot directly
    np.testing.assert_allclose(B, expected_B, atol=1e-10)
    # Expect A: delta_theta_dot = omega, omega_dot independent of state at this equilibrium
    expected_A = np.zeros((6, 6))
    expected_A[:3, 3:] = np.eye(3)
    np.testing.assert_allclose(A, expected_A, atol=1e-10, rtol=0)


@pytest.mark.unit
def test_discretize_linear_system(dynamics):
    """Test discretization: A=B=0 -> Ad=I, Bd=0; small A,B match ZOH formula."""
    A_cont = np.zeros((6, 6))
    B_cont = np.zeros((6, 3))
    Ad, Bd = dynamics.discretize_linear_system(A_cont, B_cont)

    np.testing.assert_allclose(Ad, np.eye(6), atol=1e-10)
    np.testing.assert_allclose(Bd, np.zeros((6, 3)), atol=1e-10)

    # Small A,B case with known analytic ZOH solution
    lam = 0.1  # eigenvalue
    A_small = lam * np.eye(6)
    B_small = 0.1 * np.ones((6, 3))

    Ad_exp, Bd_exp = dynamics.discretize_linear_system(A_small, B_small)

    dt = dynamics.dt
    Ad_expected = np.exp(lam * dt) * np.eye(6)
    Bd_expected = ((np.exp(lam * dt) - 1.0) / lam) * B_small

    np.testing.assert_allclose(Ad_exp, Ad_expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(Bd_exp, Bd_expected, rtol=1e-9, atol=1e-12)


@pytest.mark.integration
def test_simple_maneuver_free_motion(dynamics, ref_state, small_error_state, zero_control):
    """
    Integration test: free rigid-body motion from a small error with zero control.
    - Quaternion remains normalized.
    - Angular velocity remains approximately constant.
    - Attitude error grows roughly according to kinematics, not numerically exploding.
    """
    current_state = small_error_state
    states = [current_state]
    n_steps = 50
    dt = dynamics.dt
    T = n_steps * dt

    for _ in range(n_steps):
        next_state = dynamics.discrete_dynamics_rk4(current_state, zero_control)
        states.append(next_state)
        current_state = next_state

    # 1) Quaternion remains normalized
    for state in states:
        q = np.quaternion(*state[:4])
        assert np.isclose(q.norm(), 1.0, atol=1e-10)

    # 2) Angular velocity remains approximately constant
    initial_error = dynamics.state_error(small_error_state, ref_state)
    final_error = dynamics.state_error(states[-1], ref_state)

    # delta_omega = omega - omega_ref, and omega_ref = 0, so this is just omega
    omega_init = initial_error[3:]
    omega_final = final_error[3:]
    np.testing.assert_allclose(omega_final, omega_init, rtol=1e-3, atol=1e-6)

    # 3) Attitude error grows approximately according to small-angle kinematics
    # delta_theta(t) ≈ delta_theta(0) + omega * T
    theta_init = initial_error[:3]
    theta_expected = theta_init + omega_init * T
    theta_final = final_error[:3]

    # Allow some numerical integration error
    np.testing.assert_allclose(theta_final, theta_expected, rtol=5e-2, atol=1e-3)

    # Optionally: sanity check that it hasn't blown up to something crazy
    assert np.linalg.norm(theta_final) < 1.0


@pytest.mark.integration
def test_error_coords_in_simulation(dynamics, ref_state, small_error_state, zero_control):
    """Verify error coordinates evolve consistently with the linearized dynamics for a single step."""
    dt = dynamics.dt

    # Simulate one step
    next_state = dynamics.discrete_dynamics_euler(small_error_state, zero_control)

    delta_x_before = dynamics.state_error(small_error_state, ref_state)
    delta_x_after  = dynamics.state_error(next_state,       ref_state)

    # Get continuous-time linearization at the reference
    A, B = dynamics.linearize(ref_state, zero_control)

    # First-order prediction in error coordinates:
    # delta_x(t + dt) ≈ delta_x(t) + dt * A * delta_x(t)
    delta_x_pred = delta_x_before + dt * (A @ delta_x_before)

    # Compare actual error evolution to linear prediction
    np.testing.assert_allclose(delta_x_after, delta_x_pred,
                               rtol=5e-2, atol=1e-4)

    # Optional: sanity check that nothing exploded numerically
    assert np.linalg.norm(delta_x_after) < 1.0
