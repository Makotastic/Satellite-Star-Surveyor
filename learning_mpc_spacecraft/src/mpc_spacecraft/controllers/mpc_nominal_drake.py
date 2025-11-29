"""Nominal MPC controller using Drake as optimization backend."""

import numpy as np
import quaternion
from typing import Optional, Callable
from pydrake.all import MathematicalProgram, OsqpSolver, Solve  # pylint: disable=no-name-in-module


class NominalMPC:
    """
    Model Predictive Control (MPC) controller with manual formulation.
    
    Uses Drake's MathematicalProgram as the optimization backend, but all
    constraints, costs, and dynamics are manually implemented.
    """ 
    
    def __init__(
        self,
        dynamics_func: Callable,
        linearizer_func: Callable,
        horizon: int,
        dt: float,
        state_dim: int,
        control_dim: int,
        Q: np.ndarray,
        R: np.ndarray,
        Q_terminal: Optional[np.ndarray] = None,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
        x_min: Optional[np.ndarray] = None,
        x_max: Optional[np.ndarray] = None
    ):
        """
        Initialize MPC controller.
        
        Args:
            dynamics_func: Function f(x, u) -> x_next for discrete dynamics
            horizon: Prediction horizon N
            dt: Timestep
            state_dim: Dimension of state vector
            control_dim: Dimension of control vector
            Q: State cost matrix (state_dim x state_dim)
            R: Control cost matrix (control_dim x control_dim)
            Q_terminal: Terminal cost matrix (optional)
            u_min: Minimum control values
            u_max: Maximum control values
            x_min: Minimum state values (optional)
            x_max: Maximum state values (optional)
        """
        self.dynamics_func = dynamics_func
        self.linearizer_func = linearizer_func
        self.horizon = horizon
        self.dt = dt
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal if Q_terminal is not None else Q
        self.u_min = u_min
        self.u_max = u_max
        self.x_min = x_min
        self.x_max = x_max
        self.warm_start_x = None
        self.warm_start_u = None
        
        # Solver selection
        self.solver = OsqpSolver()

    def find_initial_guesses(self,
        x0: np.ndarray,
        x_goal: Optional[np.ndarray] = None,
        x_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None
        ) -> tuple[np.ndarray, np.ndarray]:

        assert x_goal or x_ref, "Need to provide either trajectory or terminal goal"

        intial_x = []
        intial_u = []

        if self.warm_start_u is not None and self.warm_start_x is not None:
            x_terminal = x_goal if x_goal else x_ref[-1]
            intial_u = np.concat([self.warm_start_u[1:], np.zeros((1, self.control_dim))])
            intial_x = np.concat([self.warm_start_x[1:], x_terminal])

        elif x_ref is not None:
            intial_x = x_ref
            intial_u = u_ref if u_ref is not None else np.zeros((self.horizon, self.control_dim))

        elif x_goal is not None:
            arr_interp = np.linspace(0, 1, self.horizon + 1)

            x_q = np.quaternion(*x0[:4]).normalized()
            x_q_goal = np.quaternion(*x_goal[:4]).normalized()
            x_q_guess = np.quaternion.slerp(x_q, x_q_goal, 0, 1, arr_interp)
            x_q_guess = np.quaternion.as_float_array(x_q_guess).squeeze()

            x_omega = x0[4:]
            x_omega_goal = x_goal[4:]
            x_omega_guess = (1 - arr_interp[:,None]) * x_omega[:,None].T + arr_interp[:,None] * x_omega_goal[:,None].T

            intial_x = np.concatenate([x_q_guess, x_omega_guess], axis=1)
            intial_u = np.zeros((self.horizon, self.control_dim))

        return intial_x, intial_u
    
    def build_ref_trajectory(self,
        x0: np.ndarray,
        x_goal: Optional[np.ndarray] = None,
        x_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None
        ) -> tuple[np.ndarray, np.ndarray]:

        assert x_goal or x_ref, "Need to provide either trajectory or terminal goal"

        x_ref_traj = []
        u_ref_traj = []

        if x_ref is not None:
            x_ref_traj = x_ref
            u_ref_traj = u_ref if u_ref is not None else np.zeros((self.horizon, self.control_dim))

        elif x_goal is not None:
            arr_interp = np.linspace(0, 1, self.horizon + 1)

            x_q = np.quaternion(*x0[:4]).normalized()
            x_q_goal = np.quaternion(*x_goal[:4]).normalized()
            x_q_guess = quaternion.slerp(x_q, x_q_goal, 0, 1, arr_interp)
            x_q_guess = np.quaternion.as_float_array(x_q_guess).squeeze()

            x_omega = x0[4:]
            x_omega_goal = x_goal[4:]
            x_omega_guess = (1 - arr_interp[:,None]) * x_omega[:,None].T + arr_interp[:,None] * x_omega_goal[:,None].T

            x_ref_traj = np.concatenate([x_q_guess, x_omega_guess], axis=1)
            u_ref_traj = np.zeros((self.horizon, self.control_dim))

        return x_ref_traj, u_ref_traj
    
    def solve(
        self,
        x0: np.ndarray,
        x_goal: Optional[np.ndarray] = None,
        x_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Solve the MPC optimization problem.
        
        Args:
            x0: Initial state
            x_ref: Reference state trajectory (horizon+1 x state_dim)
            u_ref: Reference control trajectory (horizon x control_dim)
            
        Returns:
            u_opt: Optimal control sequence (horizon x control_dim)
            x_opt: Optimal state trajectory (horizon+1 x state_dim)
            success: Whether optimization succeeded
        """

        assert x_goal or x_ref, "Need to provide either trajectory or terminal goal"

        prog = MathematicalProgram()

        x = prog.NewContinuousVariables(self.horizon + 1, self.state_dim, "x")
        u = prog.NewContinuousVariables(self.horizon, self.control_dim, "u")

        intial_x, intial_u = self.find_initial_guesses(x0, x_goal, x_ref, u_ref)

        x_ref_traj, u_ref_traj = self.build_ref_trajectory(x0, x_goal, x_ref, u_ref)

        # Initial condition constraint
        prog.AddLinearEqualityConstraint(x[0], x0)

        # Dynamic Contraints
        for k in range(self.horizon):
            # x_kp1 = g(x_k,u_k) = g(x_d_k,x_d_k) + A @ (x_k - x_d_k) + B @ (u_k - u_d_k)
            # x_kp1 = A @ x_k + B @ u_k + g(x_d_k,x_d_k) - A @ x_d_k - B @ u_d_k
            # x_kp1 - A @ x_k + B @ u_k = g(x_d_k,x_d_k) - A @ x_d_k - B @ u_d_k

            A_k, B_k = self.linearizer_func(intial_x[k], intial_u[k])
            right = self.dynamics_func(intial_x[k], intial_u[k]) - A_k @ intial_x[k] - B_k @ intial_u[k]
            left = x[k+1] - A_k @ x[k] - B_k @ u[k]
            prog.AddLinearEqualityConstraint(left, right)

        # Control constraints
        if self.u_min is not None or self.u_max is not None:
            for k in range(self.horizon):
                if self.u_min is not None:
                    prog.AddLinearConstraint(
                        u[k] >= self.u_min
                    )
                if self.u_max is not None:
                    prog.AddLinearConstraint(
                        u[k] <= self.u_max
                    )

        # Stage Costs
        for k in range(self.horizon):
            prog.AddQuadraticErrorCost(self.Q, x_ref_traj[k], x[k])
            prog.AddQuadraticErrorCost(self.R, u_ref_traj[k], u[k])
        
        # Terminal Cost
        prog.AddQuadraticErrorCost(self.Q_terminal, x_ref_traj[self.horizon], x[self.horizon])
        
        # Intial Guess
        prog.SetInitialGuess(x, intial_x)
        prog.SetInitialGuess(u, intial_u)

        result = self.solver.Solve(prog)

        if success := result.is_success():
            x_opt = result.GetSolution(x)
            u_opt = result.GetSolution(u)

            self.warm_start_x = x_opt
            self.warm_start_u = u_opt

        else:
            # Return zeros if optimization failed
            x_opt = np.zeros((self.horizon + 1, self.state_dim))
            u_opt = np.zeros((self.horizon, self.control_dim))

            self.warm_start_x = None
            self.warm_start_u = None
        
        return u_opt, x_opt, success
    
    def get_first_control(
        self,
        x0: np.ndarray,
        x_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Solve MPC and return only the first control input (receding horizon).
        
        Args:
            x0: Initial state
            x_ref: Reference state trajectory
            u_ref: Reference control trajectory
            
        Returns:
            First control input u[0]
        """
        u_opt, _, success = self.solve(x0, x_ref=x_ref, u_ref=u_ref)
        
        if not success:
            print("Warning: MPC optimization failed, returning zero control")
            return np.zeros(self.control_dim)
        
        return u_opt[0]