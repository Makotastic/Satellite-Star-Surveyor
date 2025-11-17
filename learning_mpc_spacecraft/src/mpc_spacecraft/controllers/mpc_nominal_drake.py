"""Nominal MPC controller using Drake as optimization backend."""

import numpy as np
from typing import Optional, Callable
from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver, SnoptSolver


class NominalMPC:
    """
    Model Predictive Control (MPC) controller with manual formulation.
    
    Uses Drake's MathematicalProgram as the optimization backend, but all
    constraints, costs, and dynamics are manually implemented.
    """
    
    def __init__(
        self,
        dynamics_func: Callable,
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
        
        # Solver selection
        self.solver = OsqpSolver()
    
    def solve(
        self,
        x0: np.ndarray,
        x_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
        warm_start_x: Optional[np.ndarray] = None,
        warm_start_u: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Solve the MPC optimization problem.
        
        Args:
            x0: Initial state
            x_ref: Reference state trajectory (horizon+1 x state_dim)
            u_ref: Reference control trajectory (horizon x control_dim)
            warm_start_x: Warm start for state trajectory
            warm_start_u: Warm start for control trajectory
            
        Returns:
            u_opt: Optimal control sequence (horizon x control_dim)
            x_opt: Optimal state trajectory (horizon+1 x state_dim)
            success: Whether optimization succeeded
        """
        # Set default references
        if x_ref is None:
            x_ref = np.zeros((self.horizon + 1, self.state_dim))
        if u_ref is None:
            u_ref = np.zeros((self.horizon, self.control_dim))
        
        # Create optimization program
        prog = MathematicalProgram()
        
        # Decision variables
        x = prog.NewContinuousVariables(
            self.horizon + 1, self.state_dim, "x"
        )
        u = prog.NewContinuousVariables(
            self.horizon, self.control_dim, "u"
        )
        
        # Initial condition constraint
        prog.AddLinearEqualityConstraint(x[0], x0)
        
        # Dynamics constraints
        for k in range(self.horizon):
            x_next = self.dynamics_func(x[k], u[k])
            prog.AddConstraint(
                lambda vars, x_next=x_next: vars - x_next,
                lb=np.zeros(self.state_dim),
                ub=np.zeros(self.state_dim),
                vars=x[k + 1]
            )
        
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
        
        # State constraints (optional)
        if self.x_min is not None or self.x_max is not None:
            for k in range(self.horizon + 1):
                if self.x_min is not None:
                    prog.AddLinearConstraint(
                        x[k] >= self.x_min
                    )
                if self.x_max is not None:
                    prog.AddLinearConstraint(
                        x[k] <= self.x_max
                    )
        
        # Stage costs
        for k in range(self.horizon):
            x_error = x[k] - x_ref[k]
            u_error = u[k] - u_ref[k]
            
            # Quadratic cost: (x-x_ref)^T Q (x-x_ref) + (u-u_ref)^T R (u-u_ref)
            prog.AddQuadraticCost(
                self.Q,
                -2 * self.Q @ x_ref[k],
                x[k]
            )
            prog.AddQuadraticCost(
                self.R,
                -2 * self.R @ u_ref[k],
                u[k]
            )
        
        # Terminal cost
        x_error_terminal = x[self.horizon] - x_ref[self.horizon]
        prog.AddQuadraticCost(
            self.Q_terminal,
            -2 * self.Q_terminal @ x_ref[self.horizon],
            x[self.horizon]
        )
        
        # Warm start (optional)
        if warm_start_x is not None and warm_start_u is not None:
            initial_guess = np.concatenate([
                warm_start_x.flatten(),
                warm_start_u.flatten()
            ])
            prog.SetInitialGuess(
                np.concatenate([x.flatten(), u.flatten()]),
                initial_guess
            )
        
        # Solve
        result = Solve(prog)
        
        # Extract solution
        success = result.is_success()
        if success:
            x_opt = result.GetSolution(x)
            u_opt = result.GetSolution(u)
        else:
            # Return zeros if optimization failed
            x_opt = np.zeros((self.horizon + 1, self.state_dim))
            u_opt = np.zeros((self.horizon, self.control_dim))
        
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
        u_opt, _, success = self.solve(x0, x_ref, u_ref)
        
        if not success:
            print("Warning: MPC optimization failed, returning zero control")
            return np.zeros(self.control_dim)
        
        return u_opt[0]