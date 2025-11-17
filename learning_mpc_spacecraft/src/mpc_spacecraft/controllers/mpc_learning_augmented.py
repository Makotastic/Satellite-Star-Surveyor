"""Learning-augmented MPC controller."""

import numpy as np
import torch
from typing import Optional, Callable
from .mpc_nominal_drake import NominalMPC


class LearningAugmentedMPC(NominalMPC):
    """
    MPC controller augmented with learned residual dynamics.
    
    Prediction model: x_{k+1} = f_nominal(x_k, u_k) + f_residual(x_k, u_k)
    
    Inherits from NominalMPC and modifies the dynamics function to include
    the learned residual model.
    """
    
    def __init__(
        self,
        nominal_dynamics_func: Callable,
        residual_model: torch.nn.Module,
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
        x_max: Optional[np.ndarray] = None,
        use_residual: bool = True
    ):
        """
        Initialize learning-augmented MPC controller.
        
        Args:
            nominal_dynamics_func: Nominal dynamics function f(x, u) -> x_next
            residual_model: PyTorch model for residual dynamics
            horizon: Prediction horizon N
            dt: Timestep
            state_dim: Dimension of state vector
            control_dim: Dimension of control vector
            Q: State cost matrix
            R: Control cost matrix
            Q_terminal: Terminal cost matrix (optional)
            u_min: Minimum control values
            u_max: Maximum control values
            x_min: Minimum state values (optional)
            x_max: Maximum state values (optional)
            use_residual: Whether to use residual model (for ablation studies)
        """
        self.nominal_dynamics_func = nominal_dynamics_func
        self.residual_model = residual_model
        self.use_residual = use_residual
        
        # Set residual model to evaluation mode
        if self.residual_model is not None:
            self.residual_model.eval()
        
        # Create augmented dynamics function
        def augmented_dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            """Dynamics with learned residual."""
            # Nominal dynamics
            x_next_nominal = self.nominal_dynamics_func(x, u)
            
            if not self.use_residual or self.residual_model is None:
                return x_next_nominal
            
            # Predict residual using learned model
            with torch.no_grad():
                # Prepare input for neural network
                xu = np.concatenate([x, u])
                xu_tensor = torch.FloatTensor(xu).unsqueeze(0)
                
                # Predict residual
                residual = self.residual_model(xu_tensor).squeeze(0).numpy()
            
            # Augmented dynamics
            x_next = x_next_nominal + residual
            
            # Normalize quaternion if needed (first 4 elements)
            if state_dim >= 4:
                q_norm = np.linalg.norm(x_next[:4])
                if q_norm > 1e-10:
                    x_next[:4] = x_next[:4] / q_norm
            
            return x_next
        
        # Initialize parent class with augmented dynamics
        super().__init__(
            dynamics_func=augmented_dynamics,
            horizon=horizon,
            dt=dt,
            state_dim=state_dim,
            control_dim=control_dim,
            Q=Q,
            R=R,
            Q_terminal=Q_terminal,
            u_min=u_min,
            u_max=u_max,
            x_min=x_min,
            x_max=x_max
        )
    
    def enable_residual(self):
        """Enable residual model in predictions."""
        self.use_residual = True
    
    def disable_residual(self):
        """Disable residual model (use only nominal dynamics)."""
        self.use_residual = False
    
    def update_residual_model(self, new_model: torch.nn.Module):
        """
        Update the residual model (e.g., after retraining).
        
        Args:
            new_model: New PyTorch residual model
        """
        self.residual_model = new_model
        self.residual_model.eval()
    
    def predict_residual(
        self,
        state: np.ndarray,
        control: np.ndarray
    ) -> np.ndarray:
        """
        Predict residual dynamics for a given state-control pair.
        
        Args:
            state: State vector
            control: Control vector
            
        Returns:
            Predicted residual
        """
        if self.residual_model is None:
            return np.zeros(self.state_dim)
        
        with torch.no_grad():
            xu = np.concatenate([state, control])
            xu_tensor = torch.FloatTensor(xu).unsqueeze(0)
            residual = self.residual_model(xu_tensor).squeeze(0).numpy()
        
        return residual
    
    def compare_predictions(
        self,
        x0: np.ndarray,
        u_sequence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compare nominal vs augmented predictions for a control sequence.
        
        Args:
            x0: Initial state
            u_sequence: Control sequence (horizon x control_dim)
            
        Returns:
            x_nominal: Trajectory using nominal dynamics
            x_augmented: Trajectory using augmented dynamics
        """
        horizon = u_sequence.shape[0]
        
        # Nominal trajectory
        x_nominal = np.zeros((horizon + 1, self.state_dim))
        x_nominal[0] = x0
        for k in range(horizon):
            x_nominal[k + 1] = self.nominal_dynamics_func(
                x_nominal[k], u_sequence[k]
            )
        
        # Augmented trajectory
        x_augmented = np.zeros((horizon + 1, self.state_dim))
        x_augmented[0] = x0
        for k in range(horizon):
            x_augmented[k + 1] = self.dynamics_func(
                x_augmented[k], u_sequence[k]
            )
        
        return x_nominal, x_augmented