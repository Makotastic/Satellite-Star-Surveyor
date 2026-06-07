"""Neural network model for learning residual dynamics."""

import torch
import torch.nn as nn
from typing import List


class ResidualDynamicsModel(nn.Module):
    """
    Multi-layer perceptron (MLP) for learning residual dynamics.
    
    Input: [state, control] concatenated
    Output: residual dynamics (same dimension as state)
    
    The model learns: residual = f_true(x, u) - f_nominal(x, u)
    """
    
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_layers: List[int] = [64, 64, 32],
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Initialize residual dynamics model.
        
        Args:
            state_dim: Dimension of state vector
            control_dim: Dimension of control vector
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
            dropout: Dropout probability (0 = no dropout)
        """
        super(ResidualDynamicsModel, self).__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.input_dim = state_dim + control_dim
        self.output_dim = state_dim
        
        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch_size, state_dim + control_dim]
            
        Returns:
            Predicted residual [batch_size, state_dim]
        """
        return self.network(x)
    
    def predict(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        Predict residual for given state and control.
        
        Args:
            state: State tensor [batch_size, state_dim]
            control: Control tensor [batch_size, control_dim]
            
        Returns:
            Predicted residual [batch_size, state_dim]
        """
        x = torch.cat([state, control], dim=-1)
        return self.forward(x)
    
    def save(self, path: str):
        """
        Save model weights to file.
        
        Args:
            path: Path to save file
        """
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'control_dim': self.control_dim,
            'architecture': {
                'hidden_layers': [layer.out_features for layer in self.network 
                                 if isinstance(layer, nn.Linear)][:-1],
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'ResidualDynamicsModel':
        """
        Load model from file.
        
        Args:
            path: Path to model file
            device: Device to load model on ('cpu' or 'cuda')
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            state_dim=checkpoint['state_dim'],
            control_dim=checkpoint['control_dim'],
            hidden_layers=checkpoint['architecture']['hidden_layers']
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        return model
    
    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)