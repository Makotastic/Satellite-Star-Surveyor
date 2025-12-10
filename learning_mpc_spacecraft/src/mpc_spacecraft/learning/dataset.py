"""Dataset utilities for training residual dynamics models."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import pickle


class DynamicsDataset(Dataset):
    """
    PyTorch Dataset for dynamics learning.
    
    Stores tuples of (state, control, next_state, residual).
    """
    
    def __init__(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
        residuals: Optional[np.ndarray] = None,
        normalize: bool = True
    ):
        """
        Initialize dynamics dataset.
        
        Args:
            states: State array [N, state_dim]
            controls: Control array [N, control_dim]
            next_states: Next state array [N, state_dim]
            residuals: Residual array [N, state_dim] (optional, will be computed if None)
            normalize: Whether to normalize inputs
        """
        self.states = torch.FloatTensor(states)
        self.controls = torch.FloatTensor(controls)
        self.next_states = torch.FloatTensor(next_states)
        
        if residuals is not None:
            self.residuals = torch.FloatTensor(residuals)
        else:
            # Compute residuals as difference (simplified)
            self.residuals = self.next_states - self.states
        
        self.normalize = normalize
        
        if self.normalize:
            self._compute_normalization_stats()
            self._normalize_data()
    
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization."""
        # Concatenate states and controls for input normalization
        inputs = torch.cat([self.states, self.controls], dim=1)
        
        self.input_mean = inputs.mean(dim=0)
        self.input_std = inputs.std(dim=0) + 1e-8  # Avoid division by zero
        
        self.output_mean = self.residuals.mean(dim=0)
        self.output_std = self.residuals.std(dim=0) + 1e-8
    
    def _normalize_data(self):
        """Normalize inputs and outputs."""
        # Normalize states
        self.states = (self.states - self.input_mean[:self.states.shape[1]]) / \
                      self.input_std[:self.states.shape[1]]
        
        # Normalize controls
        control_start = self.states.shape[1]
        self.controls = (self.controls - self.input_mean[control_start:]) / \
                        self.input_std[control_start:]
        
        # Normalize residuals
        self.residuals = (self.residuals - self.output_mean) / self.output_std
    
    def denormalize_output(self, normalized_output: torch.Tensor) -> torch.Tensor:
        """
        Denormalize model output.
        
        Args:
            normalized_output: Normalized residual predictions
            
        Returns:
            Denormalized residuals
        """
        if not self.normalize:
            return normalized_output
        
        return normalized_output * self.output_std + self.output_mean
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            (input, target) where input = [state, control] and target = residual
        """
        input_data = torch.cat([self.states[idx], self.controls[idx]])
        target = self.residuals[idx]
        
        return input_data, target
    
    def save(self, path: str):
        """Save dataset to file."""
        data = {
            'states': self.states,
            'controls': self.controls,
            'next_states': self.next_states,
            'residuals': self.residuals,
            'normalize': self.normalize,
        }
        
        if self.normalize:
            data['input_mean'] = self.input_mean
            data['input_std'] = self.input_std
            data['output_mean'] = self.output_mean
            data['output_std'] = self.output_std
        
        torch.save(data, path)
    
    @classmethod
    def load(cls, path: str) -> 'DynamicsDataset':
        """Load dataset from file."""
        data = torch.load(path)
        
        dataset = cls(
            states=data['states'].numpy(),
            controls=data['controls'].numpy(),
            next_states=data['next_states'].numpy(),
            residuals=data['residuals'].numpy(),
            normalize=False  # Data is already normalized
        )
        
        if data['normalize']:
            dataset.normalize = True
            dataset.input_mean = data['input_mean']
            dataset.input_std = data['input_std']
            dataset.output_mean = data['output_mean']
            dataset.output_std = data['output_std']
        
        return dataset


def create_dataset_from_logs(
    log_file: str,
    nominal_dynamics_func: Optional[callable] = None,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Tuple[DynamicsDataset, DynamicsDataset, DynamicsDataset]:
    """
    Create train/val/test datasets from simulation logs.
    
    Args:
        log_file: Path to simulation log file (pickle or npz)
        nominal_dynamics_func: Function to compute nominal dynamics (optional)
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Load log file
    if log_file.endswith('.pkl'):
        with open(log_file, 'rb') as f:
            logs = pickle.load(f)
    elif log_file.endswith('.npz'):
        logs = np.load(log_file)
    else:
        raise ValueError(f"Unsupported file format: {log_file}")
    
    # Extract data
    states = logs['states'][:-1]  # All but last
    controls = logs['controls']
    next_states = logs['states'][1:]  # All but first
    
    # Compute residuals if nominal dynamics provided
    if nominal_dynamics_func is not None:
        nominal_next_states = np.array([
            nominal_dynamics_func(states[i], controls[i])
            for i in range(len(states))
        ])
        residuals = next_states - nominal_next_states
    else:
        residuals = None
    
    # Split data
    n_samples = len(states)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Create datasets
    train_dataset = DynamicsDataset(
        states[train_idx],
        controls[train_idx],
        next_states[train_idx],
        residuals[train_idx] if residuals is not None else None
    )
    
    val_dataset = DynamicsDataset(
        states[val_idx],
        controls[val_idx],
        next_states[val_idx],
        residuals[val_idx] if residuals is not None else None,
        normalize=False  # Use training set statistics
    )
    
    test_dataset = DynamicsDataset(
        states[test_idx],
        controls[test_idx],
        next_states[test_idx],
        residuals[test_idx] if residuals is not None else None,
        normalize=False  # Use training set statistics
    )
    
    # Copy normalization statistics to val and test sets
    if train_dataset.normalize:
        val_dataset.normalize = True
        val_dataset.input_mean = train_dataset.input_mean
        val_dataset.input_std = train_dataset.input_std
        val_dataset.output_mean = train_dataset.output_mean
        val_dataset.output_std = train_dataset.output_std
        val_dataset._normalize_data()
        
        test_dataset.normalize = True
        test_dataset.input_mean = train_dataset.input_mean
        test_dataset.input_std = train_dataset.input_std
        test_dataset.output_mean = train_dataset.output_mean
        test_dataset.output_std = train_dataset.output_std
        test_dataset._normalize_data()
    
    return train_dataset, val_dataset, test_dataset