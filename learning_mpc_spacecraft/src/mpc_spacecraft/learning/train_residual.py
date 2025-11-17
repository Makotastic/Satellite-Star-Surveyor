"""Training utilities for residual dynamics models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm

from .residual_model import ResidualDynamicsModel
from .dataset import DynamicsDataset


def train_residual_model(
    model: ResidualDynamicsModel,
    train_dataset: DynamicsDataset,
    val_dataset: Optional[DynamicsDataset] = None,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = 'cpu',
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train residual dynamics model.
    
    Args:
        model: ResidualDynamicsModel to train
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: L2 regularization weight
        device: Device to train on ('cpu' or 'cuda')
        save_path: Path to save best model (optional)
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing training history
    """
    # Move model to device
    model = model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=verbose
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        if verbose:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        else:
            pbar = train_loader
        
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if verbose:
                pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        if val_dataset is not None:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if save_path is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(save_path)
                if verbose:
                    print(f'Saved best model with val_loss: {val_loss:.6f}')
        
        # Record learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        if verbose:
            if val_dataset is not None:
                print(f'Epoch {epoch+1}: train_loss={train_loss:.6f}, '
                      f'val_loss={val_loss:.6f}, lr={current_lr:.6f}')
            else:
                print(f'Epoch {epoch+1}: train_loss={train_loss:.6f}, '
                      f'lr={current_lr:.6f}')
    
    return history


def evaluate_model(
    model: ResidualDynamicsModel,
    test_dataset: DynamicsDataset,
    batch_size: int = 64,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        batch_size: Batch size
        device: Device to evaluate on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(test_loader)
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Mean absolute error
    mae = np.mean(np.abs(predictions - targets))
    
    # Root mean squared error
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # R-squared score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-10))
    
    metrics = {
        'test_loss': avg_loss,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2_score
    }
    
    return metrics


def cross_validate(
    model_class: type,
    dataset: DynamicsDataset,
    k_folds: int = 5,
    **train_kwargs
) -> List[Dict[str, float]]:
    """
    Perform k-fold cross-validation.
    
    Args:
        model_class: Model class to instantiate
        dataset: Full dataset
        k_folds: Number of folds
        **train_kwargs: Additional arguments for train_residual_model
        
    Returns:
        List of evaluation metrics for each fold
    """
    fold_size = len(dataset) // k_folds
    all_metrics = []
    
    for fold in range(k_folds):
        print(f'\n=== Fold {fold + 1}/{k_folds} ===')
        
        # Split data
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        
        val_indices = list(range(val_start, val_end))
        train_indices = list(range(0, val_start)) + list(range(val_end, len(dataset)))
        
        # Create subsets
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        
        # Create new model
        model = model_class()
        
        # Train
        train_residual_model(model, train_subset, val_subset, **train_kwargs)
        
        # Evaluate
        metrics = evaluate_model(model, val_subset)
        all_metrics.append(metrics)
        
        print(f'Fold {fold + 1} metrics: {metrics}')
    
    return all_metrics