import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import random_split
from src.mpc_spacecraft.learning.dataset import DynamicsDataset
from src.mpc_spacecraft.learning.residual_model import ResidualDynamicsModel
from src.mpc_spacecraft.learning.train_residual import train_residual_model, evaluate_model

# Load dataset
data_path = 'experiments/datasets/residual_dataset_full.npz'
data = np.load(data_path)

states = data['states']
controls = data['controls']
next_states = data['next_states']
residuals = data['residuals']

print(f"Loaded dataset: {states.shape[0]} samples")
print(f"Shapes: states {states.shape}, controls {controls.shape}, residuals {residuals.shape}")

# Create full dataset
state_dim = states.shape[1]  # Should be 7 (4 quat + 3 omega)
control_dim = controls.shape[1]  # Should be 3
full_dataset = DynamicsDataset(
    states=states,
    controls=controls,
    next_states=next_states,
    residuals=residuals,
    normalize=True
)

# Split dataset: 80% train, 10% val, 10% test
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

print(f"Dataset splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Instantiate model
model = ResidualDynamicsModel(
    state_dim=state_dim,
    control_dim=control_dim,
    hidden_layers=[64, 64, 32]
)

print(f"Model created with {model.get_num_parameters()} parameters")

# Train model
save_path = 'experiments/models/residual_model_full.pth'
history = train_residual_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-5,
    device=device,
    save_path=save_path,
    verbose=True
)

print("Training completed. Best model saved to", save_path)

# Evaluate on test set
test_metrics = evaluate_model(
    model=model,
    test_dataset=test_dataset,
    batch_size=64,
    device=device
)

print("Test evaluation metrics:")
for key, value in test_metrics.items():
    print(f"  {key}: {value:.6f}")