# Experiments

This directory contains experiment configurations and results.

## Structure

```
experiments/
├── configs/          # Experiment configuration files
├── data/            # Generated datasets and logs
├── models/          # Trained model checkpoints
└── results/         # Experiment results and metrics
```

## Running Experiments

Experiments can be run using scripts in `src/mpc_spacecraft/scripts/` or through notebooks.

### Example Workflow

1. **Generate Training Data**
   - Run baseline controller simulations
   - Log state, control, and disturbance data
   - Save to `experiments/data/`

2. **Train Residual Model**
   - Load training data
   - Train neural network
   - Save model to `experiments/models/`

3. **Evaluate Controllers**
   - Compare LQR, nominal MPC, and learning-augmented MPC
   - Save results to `experiments/results/`

4. **Generate Report**
   - Analyze metrics
   - Create plots
   - Save to `reports/`