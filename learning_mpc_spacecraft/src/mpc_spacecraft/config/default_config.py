"""Default configuration parameters for the MPC spacecraft project."""

import numpy as np


class SpacecraftConfig:
    """Configuration for spacecraft dynamics and physical parameters."""
    
    # Inertia matrix (kg*m^2) - example values for a small spacecraft
    INERTIA = np.diag([10.0, 12.0, 8.0])
    
    # Mass (kg)
    MASS = 100.0
    
    # Maximum torque (N*m)
    MAX_TORQUE = 1.0
    
    # Simulation timestep (s)
    DT = 0.1
    
    # Initial conditions
    INITIAL_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
    INITIAL_ANGULAR_VELOCITY = np.array([0.0, 0.0, 0.0])  # rad/s


class MPCConfig:
    """Configuration for MPC controller."""
    
    # Prediction horizon
    HORIZON = 20
    
    # State weight matrix (quaternion + angular velocity)
    Q = np.diag([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    
    # Control weight matrix
    R = np.diag([0.1, 0.1, 0.1])
    
    # Terminal cost weight
    Q_TERMINAL = 2.0 * np.diag([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    
    # Control constraints
    U_MIN = -1.0  # N*m
    U_MAX = 1.0   # N*m


class LQRConfig:
    """Configuration for LQR controller."""
    
    # State weight matrix
    Q = np.diag([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    
    # Control weight matrix
    R = np.diag([0.1, 0.1, 0.1])


class LearningConfig:
    """Configuration for learning module."""
    
    # Neural network architecture
    HIDDEN_LAYERS = [64, 64, 32]
    
    # Training parameters
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    
    # Dataset split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Model save path
    MODEL_SAVE_PATH = "models/residual_model.pth"


class SimulationConfig:
    """Configuration for simulation."""
    
    # Simulation duration (s)
    DURATION = 30.0
    
    # Timestep (s)
    DT = 0.1
    
    # Disturbance parameters
    ENABLE_DISTURBANCES = True
    DISTURBANCE_MAGNITUDE = 0.01  # N*m
    
    # Logging
    LOG_FREQUENCY = 1  # Log every N steps


class VisualizationConfig:
    """Configuration for visualization."""
    
    # Meshcat server
    MESHCAT_HOST = "localhost"
    MESHCAT_PORT = 7000
    
    # Visualization update rate
    VIZ_UPDATE_RATE = 10  # Hz
    
    # Camera settings
    CAMERA_DISTANCE = 5.0
    CAMERA_ELEVATION = 30.0