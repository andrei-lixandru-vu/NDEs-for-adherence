"""
Configuration file containing constants and hyperparameters.
"""

# Data generation parameters
N_TRAJECTORIES = 1000  # number of trajectories (patients)
T_TIME_STEPS = 183     # number of time steps (days)
D_DIMENSIONS = 7       # number of dimensions (adherence, reminder factors)

# Model architecture parameters
INPUT_DIM = 7
OUTPUT_DIM = 2  # number of classes in adherence

# Training parameters
BATCH_SIZE = 20
LEARNING_RATE = 0.001
NUM_RUNS = 3
TRAIN_SPLIT = 0.8

# Random seed
SEED = 42
