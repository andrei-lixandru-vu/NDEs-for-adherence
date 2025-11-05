"""
Clean NDEs package for adherence prediction experiments.
"""

from .config import *
from .data_generation import (
    generate_synthetic_reinforce_data,
    generate_synthetic_data_super_simple
)
from .model import AdherencePredictor
from .datasets import (
    AdherenceDataset,
    SampledLength2AdherenceDataset
)
from .losses import (
    get_batch_loss,
    get_batch_loss_validation
)
from .training import (
    train_one_epoch,
    train_multiple_runs
)
from .visualization import (
    plot_loss_comparison,
    print_loss_statistics
)

__all__ = [
    # Config
    'N_TRAJECTORIES',
    'T_TIME_STEPS',
    'D_DIMENSIONS',
    'INPUT_DIM',
    'OUTPUT_DIM',
    'BATCH_SIZE',
    'LEARNING_RATE',
    'NUM_RUNS',
    'TRAIN_SPLIT',
    'SEED',
    
    # Data generation
    'generate_synthetic_reinforce_data',
    'generate_synthetic_data_super_simple',
    
    # Model
    'AdherencePredictor',
    
    # Datasets
    'AdherenceDataset',
    'SampledLength2AdherenceDataset',
    
    # Losses
    'get_batch_loss',
    'get_batch_loss_validation',
    
    # Training
    'train_one_epoch',
    'train_multiple_runs',
    
    # Visualization
    'plot_loss_comparison',
    'print_loss_statistics',
]
