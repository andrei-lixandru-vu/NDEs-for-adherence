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
from .data_utils import (
    map_values_to_classes,
    classes_to_values,
    load_reinforce_dataset
)
from .reinforce_dataset import (
    create_reinforce_dataset,
    get_reinforce_config
)
from .augmentation import (
    slice_trajectory,
    augment_dataset,
    compute_augmentation_stats,
    WindowedDataset,
    time_warp_trajectory,
    time_warp_trajectory_simple,
    time_warp_trajectory_smooth,
    generate_warping_curve,
    augment_dataset_time_warp,
    add_gaussian_noise,
    augment_dataset_gaussian_noise
)
from .augmented_datasets import (
    AugmentedAdherenceDataset,
    create_augmented_dataset
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
    
    # Data utilities
    'map_values_to_classes',
    'classes_to_values',
    'load_reinforce_dataset',
    'create_reinforce_dataset',
    'get_reinforce_config',
    
    # Augmentation
    'slice_trajectory',
    'augment_dataset',
    'compute_augmentation_stats',
    'WindowedDataset',
    'time_warp_trajectory',
    'time_warp_trajectory_simple',
    'time_warp_trajectory_smooth',
    'generate_warping_curve',
    'augment_dataset_time_warp',
    'add_gaussian_noise',
    'augment_dataset_gaussian_noise',
    'AugmentedAdherenceDataset',
    'create_augmented_dataset',
    
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
