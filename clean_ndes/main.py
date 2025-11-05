"""
Main script for running adherence prediction experiments.

Compares intra-trajectory vs inter-trajectory sampling strategies.
"""

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, Subset

try:
    from . import config
    from .data_generation import generate_synthetic_data_super_simple as generate_synthetic_data_superSimple
    from .datasets import AdherenceDataset, SampledLength2AdherenceDataset
    from .model import AdherencePredictor
    from .training import train_multiple_runs
    from .visualization import plot_loss_comparison, print_loss_statistics
except ImportError:
    # If running as script, use absolute imports
    import config
    from data_generation import generate_synthetic_data_super_simple as generate_synthetic_data_superSimple
    from datasets import AdherenceDataset, SampledLength2AdherenceDataset
    from model import AdherencePredictor
    from training import train_multiple_runs
    from visualization import plot_loss_comparison, print_loss_statistics


def setup_data_loaders_intra(data, batch_size, use_specific_split=True):
    """
    Setup data loaders for intra-trajectory sampling.
    
    Parameters
    ----------
    data : np.ndarray
        Synthetic data array
    batch_size : int
        Batch size for training
    use_specific_split : bool
        If True, use fixed split (800/200). If False, use 80/20 random split.
        
    Returns
    -------
    tuple
        (training_loader, validation_loader)
    """
    dataset = SampledLength2AdherenceDataset(data)
    
    if use_specific_split:
        training_set = Subset(dataset, range(800))
        validation_set = Subset(dataset, range(800, 1000))
    else:
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        training_set, validation_set = random_split(dataset, [train_size, val_size])
    
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    
    print(f"Intra-trajectory sampling:")
    print(f"  Training set size: {len(training_set)}")
    print(f"  Validation set size: {len(validation_set)}")
    
    return training_loader, validation_loader


def setup_data_loaders_inter(data, batch_size):
    """
    Setup data loaders for inter-trajectory sampling.
    
    Parameters
    ----------
    data : np.ndarray
        Synthetic data array
    batch_size : int
        Batch size for training
        
    Returns
    -------
    tuple
        (training_loader, validation_loader)
    """
    dataset = AdherenceDataset(data)
    
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    training_set, validation_set = random_split(dataset, [train_size, val_size])
    
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    
    print(f"\nInter-trajectory sampling:")
    print(f"  Training set size: {len(training_set)}")
    print(f"  Validation set size: {len(validation_set)}")
    
    return training_loader, validation_loader


def main():
    """Main execution function."""
    # Set random seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    print("=" * 80)
    print("Adherence Prediction: Intra vs Inter-trajectory Sampling Comparison")
    print("=" * 80)
    
    # Generate synthetic data
    print(f"\nGenerating synthetic data:")
    print(f"  N trajectories: {config.N_TRAJECTORIES}")
    print(f"  T time steps: {config.T_TIME_STEPS}")
    print(f"  D dimensions: {config.D_DIMENSIONS}")
    
    data_synth_simple, adherence_dropped_at = generate_synthetic_data_superSimple(
        config.N_TRAJECTORIES, config.T_TIME_STEPS, config.D_DIMENSIONS
    )
    print(f"  Adherence drops: {len(adherence_dropped_at)}")
    
    # Intra-trajectory sampling
    print("\n" + "-" * 80)
    print("INTRA-TRAJECTORY SAMPLING")
    print("-" * 80)
    train_loader_intra, val_loader_intra = setup_data_loaders_intra(
        data_synth_simple, config.BATCH_SIZE, use_specific_split=True
    )
    
    runs_loss_history_intra = train_multiple_runs(
        num_runs=config.NUM_RUNS,
        model_class=AdherencePredictor,
        training_loader=train_loader_intra,
        validation_loader=val_loader_intra,
        in_dim=config.INPUT_DIM,
        out_dim=config.OUTPUT_DIM
    )
    
    # Inter-trajectory sampling
    print("\n" + "-" * 80)
    print("INTER-TRAJECTORY SAMPLING")
    print("-" * 80)
    train_loader_inter, val_loader_inter = setup_data_loaders_inter(
        data_synth_simple, config.BATCH_SIZE
    )
    
    runs_loss_history_intertraj = train_multiple_runs(
        num_runs=config.NUM_RUNS,
        model_class=AdherencePredictor,
        training_loader=train_loader_inter,
        validation_loader=val_loader_inter,
        in_dim=config.INPUT_DIM,
        out_dim=config.OUTPUT_DIM
    )
    
    # Visualization
    print("\n" + "=" * 80)
    print("RESULTS VISUALIZATION")
    print("=" * 80)
    
    print_loss_statistics(runs_loss_history_intra, runs_loss_history_intertraj)
    plot_loss_comparison(runs_loss_history_intra, runs_loss_history_intertraj)
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
