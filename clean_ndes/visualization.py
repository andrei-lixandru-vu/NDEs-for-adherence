"""
Visualization utilities for training results.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_comparison(runs_loss_history_intra, runs_loss_history_intertraj):
    """
    Plot comparison of training and validation losses for two different sampling strategies.
    
    Parameters
    ----------
    runs_loss_history_intra : np.ndarray
        Shape (runs, n_batches, 2) - loss history for intra-trajectory sampling
    runs_loss_history_intertraj : np.ndarray
        Shape (runs, n_batches, 2) - loss history for inter-trajectory sampling
    """
    # Extract train and val losses for each sampling strategy
    train_losses_intra = runs_loss_history_intra[:, :, 0]
    val_losses_intra = runs_loss_history_intra[:, :, 1]
    
    train_losses_intertraj = runs_loss_history_intertraj[:, :, 0]
    val_losses_intertraj = runs_loss_history_intertraj[:, :, 1]
    
    # Compute statistics
    mean_train_intra = np.mean(train_losses_intra, axis=0)
    min_train_intra = np.min(train_losses_intra, axis=0)
    max_train_intra = np.max(train_losses_intra, axis=0)
    
    mean_val_intra = np.mean(val_losses_intra, axis=0)
    min_val_intra = np.min(val_losses_intra, axis=0)
    max_val_intra = np.max(val_losses_intra, axis=0)
    
    mean_train_intertraj = np.mean(train_losses_intertraj, axis=0)
    min_train_intertraj = np.min(train_losses_intertraj, axis=0)
    max_train_intertraj = np.max(train_losses_intertraj, axis=0)
    
    mean_val_intertraj = np.mean(val_losses_intertraj, axis=0)
    min_val_intertraj = np.min(val_losses_intertraj, axis=0)
    max_val_intertraj = np.max(val_losses_intertraj, axis=0)
    
    batches = np.arange(mean_train_intra.shape[0])
    
    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(batches, mean_train_intra, label='short trajectory', color='tab:blue')
    plt.fill_between(batches, min_train_intra, max_train_intra, 
                     color='tab:blue', alpha=0.2)
    
    plt.plot(batches, mean_train_intertraj, label='long trajectory', color='tab:green')
    plt.fill_between(batches, min_train_intertraj, max_train_intertraj, 
                     color='tab:green', alpha=0.2)
    
    plt.title("Training Loss (mean ± range across runs)")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    # Plot validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(batches, mean_val_intra, label='short trajectory', color='tab:orange')
    plt.fill_between(batches, min_val_intra, max_val_intra, 
                     color='tab:orange', alpha=0.2)
    
    plt.plot(batches, mean_val_intertraj, label='long trajectory', color='tab:pink')
    plt.fill_between(batches, min_val_intertraj, max_val_intertraj, 
                     color='tab:pink', alpha=0.2)
    
    plt.title("Validation Loss (mean ± range across runs)")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def print_loss_statistics(runs_loss_history_intra, runs_loss_history_intertraj):
    """
    Print statistics about the loss histories.
    
    Parameters
    ----------
    runs_loss_history_intra : np.ndarray
        Shape (runs, n_batches, 2) - loss history for intra-trajectory sampling
    runs_loss_history_intertraj : np.ndarray
        Shape (runs, n_batches, 2) - loss history for inter-trajectory sampling
    """
    train_losses_intra = runs_loss_history_intra[:, :, 0]
    val_losses_intra = runs_loss_history_intra[:, :, 1]
    train_losses_intertraj = runs_loss_history_intertraj[:, :, 0]
    val_losses_intertraj = runs_loss_history_intertraj[:, :, 1]
    
    print("Intra-trajectory sampling:")
    print(f"  Train losses shape: {train_losses_intra.shape}")
    print(f"  Val losses shape: {val_losses_intra.shape}")
    
    print("\nInter-trajectory sampling:")
    print(f"  Train losses shape: {train_losses_intertraj.shape}")
    print(f"  Val losses shape: {val_losses_intertraj.shape}")
