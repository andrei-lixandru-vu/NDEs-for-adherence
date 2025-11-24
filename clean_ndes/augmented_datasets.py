"""
Augmented dataset classes that use window sampling for data augmentation.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from .augmentation import augment_dataset, compute_augmentation_stats
from .datasets import AdherenceDataset


class AugmentedAdherenceDataset(Dataset):
    """
    AdherenceDataset with trajectory windowing augmentation.
    
    This dataset applies window slicing to create multiple training samples
    from each trajectory, effectively augmenting the dataset size.
    """
    
    def __init__(self, data, window_size, stride=1, target_as_classes=True, 
                 augment_before_creation=True):
        """
        Parameters
        ----------
        data : np.ndarray
            Shape (N, T, D) where:
            - data[..., 0] is adherence (can be class indices or continuous values)
            - data[..., 1:] are control variables
        window_size : int
            Size of each window to extract from trajectories
        stride : int, optional
            Stride between windows. Default: 1 (overlapping windows)
            If stride == window_size, windows are non-overlapping
        target_as_classes : bool
            If True, treats target as class indices (integers).
            If False, treats target as continuous values (floats).
            Default: True
        augment_before_creation : bool
            If True, augment the dataset first then create AdherenceDataset.
            If False, use WindowedDataset for on-the-fly augmentation (memory efficient).
            Default: True
        """
        self.original_data = data
        self.window_size = window_size
        self.stride = stride
        self.target_as_classes = target_as_classes
        self.augment_before_creation = augment_before_creation
        
        if augment_before_creation:
            # Augment dataset upfront (uses more memory but faster access)
            print(f"Augmenting dataset with window_size={window_size}, stride={stride}...")
            stats = compute_augmentation_stats(data, window_size, stride)
            print(f"  Original trajectories: {stats['original_trajectories']}")
            print(f"  Windows per trajectory: {stats['windows_per_trajectory']}")
            print(f"  Total windows: {stats['total_windows']}")
            print(f"  Augmentation factor: {stats['augmentation_factor']:.2f}x")
            
            augmented_data = augment_dataset(data, window_size, stride)
            self.dataset = AdherenceDataset(augmented_data, target_as_classes=target_as_classes)
        else:
            # Use windowed dataset for on-the-fly augmentation (memory efficient)
            from .augmentation import WindowedDataset
            self.windowed_dataset = WindowedDataset(data, window_size, stride)
            self._create_mappings()
    
    def _create_mappings(self):
        """Create mappings for on-the-fly augmentation."""
        # This is called only if augment_before_creation=False
        self.control_dim = self.original_data.shape[2] - 1
    
    def __len__(self):
        if self.augment_before_creation:
            return len(self.dataset)
        else:
            return len(self.windowed_dataset)
    
    def __getitem__(self, idx):
        if self.augment_before_creation:
            return self.dataset[idx]
        else:
            # Extract window on the fly
            window = self.windowed_dataset[idx]  # Shape: (window_size, D)
            
            # Split into controls and target
            X = torch.tensor(window[:, 1:], dtype=torch.float32)  # (window_size, D-1)
            
            if self.target_as_classes:
                Y = torch.tensor(window[:, 0], dtype=torch.long)   # (window_size,)
            else:
                Y = torch.tensor(window[:, 0], dtype=torch.float32)   # (window_size,)
            
            return X, Y


def create_augmented_dataset(data, window_size, stride=1, target_as_classes=True,
                            augment_before_creation=True):
    """
    Convenience function to create an augmented dataset.
    
    Parameters
    ----------
    data : np.ndarray
        Dataset of shape (N, T, D)
    window_size : int
        Size of each window
    stride : int, optional
        Stride between windows. Default: 1
    target_as_classes : bool
        If True, treats target as class indices. Default: True
    augment_before_creation : bool
        If True, augment upfront. If False, augment on-the-fly. Default: True
    
    Returns
    -------
    AugmentedAdherenceDataset
        Augmented dataset ready for training
    """
    return AugmentedAdherenceDataset(
        data=data,
        window_size=window_size,
        stride=stride,
        target_as_classes=target_as_classes,
        augment_before_creation=augment_before_creation
    )

