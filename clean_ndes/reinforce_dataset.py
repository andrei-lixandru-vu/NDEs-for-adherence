"""
Helper functions for working with the Reinforce dataset.
"""

import numpy as np
from .data_utils import load_reinforce_dataset, map_values_to_classes, classes_to_values
from .datasets import AdherenceDataset


def create_reinforce_dataset(filepath, class_values=None):
    """
    Load and prepare Reinforce dataset for training.
    
    Parameters
    ----------
    filepath : str
        Path to the .npy file containing the Reinforce dataset
    class_values : list or np.ndarray, optional
        List of unique class values. If None, uses [0, 0.25, 0.5, 0.75, 1]
        Default: None
    
    Returns
    -------
    AdherenceDataset
        Dataset ready for training with class indices as targets
    """
    # Load and preprocess dataset (maps values to class indices)
    data = load_reinforce_dataset(filepath, class_values)
    
    # Create dataset with target_as_classes=True (default)
    dataset = AdherenceDataset(data, target_as_classes=True)
    
    return dataset


def get_reinforce_config(data):
    """
    Get configuration parameters for Reinforce dataset.
    
    Parameters
    ----------
    data : np.ndarray
        Dataset array of shape (n_patients, n_days, dimensions)
    
    Returns
    -------
    dict
        Configuration dictionary with:
        - n_patients: number of patients
        - n_days: number of days
        - n_dimensions: total dimensions
        - n_controls: number of control dimensions
        - n_classes: number of target classes (5 for Reinforce)
        - input_dim: input dimension for model (target + controls)
        - output_dim: output dimension for model (number of classes)
    """
    n_patients, n_days, n_dimensions = data.shape
    n_controls = n_dimensions - 1  # All dimensions except target
    n_classes = 5  # Reinforce dataset has 5 classes
    
    return {
        'n_patients': n_patients,
        'n_days': n_days,
        'n_dimensions': n_dimensions,
        'n_controls': n_controls,
        'n_classes': n_classes,
        'input_dim': n_dimensions,  # Target + controls
        'output_dim': n_classes,    # 5 classes
    }

