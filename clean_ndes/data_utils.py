"""
Utility functions for data preprocessing and value-to-class mapping.
"""

import numpy as np
import torch


def map_values_to_classes(values, class_values=None, tolerance=1e-6):
    """
    Map continuous target values to class indices.
    
    Parameters
    ----------
    values : np.ndarray or torch.Tensor
        Target values to map (e.g., [0, 0.25, 0.5, 0.75, 1])
    class_values : list or np.ndarray, optional
        List of unique class values. If None, uses [0, 0.25, 0.5, 0.75, 1]
        Default: None
    tolerance : float, optional
        Tolerance for floating point comparison. Default: 1e-6
    
    Returns
    -------
    np.ndarray or torch.Tensor
        Class indices (0, 1, 2, 3, 4, ...)
    """
    if class_values is None:
        class_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        class_values = np.array(class_values)
    
    # Convert to numpy if torch tensor
    is_tensor = isinstance(values, torch.Tensor)
    if is_tensor:
        values_np = values.cpu().numpy() if values.is_cuda else values.numpy()
        device = values.device
    else:
        values_np = np.array(values)
        device = None
    
    # Map values to class indices using nearest neighbor (vectorized)
    # This handles both exact matches and approximate matches
    # Reshape for broadcasting: (n_values,) vs (n_classes,)
    values_flat = values_np.flatten()
    # Compute distances: (n_values, n_classes)
    distances = np.abs(values_flat[:, np.newaxis] - class_values[np.newaxis, :])
    # Find closest class index for each value
    class_indices_flat = np.argmin(distances, axis=1)
    # Reshape back to original shape
    class_indices = class_indices_flat.reshape(values_np.shape).astype(np.int64)
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        return torch.from_numpy(class_indices).to(device)
    else:
        return class_indices


def classes_to_values(class_indices, class_values=None):
    """
    Convert class indices back to original continuous values.
    
    Parameters
    ----------
    class_indices : np.ndarray or torch.Tensor
        Class indices (0, 1, 2, 3, 4, ...)
    class_values : list or np.ndarray, optional
        List of unique class values. If None, uses [0, 0.25, 0.5, 0.75, 1]
        Default: None
    
    Returns
    -------
    np.ndarray or torch.Tensor
        Original continuous values
    """
    if class_values is None:
        class_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        class_values = np.array(class_values)
    
    # Convert to numpy if torch tensor
    is_tensor = isinstance(class_indices, torch.Tensor)
    if is_tensor:
        indices_np = class_indices.cpu().numpy() if class_indices.is_cuda else class_indices.numpy()
    else:
        indices_np = np.array(class_indices)
    
    # Map class indices to values
    values = class_values[indices_np]
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        return torch.from_numpy(values).to(class_indices.device).float()
    else:
        return values


def load_reinforce_dataset(filepath, class_values=None):
    """
    Load Reinforce dataset and prepare it for training.
    
    Parameters
    ----------
    filepath : str
        Path to the .npy file containing the dataset
    class_values : list or np.ndarray, optional
        List of unique class values. If None, uses [0, 0.25, 0.5, 0.75, 1]
        Default: None
    
    Returns
    -------
    np.ndarray
        Dataset with target values converted to class indices
        Shape: (n_patients, n_days, dimensions)
        - data[..., 0] contains class indices (0, 1, 2, 3, 4)
        - data[..., 1:] contains control variables
    """
    data = np.load(filepath)
    
    # Map target values to class indices
    target_values = data[:, :, 0]
    target_classes = map_values_to_classes(target_values, class_values)
    
    # Replace target dimension with class indices
    data_processed = data.copy()
    data_processed[:, :, 0] = target_classes
    
    return data_processed

