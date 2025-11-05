"""
Dataset classes for adherence data.
"""

import torch
from torch.utils.data import Dataset


class AdherenceDataset(Dataset):
    """
    Dataset for full trajectories.
    
    Returns (X, Y) where:
    - X: (T, D-1) control variables over time
    - Y: (T,) adherence target over time
    """
    
    def __init__(self, data):
        """
        Parameters
        ----------
        data : np.ndarray
            Shape (N, T, D) where:
            - data[..., 0] is adherence
            - data[..., 1:] are control variables
        """
        self.X = torch.tensor(data[..., 1:], dtype=torch.float32)  # (N, T, D-1)
        self.Y = torch.tensor(data[..., 0], dtype=torch.float32)   # (N, T)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class SampledLength2AdherenceDataset(Dataset):
    """
    Dataset for sliding window samples of length 2.
    
    Samples all possible length-2 windows from full trajectories.
    Returns (X, Y) where:
    - X: (2, D-1) control variables for 2 consecutive timesteps
    - Y: (2,) adherence for 2 consecutive timesteps
    """
    
    def __init__(self, data):
        """
        Parameters
        ----------
        data : np.ndarray
            Shape (N, T, D) where:
            - data[..., 0] is adherence
            - data[..., 1:] are control variables
        """
        self.data = data
        self.N = data.shape[0]
        self.T = data.shape[1]
        self.D = data.shape[2]
        
        # Construct all (patient_idx, t_start) index pairs for length-2 windows
        self.indices = [
            (i, t) for i in range(self.N) for t in range(self.T - 1)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        patient_idx, t_start = self.indices[idx]
        
        # Get a window of length 2 starting at t_start
        data_slice = self.data[patient_idx, t_start:t_start+2, :]
        X = torch.tensor(data_slice[:, 1:], dtype=torch.float32)  # (2, D-1)
        Y = torch.tensor(data_slice[:, 0], dtype=torch.float32)   # (2,)
        return X, Y
