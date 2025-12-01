"""
Synthetic data generation for adherence prediction.
"""

import numpy as np


def generate_synthetic_reinforce_data(N, T, D):
    """
    Generate synthetic data with complex adherence drop rules.
    
    Parameters
    ----------
    N : int
        Number of trajectories (patients)
    T : int
        Number of time steps (days)
    D : int
        Number of dimensions
    
    Returns
    -------
    data : np.ndarray
        Shape (N, T, D) containing trajectory data
    adherence_dropped_at : list
        List of (patient_idx, timestep) tuples where adherence dropped
    """
    data = np.zeros((N, T, D))
    adherence_dropped_at = []

    for n in range(N):
        count_last6dim_sameInARow = [0] * (D-1)
        prev_value_last6dim_sameInARow = [-1] * (D-1)
        for t in range(T):
            data[n, t, 1:] = np.random.randint(0, 2, D-1)
            for d in range(1, D):
                if data[n, t, d] == prev_value_last6dim_sameInARow[d-1]:
                    count_last6dim_sameInARow[d-1] += 1
                else:
                    count_last6dim_sameInARow[d-1] = 0
                prev_value_last6dim_sameInARow[d-1] = data[n, t, d]

            # Drop adherence to 0 if 2+ reminder factors have the same value 3+ times in a row
            if sum([c >= 2 for c in count_last6dim_sameInARow]) >= 2:
                data[n, t, 0] = 0
                adherence_dropped_at.append((n, t))
            else:
                data[n, t, 0] = 1
    return data, adherence_dropped_at


def generate_synthetic_data_super_simple(N, T, D):
    """
    Generate simplified synthetic data.
    
    Adherence drops to 0 if reminder factor 1 (index 1) is 1 at previous timestep.
    
    Parameters
    ----------
    N : int
        Number of trajectories (patients)
    T : int
        Number of time steps (days)
    D : int
        Number of dimensions
    
    Returns
    -------
    data : np.ndarray
        Shape (N, T, D) containing trajectory data
    adherence_dropped_at : list
        List of (patient_idx, timestep) tuples where adherence dropped
    """
    data = np.zeros((N, T, D))
    adherence_dropped_at = []

    # Set dims 1-6 to random 0 or 1 (50% each)
    data[:, :, 1:] = np.random.randint(0, 2, size=(N, T, D-1))
    
    # For each trajectory (patient)
    for n in range(N):
        # Set initial adherence at t=0 to 1
        data[n, 0, 0] = 1

        for t in range(1, T):
            # At t, check dim 1 at t-1
            if data[n, t-1, 1] == 1:
                data[n, t, 0] = 0
                adherence_dropped_at.append((n, t))
            else:
                data[n, t, 0] = 1
    return data, adherence_dropped_at


def generate_synthetic_data_super_simple_Interactions2StepsApart(N, T, D):
    """
    Generate simplified synthetic data.
    
    Adherence drops to 0 if reminder factor 1 (index 1) is 1 two timepoints before.
    
    Parameters
    ----------
    N : int
        Number of trajectories (patients)
    T : int
        Number of time steps (days)
    D : int
        Number of dimensions
    
    Returns
    -------
    data : np.ndarray
        Shape (N, T, D) containing trajectory data
    adherence_dropped_at : list
        List of (patient_idx, timestep) tuples where adherence dropped
    """
    data = np.zeros((N, T, D))
    adherence_dropped_at = []

    # Set dims 1-6 to random 0 or 1 (50% each)
    data[:, :, 1:] = np.random.randint(0, 2, size=(N, T, D-1))
    
    # For each trajectory (patient)
    for n in range(N):
        # Set initial adherence at t=0 to 1
        data[n, 0, 0] = 1

        for t in range(1, T):
            # At t, check dim 1 at t-2
            if data[n, t-2, 1] == 1:
                data[n, t, 0] = 0
                adherence_dropped_at.append((n, t))
            else:
                data[n, t, 0] = 1
    return data, adherence_dropped_at
