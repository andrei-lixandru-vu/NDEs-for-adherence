"""
Data augmentation functions for trajectory data.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
try:
    from scipy import interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback: use numpy for simple interpolation
    import warnings
    warnings.warn("scipy not available. Time warping will use simpler interpolation.")


def slice_trajectory(trajectory: np.ndarray, window_size: int, stride: int = 1) -> List[np.ndarray]:
    """
    Slice a trajectory into multiple overlapping or non-overlapping windows.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory of shape (T, D) where T is time steps and D is dimensions
    window_size : int
        Size of each window
    stride : int, optional
        Stride between windows. Default: 1 (overlapping windows)
        If stride == window_size, windows are non-overlapping
    
    Returns
    -------
    List[np.ndarray]
        List of windowed trajectories, each of shape (window_size, D)
    """
    if len(trajectory) < window_size:
        # If trajectory is shorter than window, pad or return single window
        return [trajectory]
    
    windows = []
    for i in range(0, len(trajectory) - window_size + 1, stride):
        window = trajectory[i:i + window_size]
        windows.append(window)
    
    return windows


def augment_dataset(data: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    Augment dataset by slicing trajectories into windows.
    
    Parameters
    ----------
    data : np.ndarray
        Dataset of shape (N, T, D) where:
        - N: number of trajectories (patients)
        - T: number of time steps (days)
        - D: number of dimensions
    window_size : int
        Size of each window
    stride : int, optional
        Stride between windows. Default: 1 (overlapping windows)
    
    Returns
    -------
    np.ndarray
        Augmented dataset of shape (N_augmented, window_size, D)
        where N_augmented >= N (more trajectories due to slicing)
    """
    if window_size > data.shape[1]:
        raise ValueError(f"Window size ({window_size}) cannot be larger than trajectory length ({data.shape[1]})")
    
    augmented_trajectories = []
    
    for i in range(data.shape[0]):
        trajectory = data[i]  # Shape: (T, D)
        windows = slice_trajectory(trajectory, window_size, stride)
        
        for window in windows:
            augmented_trajectories.append(window)
    
    # Stack all windows into a single array
    augmented_data = np.stack(augmented_trajectories, axis=0)
    
    return augmented_data


def compute_augmentation_stats(data: np.ndarray, window_size: int, stride: int = 1) -> dict:
    """
    Compute statistics about data augmentation before applying it.
    
    Parameters
    ----------
    data : np.ndarray
        Dataset of shape (N, T, D)
    window_size : int
        Size of each window
    stride : int, optional
        Stride between windows. Default: 1
    
    Returns
    -------
    dict
        Dictionary with augmentation statistics:
        - original_trajectories: number of original trajectories
        - trajectory_length: original trajectory length
        - window_size: window size
        - stride: stride
        - windows_per_trajectory: average number of windows per trajectory
        - total_windows: total number of windows after augmentation
        - augmentation_factor: how many times the dataset size increases
    """
    original_trajectories = data.shape[0]
    trajectory_length = data.shape[1]
    
    # Compute number of windows per trajectory
    if trajectory_length < window_size:
        windows_per_traj = 1
    else:
        windows_per_traj = max(1, (trajectory_length - window_size) // stride + 1)
    
    total_windows = original_trajectories * windows_per_traj
    augmentation_factor = total_windows / original_trajectories
    
    return {
        'original_trajectories': original_trajectories,
        'trajectory_length': trajectory_length,
        'window_size': window_size,
        'stride': stride,
        'windows_per_trajectory': windows_per_traj,
        'total_windows': total_windows,
        'augmentation_factor': augmentation_factor
    }


class WindowedDataset:
    """
    Dataset that applies window slicing augmentation on the fly.
    This is memory-efficient as it doesn't create a full augmented dataset.
    """
    
    def __init__(self, data: np.ndarray, window_size: int, stride: int = 1):
        """
        Parameters
        ----------
        data : np.ndarray
            Dataset of shape (N, T, D)
        window_size : int
            Size of each window
        stride : int, optional
            Stride between windows. Default: 1
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride
        
        # Precompute all (trajectory_idx, window_start) pairs
        self.indices = []
        for traj_idx in range(data.shape[0]):
            traj_length = data.shape[1]
            if traj_length >= window_size:
                for window_start in range(0, traj_length - window_size + 1, stride):
                    self.indices.append((traj_idx, window_start))
            else:
                # If trajectory is shorter than window, use the whole trajectory
                self.indices.append((traj_idx, 0))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_idx, window_start = self.indices[idx]
        trajectory = self.data[traj_idx]
        
        # Extract window
        if trajectory.shape[0] >= self.window_size:
            window = trajectory[window_start:window_start + self.window_size]
        else:
            # If trajectory is shorter, pad with last value
            window = np.zeros((self.window_size, trajectory.shape[1]))
            window[:trajectory.shape[0]] = trajectory
            window[trajectory.shape[0]:] = trajectory[-1]
        
        return window


def time_warp_trajectory_simple(trajectory: np.ndarray, warp_factor: float = None, 
                                max_warp: float = 0.2) -> np.ndarray:
    """
    Apply simple linear time warping to a trajectory.
    
    Non-linearly stretches or compresses the time axis by a constant factor,
    simulating observing the system at a different speed.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory of shape (T, D) where T is time steps and D is dimensions
    warp_factor : float, optional
        Warping factor. If None, randomly sampled from [-max_warp, max_warp]
        warp_factor > 0: speed up (compress time)
        warp_factor < 0: slow down (stretch time)
    max_warp : float, optional
        Maximum warping magnitude (default: 0.2, i.e., ±20%)
    
    Returns
    -------
    np.ndarray
        Warped trajectory of shape (T, D)
    """
    T, D = trajectory.shape
    
    if warp_factor is None:
        # Random warping factor: 1 + random value in [-max_warp, max_warp]
        warp_factor = 1.0 + np.random.uniform(-max_warp, max_warp)
    
    # Original time points
    original_times = np.arange(T, dtype=np.float32)
    
    # Warped time points (stretch or compress)
    warped_times = original_times * warp_factor
    
    # Clamp warped times to valid range
    warped_times = np.clip(warped_times, 0, T - 1)
    
    # Interpolate trajectory to warped time points
    warped_trajectory = np.zeros_like(trajectory)
    
    if SCIPY_AVAILABLE:
        for d in range(D):
            # Use linear interpolation for each dimension
            interp_func = interpolate.interp1d(
                original_times, 
                trajectory[:, d], 
                kind='linear',
                bounds_error=False,
                fill_value=(trajectory[0, d], trajectory[-1, d])  # Extrapolate with edge values
            )
            # Interpolate to warped times
            warped_trajectory[:, d] = interp_func(warped_times)
    else:
        # Fallback: use numpy interpolation
        for d in range(D):
            warped_trajectory[:, d] = np.interp(
                warped_times, 
                original_times, 
                trajectory[:, d]
            )
    
    return warped_trajectory


def generate_warping_curve(T: int, n_control_points: int = 4, max_warp: float = 0.2) -> np.ndarray:
    """
    Generate a smooth random warping curve using cubic splines through random control points.
    
    Creates a smooth non-linear warping function that simulates observing the system
    at slightly different speeds. The warping is constrained to ±max_warp to preserve
    the original trajectory structure.
    
    The warping curve maps: original_time -> warped_time
    where warped_time indicates which point in the original trajectory to sample.
    
    Parameters
    ----------
    T : int
        Length of trajectory (number of time steps)
    n_control_points : int, optional
        Number of control points for the spline (default: 4)
        More points = more complex warping patterns
    max_warp : float, optional
        Maximum warping magnitude (default: 0.2, i.e., ±20% speed variation)
    
    Returns
    -------
    np.ndarray
        Warping curve of shape (T,) mapping original times to warped times
        warped_times[i] gives which point in original trajectory to sample at position i
    """
    # Control points evenly spaced along the output time axis
    control_output_times = np.linspace(0, T - 1, n_control_points)
    
    # For each control point, define where in the original trajectory to sample
    # Start with identity (no warp)
    control_input_times = control_output_times.copy()
    
    # Apply random warping at intermediate control points
    # Start and end points remain fixed (no warp at boundaries)
    for i in range(1, n_control_points - 1):
        # Random warp: shift the sampling point
        # Positive warp = speed up (sample earlier), negative = slow down (sample later)
        warp_amount = np.random.uniform(-max_warp, max_warp) * (T - 1)
        control_input_times[i] += warp_amount
    
    # Ensure control points are within valid range
    control_input_times = np.clip(control_input_times, 0, T - 1)
    
    # Ensure monotonicity: input times should increase (or stay same)
    # This ensures we don't go backwards in time
    for i in range(1, n_control_points):
        if control_input_times[i] < control_input_times[i-1]:
            # Set to at least the previous value
            control_input_times[i] = control_input_times[i-1]
    
    # Ensure endpoints
    control_input_times[0] = 0
    control_input_times[-1] = T - 1
    
    # Interpolate using cubic spline to create smooth warping curve
    output_times = np.arange(T, dtype=np.float32)
    
    if SCIPY_AVAILABLE:
        try:
            # Use cubic spline interpolation for smooth curves
            # 'natural' boundary conditions for smooth endpoints
            cs = interpolate.CubicSpline(
                control_output_times, 
                control_input_times, 
                bc_type='natural'
            )
            warped_times = cs(output_times)
        except Exception:
            # Fallback to linear interpolation if cubic fails
            interp_func = interpolate.interp1d(
                control_output_times, 
                control_input_times, 
                kind='linear',
                bounds_error=False,
                fill_value=(0, T - 1)
            )
            warped_times = interp_func(output_times)
    else:
        # Fallback: use numpy linear interpolation
        warped_times = np.interp(output_times, control_output_times, control_input_times)
    
    # Ensure warped times are within valid range
    warped_times = np.clip(warped_times, 0, T - 1)
    
    # Ensure monotonicity (warped times should be non-decreasing)
    # This ensures temporal causality is preserved
    for i in range(1, len(warped_times)):
        if warped_times[i] < warped_times[i-1]:
            warped_times[i] = warped_times[i-1]
    
    return warped_times


def time_warp_trajectory_smooth(trajectory: np.ndarray, n_control_points: int = 4,
                                max_warp: float = 0.2) -> np.ndarray:
    """
    Apply smooth non-linear time warping to a trajectory using cubic splines.
    
    Generates a smooth random warping curve through random control points and
    interpolates the trajectory to the warped time points. This creates more
    realistic speed variations compared to simple linear warping.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory of shape (T, D) where T is time steps and D is dimensions
    n_control_points : int, optional
        Number of control points for the spline (default: 4)
    max_warp : float, optional
        Maximum warping magnitude (default: 0.2, i.e., ±20% speed variation)
    
    Returns
    -------
    np.ndarray
        Warped trajectory of shape (T, D)
    """
    T, D = trajectory.shape
    
    # Generate smooth warping curve
    warped_times = generate_warping_curve(T, n_control_points, max_warp)
    
    # Original time points
    original_times = np.arange(T, dtype=np.float32)
    
    # Interpolate trajectory to warped time points
    warped_trajectory = np.zeros_like(trajectory)
    
    if SCIPY_AVAILABLE:
        for d in range(D):
            # Use cubic interpolation for smoother results
            try:
                cs = interpolate.CubicSpline(original_times, trajectory[:, d], bc_type='natural')
                warped_trajectory[:, d] = cs(warped_times)
            except Exception:
                # Fallback to linear interpolation
                interp_func = interpolate.interp1d(
                    original_times, 
                    trajectory[:, d], 
                    kind='linear',
                    bounds_error=False,
                    fill_value=(trajectory[0, d], trajectory[-1, d])
                )
                warped_trajectory[:, d] = interp_func(warped_times)
    else:
        # Fallback: use numpy interpolation
        for d in range(D):
            warped_trajectory[:, d] = np.interp(
                warped_times, 
                original_times, 
                trajectory[:, d]
            )
    
    return warped_trajectory


def time_warp_trajectory(trajectory: np.ndarray, method: str = 'smooth', 
                        max_warp: float = 0.2, **kwargs) -> np.ndarray:
    """
    Apply time warping to a trajectory.
    
    Convenience function that applies time warping using the specified method.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory of shape (T, D) where T is time steps and D is dimensions
    method : str, optional
        Warping method: 'simple' or 'smooth' (default: 'smooth')
    max_warp : float, optional
        Maximum warping magnitude (default: 0.2, i.e., ±20%)
    **kwargs : dict
        Additional arguments passed to the warping function
    
    Returns
    -------
    np.ndarray
        Warped trajectory of shape (T, D)
    """
    if method == 'simple':
        return time_warp_trajectory_simple(trajectory, max_warp=max_warp, **kwargs)
    elif method == 'smooth':
        return time_warp_trajectory_smooth(trajectory, max_warp=max_warp, **kwargs)
    else:
        raise ValueError(f"Unknown warping method: {method}. Use 'simple' or 'smooth'.")


def augment_dataset_time_warp(data: np.ndarray, max_warp: float = 0.2, 
                              method: str = 'smooth', probability: float = 0.5) -> np.ndarray:
    """
    Augment dataset by applying time warping to trajectories.
    
    Parameters
    ----------
    data : np.ndarray
        Dataset of shape (N, T, D) where:
        - N: number of trajectories (patients)
        - T: number of time steps (days)
        - D: number of dimensions
    max_warp : float, optional
        Maximum warping magnitude (default: 0.2, i.e., ±20%)
    method : str, optional
        Warping method: 'simple' or 'smooth' (default: 'smooth')
    probability : float, optional
        Probability of applying warping to each trajectory (default: 0.5)
    
    Returns
    -------
    np.ndarray
        Augmented dataset of shape (N, T, D)
        Note: Same number of trajectories, but each may be warped
    """
    augmented_data = data.copy()
    
    for i in range(data.shape[0]):
        # Apply warping with given probability
        if np.random.rand() < probability:
            trajectory = data[i]  # Shape: (T, D)
            warped_trajectory = time_warp_trajectory(
                trajectory, 
                method=method, 
                max_warp=max_warp
            )
            augmented_data[i] = warped_trajectory
    
    return augmented_data


def add_gaussian_noise(trajectory: np.ndarray, noise_level: float = 0.02,
                       noise_per_dimension: bool = True, 
                       per_dimension_scales: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Add Gaussian noise to a trajectory, scaled to signal magnitude.
    
    Adds small random noise to observations while keeping the underlying dynamics
    intact. The noise is scaled by the standard deviation of the trajectory to
    maintain relative signal strength.
    
    Implementation: x_aug = x + ε * σ(x) * N(0, 1)
    where σ(x) is the standard deviation (per dimension or global).
    
    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory of shape (T, D) where T is time steps and D is dimensions
    noise_level : float, optional
        Noise level (ε) as a fraction of signal magnitude (default: 0.02, i.e., 2%)
        Typical range: 0.01-0.05 (1-5% noise level)
    noise_per_dimension : bool, optional
        If True, compute noise scale per dimension independently (default: True)
        If False, use global standard deviation
    per_dimension_scales : np.ndarray, optional
        Optional array of shape (D,) specifying noise scale for each dimension.
        If provided, overrides noise_level and noise_per_dimension.
        Useful when measurement uncertainty varies by dimension.
    
    Returns
    -------
    np.ndarray
        Noisy trajectory of shape (T, D)
    
    Examples
    --------
    >>> trajectory = np.random.randn(100, 3)
    >>> noisy = add_gaussian_noise(trajectory, noise_level=0.02)
    >>> # Equivalent to user's example:
    >>> # noise = np.random.normal(0, 0.02 * np.std(trajectory, axis=0), trajectory.shape)
    >>> # augmented = trajectory + noise
    """
    T, D = trajectory.shape
    
    # Compute noise scales
    if per_dimension_scales is not None:
        # Use provided per-dimension scales
        if per_dimension_scales.shape != (D,):
            raise ValueError(f"per_dimension_scales must have shape ({D},), got {per_dimension_scales.shape}")
        noise_scales = per_dimension_scales
    elif noise_per_dimension:
        # Compute standard deviation per dimension (along time axis)
        std_per_dim = np.std(trajectory, axis=0)  # Shape: (D,)
        # Avoid division by zero
        std_per_dim = np.where(std_per_dim > 1e-10, std_per_dim, 1.0)
        noise_scales = noise_level * std_per_dim  # Shape: (D,)
    else:
        # Use global standard deviation
        global_std = np.std(trajectory)
        if global_std < 1e-10:
            global_std = 1.0
        noise_scales = noise_level * global_std
    
    # Generate Gaussian noise
    # Match user's example: np.random.normal(0, noise_level * np.std(trajectory, axis=0), trajectory.shape)
    if isinstance(noise_scales, np.ndarray) and noise_scales.ndim == 1:
        # Per-dimension scales: broadcast to (T, D)
        noise = np.random.normal(0, noise_scales, size=(T, D))
    else:
        # Scalar scale: same for all dimensions
        noise = np.random.normal(0, noise_scales, size=(T, D))
    
    # Add noise to trajectory
    augmented_trajectory = trajectory + noise
    
    return augmented_trajectory


def augment_dataset_gaussian_noise(data: np.ndarray, noise_level: float = 0.02,
                                   noise_per_dimension: bool = True,
                                   per_dimension_scales: Optional[np.ndarray] = None,
                                   probability: float = 1.0) -> np.ndarray:
    """
    Augment dataset by adding Gaussian noise to trajectories.
    
    Adds small random noise scaled to signal magnitude, simulating measurement
    uncertainty while preserving underlying dynamics.
    
    Parameters
    ----------
    data : np.ndarray
        Dataset of shape (N, T, D) where:
        - N: number of trajectories (patients)
        - T: number of time steps (days)
        - D: number of dimensions
    noise_level : float, optional
        Noise level (ε) as a fraction of signal magnitude (default: 0.02, i.e., 2%)
        Typical range: 0.01-0.05 (1-5% noise level)
    noise_per_dimension : bool, optional
        If True, compute noise scale per dimension independently (default: False)
        If False, use global standard deviation for all dimensions
    per_dimension_scales : np.ndarray, optional
        Optional array of shape (D,) specifying noise scale for each dimension.
        If provided, overrides noise_level and noise_per_dimension.
        Useful when measurement uncertainty varies by dimension.
    probability : float, optional
        Probability of applying noise to each trajectory (default: 1.0, apply to all)
    
    Returns
    -------
    np.ndarray
        Augmented dataset of shape (N, T, D)
        Note: Same number of trajectories, but each may have noise added
    """
    augmented_data = data.copy()
    
    for i in range(data.shape[0]):
        # Apply noise with given probability
        if np.random.rand() < probability:
            trajectory = data[i]  # Shape: (T, D)
            noisy_trajectory = add_gaussian_noise(
                trajectory,
                noise_level=noise_level,
                noise_per_dimension=noise_per_dimension,
                per_dimension_scales=per_dimension_scales
            )
            augmented_data[i] = noisy_trajectory
    
    return augmented_data

