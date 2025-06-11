import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
import matplotlib.pyplot as plt

def mutual_information_matrix(bold_data: np.ndarray, plot: int = 0) -> np.ndarray:
    """
    Computes the mutual information (MI) matrix for a set of signals using k-nearest neighbors.

    Parameters
    ----------
    bold_data : np.ndarray
        A 2D numpy array where each column represents a signal (e.g., BOLD time series from a different brain region),
        and each row represents a time point. Shape: (n_samples, n_signals).
    plot : int, optional
        Whether to plot the result or not (1 or 0). Default is 0.

    Returns
    -------
    I_XY : np.ndarray
        A 2D numpy array (n_signals, n_signals) representing the mutual information matrix.
        Each element I_XY[i, j] is the mutual information between the i-th and j-th signal.
    """
    # Input validation
    if not isinstance(bold_data, np.ndarray):
        raise TypeError("bold_data must be a numpy array")
    if bold_data.ndim != 2:
        raise ValueError("bold_data must be a 2D array with shape (n_samples, n_signals)")
    if not isinstance(plot, int) or plot not in [0, 1]:
        raise ValueError("plot must be either 0 or 1")
    
    n_samples, n_signals = bold_data.shape
    k = 1  # Fixed k for nearest neighbors
    
    if k >= n_samples:
        raise ValueError("k must be smaller than the number of samples")
    
    I_XY = np.zeros((n_signals, n_signals))

    for i in range(n_signals):
        for j in range(i + 1, n_signals):
            X, Y = bold_data[:, i], bold_data[:, j]
            points = np.column_stack((X, Y))

            # Build KD-tree and find distances to the k-th nearest neighbor
            tree = cKDTree(points)
            distances, _ = tree.query(points, k=k+1)  # k+1 because the first neighbor is the point itself
            epsilon = distances[:, -1]  # Get k-th nearest neighbor distances

            # Count neighbors within epsilon using KD-tree
            n_x = np.array([np.sum(np.abs(X - X[m]) <= epsilon[m]) - 1 for m in range(n_samples)])
            n_y = np.array([np.sum(np.abs(Y - Y[m]) <= epsilon[m]) - 1 for m in range(n_samples)])

            # Compute mutual information using digamma functions
            mi = digamma(k) + digamma(n_samples) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
            I_XY[i, j] = I_XY[j, i] = max(0, mi)  # MI is symmetric and should be non-negative
            
    # Plot if requested
    if plot == 1:
        plt.imshow(I_XY, aspect='auto', cmap='viridis')
        plt.colorbar(label="Mutual Information")
        plt.xlabel("Regions of Interest", fontsize=14, fontweight="bold")
        plt.ylabel("Regions of Interest", fontsize=14, fontweight="bold")
        plt.title("Mutual Information Matrix", fontsize=16, fontweight="bold")
        plt.show()
    
    return I_XY
