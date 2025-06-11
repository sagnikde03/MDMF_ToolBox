import numpy as np
import matplotlib.pyplot as plt

def func_connec(bold_data: np.ndarray, plot: int) -> np.ndarray:
    """
    Computes the Functional Connectivity (FC) matrix using Pearson correlation.

    Parameters:
    -----------
    bold_data : np.ndarray
        2D array (time_points, regions) containing BOLD signals.
    plot : int 
        Whether to plot the result or not (1 or 0).

    Returns:
    --------
    np.ndarray
        Functional Connectivity matrix of shape (regions, regions).
    """
    
    # Validate input
    if not isinstance(bold_data, np.ndarray) or bold_data.ndim != 2:
        raise ValueError("Input must be a 2D numpy array (time_points, regions).")
    
    if bold_data.shape[0] < 2:
        raise ValueError("At least two time points are required for correlation computation.")
    
    # Compute correlation matrix
    fc_sim = np.corrcoef(bold_data, rowvar=False)
    
    # Plot if requested
    if plot == 1:
        plt.imshow(fc_sim, aspect='auto', cmap='viridis')
        plt.colorbar(label="Correlation")
        plt.xlabel("Regions of Interest", fontsize=14, fontweight="bold")
        plt.ylabel("Regions of Interest", fontsize=14, fontweight="bold")
        plt.title("Functional Connectivity Matrix", fontsize=16, fontweight="bold")
        plt.show()
        
    return fc_sim
