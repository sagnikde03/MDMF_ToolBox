import numpy as np
import matplotlib.pyplot as plt  # Added import for plotting
from mutual_information import mutual_information_matrix

def slicing(bold_data: np.ndarray, window: float, overlap: float, plot: int, method: string) -> np.ndarray:
    """
    Computes the dynamic Functional Connectivity (dFC) matrix using a sliding window technique.

    Parameters
    ----------
    bold_data : ndarray
        A 2D numpy array of shape (time_points, regions) representing BOLD Time Series data.
    
    window : int
        The length of the sliding window in number of time points. 

    overlap : int
        The number of time points shared between consecutive windows.
    
    plot : int
        Whether to plot the dFC matrix (1 for yes, 0 for no).

    method : str
        The method for computing functional connectivity ('correlation' or 'mutual_information').    

    Returns
    -------
    dFC : ndarray
        A 2D numpy array of shape (number_of_slices, number_of_slices) representing the dynamic Functional Connectivity matrix. 
        Each entry dFC[i, j] represents the correlation between the correlation matrices of the i-th and j-th windows.
    """

    # Ensure input is a numpy array
    bold_data = np.asarray(bold_data)
    
    # Validate window size
    if window > bold_data.shape[0]:
        raise ValueError("Window size must be smaller than the number of time points in bold_data.")

    # Create time slices using a sliding window
    step = window - overlap  # Step size between windows
    indices = list(range(0, bold_data.shape[0] - window + 1, step))
    number_of_slices = len(indices)

    # Extract windows
    slices = np.array([bold_data[i:i + window, :] for i in indices])

    # Compute correlation matrices for each window
    if method == "correlation":
        correlation_matrices = np.array([np.corrcoef(slices[i], rowvar=False) for i in range(number_of_slices)])

    if method == "mutual_information":
        correlation_matrices = np.array([mutual_information_matrix(slices[i]) for i in range(number_of_slices)])

    
    # Compute dFC matrix (correlation between correlation matrices)
    dFC = np.zeros((number_of_slices, number_of_slices))

    if method == "correlation":
    
        for i in range(number_of_slices):
            for j in range(i, number_of_slices):  # Only compute upper triangle
                m1 = correlation_matrices[i].flatten()
                m2 = correlation_matrices[j].flatten()
                dFC[i, j] = dFC[j, i] = np.corrcoef(m1, m2)[0, 1]

    if method == "mutual_information":

        for i in range(number_of_slices):
            for j in range(i, number_of_slices):  # Only compute upper triangle
                m1 = correlation_matrices[i].flatten()
                m2 = correlation_matrices[j].flatten()
                m = np.array([m1, m2])
                dFC[i, j] = dFC[j, i] = mutual_information_matrix(m)
                
                
    # Optional: Plot the dFC matrix
    if plot == 1:
        plt.imshow(dFC, aspect='auto', cmap='jet')
        plt.colorbar(label="Correlation")
        plt.xlabel('Time Windows', fontsize=14, fontweight='bold')
        plt.ylabel('Time Windows', fontsize=14, fontweight='bold')
        plt.title('Dynamic Functional Connectivity (dFC)', fontsize=16, fontweight='bold')
        plt.show()

    return dFC
