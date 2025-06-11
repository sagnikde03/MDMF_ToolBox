import numpy as np
from scipy.stats import ks_2samp

def ks_distance_between_matrices(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
    This function calculates the Kolmogorov-Smirnov (KS) distance between two matrices.
    The KS distance is a non-parametric test that measures the difference between the 
    cumulative distribution functions (CDF) of two sample sets. It is used to assess 
    whether two datasets (or matrices, in this case) come from the same distribution.

    Parameters
    ----------
    matrix1 : np.ndarray
        The first matrix.
    matrix2 : np.ndarray
        The second matrix, which must have the same shape as matrix1.

    Returns
    -------
    float
        The Kolmogorov-Smirnov distance between the two matrices.
    
    Raises
    ------
    ValueError
        If `matrix1` and `matrix2` have different shapes.
    """

    # Validate that matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Input matrices must have the same shape.")

    # Flatten the matrices into 1D arrays
    flat_matrix1 = matrix1.flatten()
    flat_matrix2 = matrix2.flatten()

    # Compute the Kolmogorov-Smirnov distance between the two flattened matrices
    ks_distance, _ = ks_2samp(flat_matrix1, flat_matrix2)

    return ks_distance
