import numpy as np

def eucledian_dist(matrix1: np.ndarray, matrix2: np.ndarray, nAreas: int) -> float:
    """
    This function calculates the Frobenius Norm or Eucledian Distance between two matrices.
    The Frobenius norm is the square root of the sum of the squared differences of corresponding elements.

    Parameters
    ----------
    matrix1 : np.ndarray
        The first matrix.
    matrix2 : np.ndarray
        The second matrix, which must have the same shape as matrix1.
    nAreas : int
        Number of region of interests or ROIs

    Returns
    -------
    float
        The Frobenius norm of the difference between the two matrices, optionally normalized by 68.
    """
    
    # Ensure that the matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Input matrices must have the same shape")
    
    # Flatten the matrices into 1D arrays
    vector1 = matrix1.flatten()
    vector2 = matrix2.flatten()
    
    # Compute the squared differences element-wise
    squared_diff = (vector1 - vector2) ** 2
    
    # Sum the squared differences and take the square root
    distance = np.sqrt(np.sum(squared_diff))
    
    # Return the distance, divided by 68 if that is part of the requirement
    return distance / nAreas
