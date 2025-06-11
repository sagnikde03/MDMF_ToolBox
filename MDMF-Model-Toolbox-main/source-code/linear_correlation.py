import numpy as np

def matrix_corrcoef(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
    Computes the Pearson correlation coefficient between two square matrices.

    Parameters
    ----------
    matrix1 : np.ndarray
        First input square matrix.
    matrix2 : np.ndarray
        Second input square matrix.

    Returns
    -------
    float
        Pearson correlation coefficient between the flattened matrices.

    Raises
    ------
    ValueError
        If the input matrices are not square or their dimensions do not match.
    """
    if matrix1.shape[0] != matrix1.shape[1] or matrix2.shape[0] != matrix2.shape[1]:
        raise ValueError("Both input matrices must be square.")
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both input matrices must have the same dimensions.")
    
    vector1 = matrix1.flatten()
    vector2 = matrix2.flatten()
    return np.corrcoef(vector1, vector2)[0, 1]
