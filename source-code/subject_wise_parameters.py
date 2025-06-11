import numpy as np

def find_optimal_parameters(firing_rate: np.ndarray, metastability: np.ndarray, fcd: np.ndarray, 
                            fcc: np.ndarray, mid: np.ndarray, dfc_ksd: np.ndarray) -> tuple:
    '''
    This function searches for the optimal Glutamate and GABA model parameters in the parameter space 
    based on multiple criteria (firing rate, metastability, functional connectivity measures, etc.).
    
    Parameters
    ----------
    firing_rate : np.ndarray
        firing_rate[i,j] is the average firing rate of the simulated nodes for Tgaba = 0.25 + 0.25*i, Tglu = 7.5 + 0.25*j;  0 < i, j < 30
    metastability : np.ndarray
        metastability[i,j] is the metastability index of the simulated BOLD signals.
    fcd : np.ndarray
        fcd[i,j] is the Euclidean distance between simulated and empirical functional connectivity matrices.
    fcc : np.ndarray
        fcc[i, j] is the correlation between simulated and empirical functional connectivity matrices.
    mid : np.ndarray
        mid[i, j] is the Euclidean distance between simulated and empirical mutual information matrices.
    dfc_ksd : np.ndarray
        dfc_ksd[i, j] is the Kolmogorov-Smirnov distance between simulated and empirical dynamic functional connectivity matrices.

    Returns
    -------
    tuple
        (optimal_row, optimal_col) values of the optimal parameters.
    '''
    
    # Compute least squares distance for all parameters, including firing rate and metastability
    distances = np.sqrt((firing_rate - 9.0)**2 + (metastability - 0.025)**2 + 
                        fcd**2 + fcc**2 + mid**2 + dfc_ksd**2)
    
    # Select the index that minimizes the least squares distance
    optimal_idx = np.unravel_index(np.argmin(distances), distances.shape)
    optimal_row, optimal_col = optimal_idx
    
    return 0.25 + 0.25 * optimal_row, 7.5 + 0.25 * optimal_col
