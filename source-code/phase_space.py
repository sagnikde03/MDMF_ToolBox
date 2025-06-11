import numpy as np
import matplotlib.pyplot as plt
from .ks_distance_between_matrices import ks_distance_between_matrices
from .eucledian_distance import eucledian_dist
from .dynamic_functional_connectivity import slicing
from .metastability import metastability_index
from .mutual_information import mutual_information_matrix
from .functional_connectivity import func_connec
from .BOLD_model import bold
from .mdmf_model import firing_rate
from .subject_wise_parameters import find_optimal_parameters

def find_parameters(Tglu_low, Tglu_high, Tgaba_low, Tgaba_high, discreteness,  
                   sc: np.ndarray | int, num_min: float, method: str,  ):
    """
    Finds optimal GABA and Glutamate model parameters over a parameter space.
    
    Parameters:
    ----------
    Tglu_low, Tglu_high : float
        Lower and upper limits for glutamate (y-axis).
    Tgaba_low, Tgaba_high : float
        Lower and upper limits for GABA (x-axis).
    discreteness : float
        Step size for parameter search (should be between 0 and 0.3).
    FC_emp, mi_emp, dFC_emp : np.ndarray
        Empirical matrices for functional connectivity, mutual information, and dynamic FC.
    window, overlap : int
        Parameters for dynamic functional connectivity slicing.
    plot : int (0 or 1)
        If 1, plots the parameter maps.
    
    Returns:
    ----------
    Xi : np.ndarray
        A matrix storing results of different measures over parameter space.
    """
    Tglu_vals = np.arange(Tglu_low, Tglu_high + discreteness, discreteness)
    Tgaba_vals = np.arange(Tgaba_low, Tgaba_high + discreteness, discreteness)
    n_glu, n_gaba = len(Tglu_vals), len(Tgaba_vals)
    
    Xi = np.zeros((n_glu, n_gaba, 6))
    
    for i, Tglu in enumerate(Tglu_vals):
        for j, Tgaba in enumerate(Tgaba_vals):
            r, x = firing_rate(Tglu, Tgaba, sc, num_min, method: "RK4")
            bold_sim = bold(r,TR,plot=0)
            FC_sim = func_connec(bold_sim,0)
            fc_dist = eucledian_dist(FC_sim, FC_emp)
            fc_corr = matrix_corrcoef(FC_sim, FC_emp,)
            meta = metastability_index(bold_sim,0)
            mi_sim = mutual_information_matrix(bold_sim,0)
            mi_dist = eucledian_dist(mi_sim, mi_emp)
            dFC_sim = slicing(bold_sim, window, overlap, 0)
            ksd = ks_distance_between_matrices(dFC_sim, dFC_emp)
            
            Xi[i, j, :] = [x, meta, fc_dist, fc_corr, mi_dist, ksd]
    
    a, b = find_optimal_parameters(firing_rate=Xi[:,:,0], metastability=Xi[:,:,1], 
                            fcd=Xi[:,:,2], fcc=Xi[:,:,3], mid=Xi[:,:,4], dfc_ksd=Xi[:,:,5])
    
    if plot == 1:
        titles = ["Firing Rate", "Metastability", "FC Distance", "FC Correlation", "MI Distance", "KS Distance"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i in range(6):
            im = axes[i].imshow(Xi[:, :, i].T, aspect='equal', origin='lower', 
                                 extent=[Tglu_vals[0], Tglu_vals[-1], Tgaba_vals[0], Tgaba_vals[-1]], cmap='viridis')
            axes[i].set_title(titles[i], fontsize=20, fontname='serif')
            axes[i].set_xlabel("Tglu (mmol)", fontsize=20, fontname='serif')
            axes[i].set_ylabel("Tgaba (mmol)", fontsize=20, fontname='serif')
            fig.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.show()
    
    print("Optimization complete. Results plotted.")
    print(f"Optimal Glutamate and GABA parameters are: {a} and {b}")
  
    return a, b

