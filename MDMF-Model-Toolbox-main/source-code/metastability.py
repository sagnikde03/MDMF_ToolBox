import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def metastability_index(bold_data: np.ndarray, plot: int) -> float:
    """
    Computes and optionally plots the metastability index of BOLD (Blood Oxygen Level Dependent) time-series data.

    The metastability index quantifies variability in phase synchronization across brain regions over time.
    It is computed as the standard deviation of the Kuramoto order parameter, which measures global 
    phase coherence across regions at each time point.

    Parameters
    ----------
    bold_data : ndarray
        A 2D numpy array where rows represent time points and columns represent brain regions.
        Shape: (time_points, regions).
    
    plot : int ( 0 or 1)
        Whether to plot the result or not (1 or 0).

    Returns
    -------
    m : float
        Metastability index (higher values indicate greater variability in phase synchronization).
    """

    # Compute the phase of the BOLD signal using Hilbert transform
    phase_signal = np.angle(hilbert(bold_data, axis=0))  

    # Convert phase to complex exponential form
    phase_complex = np.exp(1j * phase_signal)  

    # Compute the Kuramoto order parameter at each time point
    order_parameter = np.abs(np.mean(phase_complex, axis=1))  

    # Compute the standard deviation of the Kuramoto order parameter over time
    m = np.std(order_parameter)

    # Plot the order parameter time series if requested
    if plot == 1:
        plt.figure(figsize=(8, 5))
        plt.plot(order_parameter, color='blue', linewidth=2)
        plt.xlabel("Time Points", fontsize=14, fontweight='bold')
        plt.ylabel("Kuramoto Order Parameter", fontsize=14, fontweight='bold')
        plt.title("Time Evolution of Phase Synchronization", fontsize=16, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    return m
