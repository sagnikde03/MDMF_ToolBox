import numpy as np
import matplotlib.pyplot as plt

def bold(r: np.ndarray, TR: float, plot: int) -> np.ndarray:
    """
    Converts the excitatory firing rate Time Series to the BOLD Time Series.

    Parameters
    ----------
    r : ndarray
        Excitatory firing rates with shape (number of time steps, number of ROIs).
    TR : int or float
        "TR" stands for "Repetition Time" (should be positive).
    plot : int, optional
        If 1, plots the BOLD signals. Default is 0.

    Returns
    -------
    ndarray
        Matrix with shape (number of time steps, number of ROIs) giving row-wise BOLD Time Series signals per ROI. 
    """

    # Input validation
    if not isinstance(r, np.ndarray):
        raise ValueError("Error: 'r' must be a NumPy array.")
    if not np.issubdtype(r.dtype, np.floating):
        raise ValueError("Error: 'r' must contain only float values.")
    if not isinstance(TR, (int, float)) or TR <= 0:
        raise ValueError("Error: 'TR' (Repetition Time) must be a positive number.")
    if not isinstance(plot, int) or plot not in [0, 1]:
        raise ValueError("Error: 'plot' must be either 0 (no plot) or 1 (plot).")

    # Model parameters
    taus, tauf, tauo = 0.65, 0.41, 0.98
    alpha, Eo, vo = 0.32, 0.34, 0.02
    itaus, itauf, itauo, ialpha = 1/taus, 1/tauf, 1/tauo, 1/alpha
    k1, k2, k3 = 7 * Eo, 2, 2 * Eo - 0.2

    # Shape handling
    num_sim, nAreas = r.shape  # Corrected shape assignment

    # Initialize state variables
    x1, x2, x3, x4 = np.ones(nAreas), np.ones(nAreas), np.ones(nAreas), np.ones(nAreas)
    dt = 0.001
    bold_ts = np.zeros((num_sim // int(TR * 1000), nAreas))

    # Slope functions
    def slope_x1(x1, x2, x3, x4): return -itaus * x1 - itauf * (x2 - 1)
    def slope_x2(x1, x2, x3, x4): return x1
    def slope_x3(x1, x2, x3, x4): return itauo * (x2 - (x3) ** ialpha)
    def slope_x4(x1, x2, x3, x4): return itauo * (x2 * (1 - (1 - Eo) ** (1 / x2)) / Eo - (x3 ** ialpha) * x4 / x3)

    # Simulation loop
    for i in range(num_sim):
        x1 += dt * (slope_x1(x1, x2, x3, x4) + r[i, :])
        x2 += dt * slope_x2(x1, x2, x3, x4)
        x3 += dt * slope_x3(x1, x2, x3, x4)
        x4 += dt * slope_x4(x1, x2, x3, x4)

        if i % int(TR * 1000) == 0:
            bold_signal = 100 / Eo * vo * (k1 * (1 - x4) + k2 * (1 - x4 / x3) + k3 * (1 - x3))
            bold_ts[i // int(TR * 1000), :] = bold_signal

    # Plot if requested
    if plot == 1:
        time_axis = np.arange(bold_ts.shape[0]) * TR
        plt.figure(figsize=(10, 5))
        plt.plot(time_axis, bold_ts)
        plt.xlabel("Time (seconds)", fontsize=14, fontweight='bold', fontname='serif')
        plt.ylabel("BOLD Signals", fontsize=14, fontweight='bold', fontname='serif')
        plt.title("BOLD Time Series", fontsize=16, fontweight='bold', fontname='serif')
        plt.grid(True)
        plt.show()

    return bold_ts
