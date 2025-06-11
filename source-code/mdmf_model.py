import numpy as np
import matplotlib.pyplot as plt

def firing_rate(Tglu: float, Tgaba: float, sc: np.ndarray | int, num_min: float, method: str, plot: bool = False) -> tuple[np.ndarray, float]:
    """
    Simulates excitatory and inhibitory neural firing rates and synaptic gating variables
    using either the Euler or RK4 integration method over a given simulation time.
    
    Parameters 
    ----------
    Tglu : float
        Excitatory neurotransmitter concentration.
    Tgaba : float
        Inhibitory neurotransmitter concentration.
    sc : int (0) or ndarray
        Structural connectivity matrix (or 0 for a single area model).
    num_min : float (> 200_000)
        Simulation time in minutes.
    method : str
        Integration method ('Euler' or 'RK4').
    plot : bool, optional
        Whether to plot the firing rate (default is False).
    
    Returns
    -------
    tuple
        - Synaptic gating variable (ndarray)
        - Maximum mean firing rate of the network (float)
    """
    
    if not isinstance(Tglu, float) or not isinstance(Tgaba, float):
        raise TypeError("Both Tglu and Tgaba must be float values.")
    if not isinstance(num_min, float):
        raise TypeError("num_min must be a float value.")
    if not isinstance(sc, (np.ndarray, int)):
        raise TypeError("sc must be either an integer (0) or a 2D NumPy array.")
    if isinstance(sc, np.ndarray) and (sc.ndim != 2 or sc.dtype.kind not in {'f', 'i'}):
        raise TypeError("sc must be a 2D array containing float or integer values.")
    if method not in {"RK4", "Euler"}:
        raise ValueError("method must be either 'Euler' or 'RK4'.")
    
    num_sim = int(num_min * 60 * 1_000)
    G, I0, JNMDA, WE, WI, wplus = 0.69, 0.382, 0.15, 1.0, 0.7, 1.4
    aI, bI, dI, aE, bE, dE = 615, 177, 0.087, 310, 125, 0.16
    alphaE, betaE, alphaI, betaI, sigma, gamma, rho = 0.072, 0.0066, 0.53, 0.18, 0.001, 1, 3
    
    nAreas = 1 if isinstance(sc, int) and sc == 0 else len(sc[:, 0])
    
    def dSE(SE, SI, J, IE, II, rE, rI):
        return -betaE * SE + (alphaE / 1000) * Tglu * rE * (1 - SE) + sigma * 0
    
    def dSI(SE, SI, J, IE, II, rE, rI):
        return -betaI * SI + (alphaI / 1000) * Tgaba * rI * (1 - SI) + sigma * 0
    
    def dJ(SE, SI, J, IE, II, rE, rI):
        return gamma * (rI / 1000) * (rE - rho) / 1000
    
    SE, SI, J = np.full(nAreas, 0.001), np.full(nAreas, 0.001), np.ones(nAreas)
    dt = 0.1
    synaptic_gating_var, excit_firing_rate = np.zeros((num_sim, nAreas)), np.zeros((num_sim, nAreas))
    
    for i in range(num_sim):
        IE = WE * I0 + wplus * JNMDA * SE + (G * JNMDA * np.matmul(sc / np.max(sc), SE) if isinstance(sc, np.ndarray) else 0) - J * SI
        II = WI * I0 + JNMDA * SE - SI
        rE = (aE * IE - bE) / (1 - np.exp(-dE * (aE * IE - bE)))
        rI = (aI * II - bI) / (1 - np.exp(-dI * (aI * II - bI)))

        if method == 'RK4':
            k1_SE, k1_SI, k1_J = dt * dSE(SE, SI, J, IE, II, rE, rI), dt * dSI(SE, SI, J, IE, II, rE, rI), dt * dJ(SE, SI, J, IE, II, rE, rI)
            k2_SE, k2_SI, k2_J = dt * dSE(SE + 0.5 * k1_SE, SI + 0.5 * k1_SI, J + 0.5 * k1_J, IE, II, rE, rI), dt * dSI(SE + 0.5 * k1_SE, SI + 0.5 * k1_SI, J + 0.5 * k1_J, IE, II, rE, rI), dt * dJ(SE + 0.5 * k1_SE, SI + 0.5 * k1_SI, J + 0.5 * k1_J, IE, II, rE, rI)
            k3_SE, k3_SI, k3_J = dt * dSE(SE + 0.5 * k2_SE, SI + 0.5 * k2_SI, J + 0.5 * k2_J, IE, II, rE, rI), dt * dSI(SE + 0.5 * k2_SE, SI + 0.5 * k2_SI, J + 0.5 * k2_J, IE, II, rE, rI), dt * dJ(SE + 0.5 * k2_SE, SI + 0.5 * k2_SI, J + 0.5 * k2_J, IE, II, rE, rI)
            k4_SE, k4_SI, k4_J = dt * dSE(SE + k3_SE, SI + k3_SI, J + k3_J, IE, II, rE, rI), dt * dSI(SE + k3_SE, SI + k3_SI, J + k3_J, IE, II, rE, rI), dt * dJ(SE + k3_SE, SI + k3_SI, J + k3_J, IE, II, rE, rI)
            SE += (k1_SE + 2 * k2_SE + 2 * k3_SE + k4_SE) / 6
            SI += (k1_SI + 2 * k2_SI + 2 * k3_SI + k4_SI) / 6
            J  += (k1_J + 2 * k2_J + 2 * k3_J + k4_J) / 6
        else:
            SE += dt * dSE(SE, SI, J, IE, II, rE, rI)
            SI += dt * dSI(SE, SI, J, IE, II, rE, rI)
            J  += dt * dJ(SE, SI, J, IE, II, rE, rI)
        
        SE, SI = np.clip(SE, 0, 1), np.clip(SI, 0, 1)
        synaptic_gating_var[i, :], excit_firing_rate[i, :] = SE, rE
    
    if plot:
        plt.plot(excit_firing_rate)
        plt.xlabel(f'{num_min} minutes')
        plt.ylabel('BOLD Signals')
        plt.show()
    
    return synaptic_gating_var, np.max(np.mean(excit_firing_rate[200_000:, :], axis=1))
