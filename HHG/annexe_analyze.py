"""
annexe pour l'analyse des resultats de HHG
:auteur: Maxence BARRE
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import random as rd



def compute_rho(psi):
    """
    Compute the density matrix from the wavefunction.
    
    :param psi: Wavefunction data
    :return: Density matrix
    """
    if isinstance(psi, np.ndarray):
        return np.abs(psi) ** 2
    elif isinstance(psi, h5py.Dataset):
        return np.abs(psi[...]) ** 2
    else:
        raise TypeError("Unsupported type for psi. Must be numpy array or h5py Dataset.")



def plot_a_matrix(hdf5_group: h5py.Group, index:int = None, title=None, xlabel=None, ylabel=None, function_to_apply=None):
    """
    Plot a random matrix from the HDF5 group.
    
    :param hdf5_group: HDF5 group containing the data
    """
    if index is not None:
        data = hdf5_group[list(hdf5_group.keys())[index]]
    else:
        matrix_name = rd.choice(list(hdf5_group.keys()))
        data = hdf5_group[matrix_name]
    name = data.name
    data = np.array(data)

    plt.figure()
    if function_to_apply is not None:
        data = function_to_apply(data)
    plt.imshow(data, cmap='viridis')
    plt.xlabel(xlabel if xlabel else "X-axis")
    plt.ylabel(ylabel if ylabel else "Y-axis")
    if title is not None:
        plt.title(title)
    plt.colorbar(label='Intensity')
    plt.set_cmap('turbo')
    plt.suptitle(f"Matrix: {name}")
    plt.tight_layout()