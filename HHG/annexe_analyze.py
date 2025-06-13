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


def plot_direct_info(hdf5_file):
    """
    this function will plot graphs that can be directly plotted from the hdf5 file
    the aim is to gather here all the plots that are always done not to put to much code in the main file
    :param hdf5_file: h5py File object
    :plots done:
        - the potential field
        - the electric field
    """

    parametres = hdf5_file["simulation_parameters"].attrs
    x = np.arange(parametres["x_start"], parametres["x_end"], parametres["dx"])
    t = np.arange(parametres["t_start"], parametres["t_end"], parametres["dt"])

    # potential field
    potential_field = hdf5_file["potentials_fields"]["potentiel_spatial"]
    plt.figure()
    plt.plot(potential_field[:, 0], potential_field[:, 1], label="Potential Field")
    plt.xlabel("Position (a.u.)")
    plt.ylabel("Potential (a.u.)")
    plt.title("Potential Field")
    plt.legend()
    plt.grid()

    # electric field
    electric_field = hdf5_file["potentials_fields"]["champE"]
    plt.figure()
    plt.plot(electric_field[:, 0], electric_field[:, 1], label="Electric Field")
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Electric Field (a.u.)")
    plt.title("Electric Field")
    plt.legend()
    plt.grid()

    # plot the very beginning of the simulation to see how it behaves
    data = compute_rho(hdf5_file["psi_history"][list(hdf5_file["psi_history"].keys())[0]])
    num_batch = 0
    plt.figure()
    plt.imshow(data.T, cmap='turbo', extent=( t[num_batch*data.shape[0]], t[(num_batch+1)*data.shape[0] -1] , x[0], x[-1]), aspect='auto')
    temp = electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 1] / np.max(electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 1]) * abs(x[-1])
    plt.plot(electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 0], temp, color='red', label='Electric Field')
    plt.legend()
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Position (a.u.)")
    plt.title("Density Matrix for psi_history[6]")
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.clim(0, 0.001)  # Set color limits for better visibility



    # plot an example of the density matrix
    data = compute_rho(hdf5_file["psi_history"][list(hdf5_file["psi_history"].keys())[6]])
    num_batch = 6
    plt.figure()
    plt.imshow(data.T, cmap='turbo', extent=( t[num_batch*data.shape[0]], t[(num_batch+1)*data.shape[0] -1] , x[0], x[-1]), aspect='auto')
    temp = electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 1] / np.max(electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 1]) * abs(x[-1])
    plt.plot(electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 0], temp, color='red', label='Electric Field')
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Position (a.u.)")
    plt.title("Density Matrix for psi_history[6]")
    plt.colorbar(label='Intensity')
    plt.tight_layout()

    plt.clim(0, 0.001)  # Set color limits for better visibility



def compute_dipole(hdf5_file):
    """
    Compute the dipole moment from the wavefunction data in the HDF5 file.
    
    :param hdf5_file: HDF5 file containing the wavefunction data
    the aim is to compute is while saving the memory of my computer
    :return: Dipole moment
    """
    parametres = hdf5_file["simulation_parameters"].attrs
    x = np.arange(parametres["x_start"], np.round(parametres["x_end"], 0), parametres["dx"])
    dx = parametres["dx"]
    t = np.arange(parametres["t_start"], np.round(parametres["t_end"], 0), parametres["dt"])

    dipole_retour = np.array([])
    for i in range(len(hdf5_file["psi_history"])):
        psi_history = hdf5_file["psi_history"][list(hdf5_file["psi_history"].keys())[i]]
        psi_fonda_history = hdf5_file["psi_fonda_history"][list(hdf5_file["psi_fonda_history"].keys())[i]]
        dipole_current_batch = np.sum(np.conj(psi_fonda_history) * x * psi_history, axis=1) * dx
        dipole_retour = np.append(dipole_retour, dipole_current_batch)
    return dipole_retour, t