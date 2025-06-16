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
    intensity = compute_intensity_from_efield(electric_field[:, 1], electric_field[:, 0])
    print(f"Intensity in W/m^2 : Mean = {intensity[0]:.2e}, Max = {intensity[1]:.2e}")

    # plot the very beginning of the simulation to see how it behaves
    data = compute_rho(hdf5_file["psi_history"]["psi_history_0"])
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
    if "psi_history_6" in hdf5_file["psi_history"]:
        data = compute_rho(hdf5_file["psi_history"]["psi_history_6"])
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
        psi_history = hdf5_file["psi_history"][sorted(list(hdf5_file["psi_history"].keys()), key = lambda x: int(x.split("_")[-1]))[i]]
        psi_fonda_history = hdf5_file["psi_fonda_history"][list(hdf5_file["psi_fonda_history"].keys())[i]]
        dipole_current_batch = np.sum(np.conj(psi_fonda_history) * x * psi_history, axis=1) * dx
        dipole_retour = np.append(dipole_retour, dipole_current_batch)
    return dipole_retour, t


def compute_intensity_from_efield(amplitudes, time, in_au=True):
    """
    Calculate the total intensity from an array of amplitudes over time.

    Parameters:
    amplitudes (array-like): Array of amplitude values.
    time (array-like): Array of time values corresponding to the amplitudes.

    Returns:
    float: The total intensity.
    """
    e = 1.602176634e-19         # elementary charge, C
    epsilon_0 = 8.8541878128e-12 # vacuum permittivity, F/m
    a0 = 5.29177210903e-11      # Bohr radius, m
    t_au = 2.4188843265857e-17    # atomic unit of time, s
    c = 299792458                # speed of light, m/s
    if in_au:
        # Convert time from atomic units to seconds
        time = time.copy() * t_au
        # Convert amplitudes from atomic units (a.u.) to SI units (V/m)
        # 1 a.u. of electric field = e / (4 * pi * epsilon_0 * a0^2)
        # TODO, verifier que c'est correct
        
        amplitudes = amplitudes.copy() * (e / (4 * np.pi * epsilon_0 * a0**2))

    # Calculate the instantaneous intensity using the full formula
    instantaneous_intensity = 0.5 * c * epsilon_0 * np.square(amplitudes)
    
    # Integrate the instantaneous intensity over time to get the total intensity
    total_intensity = np.trapezoid(instantaneous_intensity, time)
    
    return total_intensity/(time[-1]-time[0]), np.max(instantaneous_intensity)  # Return mean and max intensity
