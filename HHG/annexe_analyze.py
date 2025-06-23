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
    x = np.arange(parametres["x_start"], round(parametres["x_end"], 0), parametres["dx"])
    print("YOIIIIIIIIIIIIIIII", parametres["x_start"], int(parametres["x_end"]), parametres["dx"])
    print("YAAAAAAAAAAAAAAA", x.shape, x[0], x[-1], x[1]-x[0])
    t = np.arange(parametres["t_start"], round(parametres["t_end"], 0), parametres["dt"])

    print(f"[INFO] Simulation parameters:")
    print(f"  - Space step (dx): {parametres["dx"]:.4e} a.u. ({len(x)} points)")
    print(f"  - Time step (dt): {parametres["dt"]:.4e} a.u. ({len(t)} points)")
    print(f"  - Wavelength: {parametres["wavelength"]:.4e} nm")
    print(f"  - Intensity: {parametres["I_wcm2"]:.4e} W/cm^2")

    wavelength = parametres["wavelength"]  # in nm
    t_au = 2.418884e-17  # s, atomic unit of time
    I_wcm2 = parametres["I_wcm2"]  # in W
    epsilon_0 = 8.8541878128e-12  # vacuum permittivity in F/m
    E_h = 4.35974e-18  # Hartree energy in J
    e = 1.602176634e-19  # elementary charge in C
    a0 = 5.29177210903e-11  # Bohr radius in m
    c = 299792458  # speed of light in m/s
    freq = 3e8 / (wavelength * 1e-9)                    # Frequency in Hz, converting nm to m
    omega_au = 2*np.pi*freq*t_au
    periode_au = 2*np.pi / omega_au                     # Period in atomic units
    pulse_duration = 25 * periode_au                    # Pulse duration in atomic units
    E0_laser = np.sqrt(2/(epsilon_0*c)) * np.sqrt(I_wcm2*1e4) / (E_h/(e*a0))       # from intensity to eletric field in a.u.
    print(f"[INFO] Laser parameters:"
        f"\n  - Frequency: {freq:.2e} Hz"
        f"\n  - Angular frequency (omega_au): {omega_au:.2e} a.u."
        f"\n  - Period (periode_au): {periode_au:.2e} a.u."
        f"\n  - Pulse duration: {pulse_duration:.2e} a.u."
        f"\n  - Electric field amplitude (E0_laser): {E0_laser:.2e} a.u.")

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
    plt.imshow(data, cmap='turbo', extent=( x[0], x[-1], t[(num_batch+1)*data.shape[0] -1],t[num_batch*data.shape[0]]), aspect='auto')
    # plt.imshow(data.T, cmap='turbo', extent=( t[num_batch*data.shape[0]], t[(num_batch+1)*data.shape[0] -1] , x[-1], x[0]), aspect='auto')
    temp = electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 1] / np.max(electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 1]) * abs(x[-1])
    plt.plot(temp, electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 0], color='red', label='Electric Field')
    # plt.plot(electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 0], temp, color='red', label='Electric Field')
    plt.legend()
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Position (a.u.)")
    plt.title("Density Matrix for psi_history[0]")
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.clim(0, 0.001)  # Set color limits for better visibility



    # plot an example of the density matrix
    if "psi_history_6" in hdf5_file["psi_history"] or "psi_history_1" in hdf5_file["psi_history"]:
        
        num_batch = 6 if "psi_history_6" in hdf5_file["psi_history"] else 1
        data = compute_rho(hdf5_file["psi_history"]["psi_history_" + str(num_batch)])
        # plt.figure()
        # plt.imshow(data)
        # plt.clim(0, 0.0005)  # Set color limits for better visibility
        plt.figure()
        plt.imshow(data, cmap='turbo', extent=( x[0], x[-1], t[(num_batch+1)*data.shape[0] -1],t[num_batch*data.shape[0]]), aspect='auto')
        # plt.imshow(data.T, cmap='turbo', extent=( t[num_batch*data.shape[0]], t[(num_batch+1)*data.shape[0] -1] , x[-1], x[0]), aspect='auto')
        temp = electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 1] / np.max(electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 1]) * abs(x[-1])
        # plt.plot(electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 0], temp, color='red', label='Electric Field')
        plt.plot(temp, electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 0], color='red', label='Electric Field')
        plt.ylabel("Time (a.u.)")
        plt.xlabel("Position (a.u.)")
        plt.title("Density Matrix for psi_history[6]")
        plt.colorbar(label='Intensity')
        plt.tight_layout()

        plt.clim(0, 0.001)  # Set color limits for better visibility


        plt.figure()
        plt.imshow(compute_rho(hdf5_file["psi_fonda_history"]["psi_fonda_history_" + str(num_batch)]), cmap='turbo', extent=( x[0], x[-1], t[(num_batch+1)*data.shape[0] -1],t[num_batch*data.shape[0]]), aspect='auto')
        plt.plot(temp, electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 0], color='red', label='Electric Field')
        plt.ylabel("Time (a.u.)")
        plt.xlabel("Position (a.u.)")
        plt.title(f"Fundamental Wavefunction for psi_fonda_history[{num_batch}]")
        plt.colorbar(label='Intensity')
        plt.tight_layout()
        plt.clim(0, 0.01)
        


        psi_history = hdf5_file["psi_history"]["psi_history_" + str(num_batch)]
        psi_fonda_history = hdf5_file["psi_fonda_history"]["psi_fonda_history_" + str(num_batch)]
        print("YOOOOOOOOOO", x.shape, x[0], x[-1], x[1]-x[0])
        dipole_not_summed = np.conj(psi_fonda_history) * x * psi_history * parametres["dx"]
        plt.figure()
        plt.imshow(np.abs(dipole_not_summed), cmap='turbo', extent=( x[0], x[-1], t[(num_batch+1)*data.shape[0] -1],t[num_batch*data.shape[0]]), aspect='auto')
        plt.plot(temp, electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 0], color='red', label='Electric Field')
        plt.ylabel("Time (a.u.)")
        plt.xlabel("Position (a.u.)")
        plt.title(f"Dipole Moment (not summed) for psi_history[{num_batch}]")
        plt.colorbar(label='Dipole Moment (a.u.)')
        plt.tight_layout()
        plt.clim(0, 0.01)

    # elif "psi_history_1" in hdf5_file["psi_history"]:
    #     data = compute_rho(hdf5_file["psi_history"]["psi_history_1"])
    #     num_batch = 1
    #     plt.figure()
    #     plt.imshow(data, cmap='turbo', extent=( x[0], x[-1], t[(num_batch+1)*data.shape[0] -1],t[num_batch*data.shape[0]]), aspect='auto')
    #     # plt.imshow(data.T, cmap='turbo', extent=( t[num_batch*data.shape[0]], t[(num_batch+1)*data.shape[0] -1] , x[-1], x[0]), aspect='auto')
    #     temp = electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 1] / np.max(electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 1]) * abs(x[-1])
    #     # plt.plot(electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 0], temp, color='red', label='Electric Field')
    #     plt.plot(temp, electric_field[num_batch*data.shape[0]:(num_batch+1)*data.shape[0]-1][:, 0], color='red', label='Electric Field')
    #     plt.xlabel("Time (a.u.)")
    #     plt.ylabel("Position (a.u.)")
    #     plt.title("Density Matrix for psi_history[1]")
    #     plt.colorbar(label='Intensity')
    #     plt.tight_layout()

    #     plt.clim(0, 0.001)  # Set color limits for better visibility



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



def gabor_transform(a_t, t, sigma, omega_range, omega_steps):
    """
    Compute the Gabor transform of a signal a_t at times t with a Gaussian window.
    basically, it computes the Fourier transform of the signal a_t multiplied by a Gaussian window centered at each time t0.
    :note: the aim is to look at the evolution of the frequency-component of the signal a_t over time t.
    :param: a_t: signal to analyze, numpy array of shape (T,)
    :param: t: time vector, numpy array of shape (T,)
    :param: sigma: standard deviation of the Gaussian window, float
    :param: omega_range: tuple (min_omega, max_omega) defining the frequency range for the Gabor transform
    :param: omega_steps: number of frequency steps in the range, int
    :return: Gabor transform G, numpy array of shape (omega_steps, T)"""
    dt = t[1] - t[0]
    T = len(t)

    # Vecteurs temps et fréquences
    t = np.array(t)
    omegas = np.linspace(*omega_range, omega_steps)

    # Création de matrices t0 (centre de fenêtre) et t (temps réel)
    t0 = t[:, np.newaxis]      # shape (T, 1)
    t_mat = t[np.newaxis, :]   # shape (1, T)

    # Matrice des fenêtres gaussiennes centrées en t0
    window = np.exp(-((t_mat - t0)**2) / (2 * sigma**2))  # shape (T, T)

    # Signal multiplié par les fenêtres (shape (T, T))
    a_windowed = window * a_t[np.newaxis, :]

    # Calcul de la transformée de Gabor pour chaque fréquence (broadcasting)
    # Résultat final : shape (omega_steps, T)
    exp_matrix = np.exp(-1j * np.outer(omegas, t))  # shape (omega_steps, T)
    G = exp_matrix @ a_windowed.T * dt              # produit matriciel

    return G, omegas