"""
ce fichier permet de resoudre TDSE en 1D pour la simulation de HHG
:auteur: Maxence BARRE
TODO: modifier l enregistrement via buffer pour utiliser hdf5 car c est plus rapide et plus efficace
TODO: peut etre sauvegarder en local (et pas de les envoyer sur le serveur)
"""


# importation des modules
import numpy as np
from annexe_HHG import *
from core_calculation import *
from tqdm import tqdm
import os
import h5py


#################################################################################
## CONFIGURATION
#################################################################################
# Space and time parameters
dx = 0.1                                         # in a.u. => should be small enough to resolve well the wave function => dx < 0.1 is a good value
x = np.arange(-120, 120, dx)                     # in a.u. => 1au=roughly 24as, this is the space grid for the simulation
dt = 0.05                                           # in a:u , if dt>0.05, we can t see electron that comes back to the nucleus
t = np.arange(-2000, 2000, dt)                      # also in a.u. => 1au=roughly 24as
N = len(x)
position_cap_abs = 80
epsilon = 0.0001                                    # small value to avoid division by zero in potential calculation
do_plot_at_end = True                               # if True, plot quite a lot of things at the end of the simulation
save_with_buffer = True                          # if True, save the wavefunction history every given timestep (very useful for huge and long simulations)
buffer_size = 10000                                  # used only if save_with_buffer is True, the size of the buffer to save the wavefunction history



# Laser Parameters
wavelength = 800                                    # nm, NOT IN A.U. the conversion is done later
I_wcm2 = 1e14                                       # Intensity in W/cm^2, NOT IN A.U. the conversion is done later


# Name of file to save the results
main_directory = "C:/maxence_data_results/HHG_simulation/"
file_psi = main_directory + f"psi_history_{dx:.4e}_{dt:.4e}_{wavelength:.4e}_{I_wcm2:.4e}.h5"  # File to save the wavefunction history
file_psi_fonda = main_directory + f"psi_fonda_history_{dx:.4e}_{dt:.4e}_{wavelength:.4e}_{I_wcm2:.4e}.h5"  # File to save the fundamental wavefunction history

# Initial wavefunction
psi_init = np.exp(-np.abs(x))                       # will be used both as initial wavefunction and as initial fondamental wavefunction
psi_init /= np.sqrt(np.sum(np.abs(psi_init)**2) * dx)  # Normalisation of the initial wavefunction

# constants (DO NOT CHANGE THESE VALUES)
c = 2.99792458e8                                    # m/s, speed of light
e = 1.602176634e-19                                 # C, elementary charge
m = 9.10938356e-31                                  # kg, electron mass
a0 = 5.291772108e-11                                # m, Bohr radius
hbar = 1.0545718e-34                                # J.s, reduced Planck's constant
t_au = 2.418884e-17                                 # s, constant for conversion to atomic units

# Affichage des parametres de la simulation
print("##################################################################################")
print("High Harmonic Generation Simulation Parameters")
print("##################################################################################")
print(f"[INFO] Simulation parameters:")
print(f"  - Space step (dx): {dx:.4e} a.u. ({len(x)} points)")
print(f"  - Time step (dt): {dt:.4e} a.u. ({len(t)} points)")
print(f"  - Wavelength: {wavelength:.4e} nm")
print(f"  - Intensity: {I_wcm2:.4e} W/cm^2")
print(f"  - Buffer size: {buffer_size} (if save_with_buffer is True)")
print(f"  - Save with buffer: {save_with_buffer}")
print(f"  - Do plot at end: {do_plot_at_end}")
print(f"  - File to save wavefunction history: {file_psi}")
print(f"  - File to save fundamental wavefunction history: {file_psi_fonda}")
print(f"Estimated time for the simulation (for 200it/s): {len(t) / 200:.2f} seconds")
print(f"Estimated memory required for the simulation: {2 * len(t) * len(x) * 16 / (1024 ** 2):.2f} MB")  # 2 for psi and psi_fonda, 16 bytes for complex128
##################################################################################



###############################################################################
## CONVERSION ET GENERATION DES POTENTIELS/LASER
###############################################################################
freq = 3e8 / (wavelength * 1e-9)                    # Frequency in Hz, converting nm to m
omega_au = 2*np.pi*freq*t_au
periode_au = 2*np.pi / omega_au                     # Period in atomic units
pulse_duration = 25 * periode_au                    # Pulse duration in atomic units
E0_laser = 5.338e-9 * np.sqrt(I_wcm2)

# Calcul de l'amplitude du laser en a.u.
champE_func = lambda x, t: E0_laser*np.cos(omega_au * t) * envelope(t, periode_au=periode_au)
champE = champE_func(x[:, None], t)                 # Champ électrique en fonction de x et t

# Calcul du potentiel atomique
potentiel_atomique = -1.0 / np.sqrt(x**2 + epsilon) +0j
potentiel_spatial = potentiel_atomique + potentiel_CAP(x, x_start=-abs(position_cap_abs), x_end=abs(position_cap_abs), eta_0=0.1)


# check that the files do not already exist
if os.path.exists(file_psi) or os.path.exists(file_psi_fonda):
    raise FileExistsError(f"Files {file_psi} or {file_psi_fonda} already exist. Please delete or rename them before running the simulation.")

# approximate memory required for the psi_history and psi_fonda_history arrays
bytes_per_element = np.dtype(np.complex128).itemsize  # 16 bytes
total_bytes = len(t) * len(x) * bytes_per_element
total_MB = 2*total_bytes / (1024 ** 2)
print(f"[INFO] Approximate memory required for a (2x{len(t)}x{len(x)}) complex128 matrix: {total_MB:.2f} MB")
if not save_with_buffer:
    if total_MB > 5000:  # if more than 5GB, raise a warning
        print("[WARNING] The simulation will require a lot of memory, and may crach depending on your machine's RAM. Proceed with caution.")
    if total_MB > 10000:  # if more than 10GB, raise an error
        raise MemoryError(f"[ERROR] The simulation requires more than 10GB of memory ({total_MB:.2f} MB). For sure, your machine will crash.")
###############################################################################



###############################################################################
## INITIALISATION DE LA SIMULATION
###############################################################################
if not save_with_buffer:
    psi_history = np.zeros((len(t), len(x)), dtype=np.complex128)  # Store wavefunction history
    psi_fonda_history = np.zeros((len(t), len(x)), dtype=np.complex128)  # Store initial wavefunction history
else:
    psi_history = np.zeros((buffer_size, len(x)), dtype=np.complex128)
    psi_fonda_history = np.zeros((buffer_size, len(x)), dtype=np.complex128)

psi = np.complex128(psi_init.copy())  # Initial wavefunction
psi_fonda = np.complex128(psi_init.copy())  # Initial fundamental wavefunction

psi_history[0] = psi.copy()  # Store initial wavefunction
psi_fonda_history[0] = psi_fonda.copy()  # Store initial fundamental wavefunction
###############################################################################



###############################################################################
## SIMULATION
###############################################################################
print("[INFO] Starting the simulation...")
buffer_number = 0
i = -1
for En in tqdm(champE):
    i += 1

    psi = evolve_crank_nikolson(psi, potentiel_spatial, En, dt, x)
    assert np.isclose(np.sum(np.abs(psi)**2) * dx, 1.0, atol=1e-6), f"Normalization condition not satisfied {np.sum(np.abs(psi)**2) * dx}"
    psi_history[i, :] = psi.copy()  # Store the wavefunction at this time step


    psi_fonda = evolve_crank_nikolson(psi_fonda, potentiel_spatial, 0, dt, x)       # Evolve the fundamental state without the laser field
    assert np.isclose(np.sum(np.abs(psi_fonda)**2) * dx, 1.0, atol=1e-6), f"Normalization condition not satisfied for fundamental state {np.sum(np.abs(psi_fonda)**2) * dx}"
    psi_fonda_history[i, :] = psi_fonda.copy()  # Store the fundamental state at this time step

    if save_with_buffer and (i + 1) % buffer_size == 0:
        print(f"[INFO] Saving buffer to files")
        # with open(file_psi, 'a') as file:
        #     for row in psi_history[:-1]:
        #         # Format each element in the row as a string and join them with spaces
        #         line = ' '.join([f"{elem.real}+{elem.imag}i" for elem in row]) + '\n'
        #         file.write(line)

        # with open(file_psi_fonda, 'a') as file_fonda:
        #     for row in psi_fonda_history[:-1]:
        #         # Format each element in the row as a string and join them with spaces
        #         line = ' '.join([f"{elem.real}+{elem.imag}i" for elem in row]) + '\n'
        #         file_fonda.write(line)

        with h5py.File(file_psi, 'a') as f:
            f.create_dataset(f'psi_history_{buffer_number}', data=psi_history)
        
        with h5py.File(file_psi_fonda, 'a') as f_fonda:
            f_fonda.create_dataset(f'psi_fonda_history_{buffer_number}', data=psi_fonda_history)
        buffer_number += 1

        # and reset the buffers
        psi_history = np.zeros((buffer_size, len(x)), dtype=np.complex128)
        psi_fonda_history = np.zeros((buffer_size, len(x)), dtype=np.complex128)

        psi_history[0] = psi.copy()  # Store initial wavefunction
        psi_fonda_history[0] = psi_fonda.copy()  # Store initial fundamental wavefunction

        i = -1
        print(f"[INFO] Buffer saved, arrays reset, continuing the simulation...")



print("[INFO] Simulation completed.")

print("[INFO] Calculating the density probability...")
rho_fonda_history = np.abs(psi_fonda_history)**2 * dx  # Density probability for the fundamental state
rho_history = np.abs(psi_history)**2 * dx  # Density probability for the wavefunction
##############################################################################


###############################################################################
## SAVING AND PLOTTING RESULTS
###############################################################################
print("[INFO] Saving results to files, DO NOT CLOSE THE PROGRAM UNTIL THIS IS DONE")
if save_with_buffer:
    with h5py.File(file_psi, 'a') as f:
        f.create_dataset(f'psi_history_{buffer_number}', data=psi_history)
    with h5py.File(file_psi_fonda, 'a') as f_fonda:
        f_fonda.create_dataset(f'psi_fonda_history_{buffer_number}', data=psi_fonda_history)
else:
    with h5py.File(file_psi, 'w') as f:
        f.create_dataset('psi_history', data=psi_history)
    with h5py.File(file_psi_fonda, 'w') as f_fonda:
        f_fonda.create_dataset('psi_fonda_history', data=psi_fonda_history)

if do_plot_at_end:
    print("[INFO] Plotting results...")

    plt.figure()
    plt.plot(t, champE)
    plt.plot(t, envelope(t, periode_au=periode_au) * E0_laser, 'r--', label='Envelope')
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Electric Field (a.u.)')
    plt.title('Electric Field of the Laser')
    plt.legend()


    plt.figure()
    plt.plot(x, np.imag(potentiel_spatial), label='Potential')
    plt.xlabel('Position (a.u.)')
    plt.ylabel('Potential (a.u.)')
    plt.title('Potential in the Simulation')
    plt.legend()

    plt.figure()
    plt.imshow(rho_fonda_history.T, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]], origin='lower', cmap="turbo")
    plt.clim(0, 0.001)
    plt.colorbar(label='Density')
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Position (a.u.)')
    plt.title('Density Probability Evolution')

    plt.figure()
    plt.imshow(rho_history.T, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]], origin='lower', cmap="turbo")
    plt.plot(t, champE/np.max(champE)*60, 'k--', color="red", label='Electric Field (normalized)', alpha=0.5)
    plt.clim(0, 0.001)
    plt.colorbar(label='Density')
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Position (a.u.)')
    plt.title('Density Probability Evolution')
    plt.show()

print("DONE, program finished successfully.")