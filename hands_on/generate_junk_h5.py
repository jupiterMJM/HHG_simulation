"""
the aim of this file is to generate a junk h5 file that can be used for testing purposes
we will therefore know for sure how the file is structured and what data it contains
:author: Maxence BARRE
:note: basically we will take the same configuration as the simulation 
"""

# importation des modules
import numpy as np
from tqdm import tqdm
import h5py
from scipy.constants import e, m_e, epsilon_0, c, h, hbar
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../HHG')))
from annexe_HHG import *
from core_calculation import *



#################################################################################
## CONFIGURATION
#################################################################################
# Space and time parameters
dx = 0.05                                         # in a.u. => should be small enough to resolve well the wave function => dx < 0.1 is a good value
x = np.arange(-100, 100, dx)                     # in a.u. => 1au=roughly 24as, this is the space grid for the simulation
dt = 0.05                                           # in a:u , if dt>0.05, we can t see electron that comes back to the nucleus
t = np.arange(-1000, 1000, dt)                      # also in a.u. => 1au=roughly 24as
N = len(x)
position_cap_abs = 80
epsilon = 0.00001                                    # small value to avoid division by zero in potential calculation
do_plot_at_end = True                               # if True, plot quite a lot of things at the end of the simulation
save_with_buffer = True                          # if True, save the wavefunction history every given timestep (very useful for huge and long simulations)
buffer_size_nb_point = 1e9                                  # used only if save_with_buffer is True, the number of points that will be in the buffer (prevents memory overflow when changing dx)
save_the_psi_history = False                     # if True, save the wavefunction history in a file (useful for debugging and analysis), but can take a fucking lot of memory
# if False, only 2 or 3 batches will be saved just for debugging purposes, and the rest will be discarded
charact_only_save_dipole = True          # if True, amongst the characteristics of the wavepacket, only the dipole will be saved (save space)



buffer_size = int(buffer_size_nb_point / (16*N))  # number of points in the buffer, used only if save_with_buffer is True
nb_buffer_full = int(len(t) / buffer_size)  # number of full buffers that will be saved, used only if save_with_buffer is True
print(buffer_size, nb_buffer_full)

# Laser Parameters
wavelength = 800                                    # nm, NOT IN A.U. the conversion is done later
I_wcm2 = 1e14                                       # Intensity in W/cm^2, NOT IN A.U. the conversion is done later


# Name of file to save the results
main_directory = "C:/maxence_data_results/HHG_simulation/"
# file_psi = main_directory + f"psi_history_{dx:.4e}_{dt:.4e}_{wavelength:.4e}_{I_wcm2:.4e}.h5"  # File to save the wavefunction history
# file_psi_fonda = main_directory + f"psi_fonda_history_{dx:.4e}_{dt:.4e}_{wavelength:.4e}_{I_wcm2:.4e}.h5"  # File to save the fundamental wavefunction history
file_output = main_directory + f"junk_file_{dx:.4e}_{dt:.4e}_{wavelength:.4e}_{I_wcm2:.4e}.h5"  # File to save all the results


###############################################################################
## CONVERSION ET GENERATION DES POTENTIELS/LASER
###############################################################################
t_au = 2.418884e-17                              # s, constant for conversion to atomic units
E_h = 4.35974e-18                                # J, energy in atomic units
a0 = 5.291772108e-11                             # m, Bohr radius
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

# Calcul de l'amplitude du laser en a.u.
champE_func = lambda x, t: E0_laser*np.cos(omega_au * t) * envelope(t, periode_au=periode_au)
champE = champE_func(x[:, None], t)                 # Champ électrique en fonction de x et t

# Calcul du potentiel atomique
potentiel_atomique = -1.0 / np.sqrt(x**2 + epsilon) +0j
potentiel_spatial = potentiel_atomique + potentiel_CAP(x, x_start=-abs(position_cap_abs), x_end=abs(position_cap_abs), eta_0=0.1)


# check that the files do not already exist
if os.path.exists(file_output):
    raise FileExistsError(f"Files {file_output} already exists. Please delete or rename them before running the simulation.")


###############################################################################


###############################################################################
## INITIALISATION DE LA SIMULATION/JUNK FILE
###############################################################################

psi_init = np.exp(-((x - 0)**2) / (10)) * np.exp(1j * np.random.uniform(0, 2 * np.pi, len(x)))  # Initial wavefunction, Gaussian centered at 0 with random phase
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
## afin de faciliter l'analyse de données, on va sauvegarder tous les parametres, potentiels et champs dans le même fichier hdf5
###############################################################################
with h5py.File(file_output, 'w') as f:
    f.create_group('simulation_parameters')
    f.create_group("potentials_fields")
    f.create_group("psi_history")
    f.create_group("psi_fonda_history")
    charac = f.create_group("wavepacket_characteristics")
    charac.create_group("dipole_on_fonda")
    charac.create_group("dipole_on_itself")
    if not charact_only_save_dipole:
        charac.create_group("momentum_on_fonda")
        charac.create_group("momentum_on_itself")
        charac.create_group("scalar_product_fonda")
        charac.create_group("kinetic_energy")
        charac.create_group("stdx_fonda")
        charac.create_group("stdx_itself")
        charac.create_group("stdp_fonda")
        charac.create_group("stdp_itself")


    # Save simulation parameters
    f['simulation_parameters'].attrs['dx'] = dx
    f['simulation_parameters'].attrs['x_start'] = x[0]
    f['simulation_parameters'].attrs['x_end'] = x[-1]
    f['simulation_parameters'].attrs['dt'] = dt
    f['simulation_parameters'].attrs['t_start'] = t[0]
    f['simulation_parameters'].attrs['t_end'] = t[-1]
    f['simulation_parameters'].attrs['wavelength'] = wavelength
    f['simulation_parameters'].attrs['I_wcm2'] = I_wcm2
    f["simulation_parameters"].attrs["epsilon"] = epsilon
    f["simulation_parameters"].attrs["position_cap_abs"] = position_cap_abs
    f["simulation_parameters"].attrs["save_the_psi_history"] = save_the_psi_history  # Save the flag for saving psi history
    f["psi_initial"] = psi_init  # Save the initial wavefunction


    # Save the potential and electric field
    f['potentials_fields'].create_dataset('potentiel_atomique', data=np.vstack((x, potentiel_atomique)).T)
    f['potentials_fields'].create_dataset('potentiel_spatial', data=np.vstack((x, potentiel_spatial)).T)
    f['potentials_fields'].create_dataset('champE', data=np.vstack((t, champE)).T)


###########################################################################################
## JUNK FILE GENERATION
###########################################################################################
print("[INFO] Starting the simulation...")
buffer_number = 0
i = -1
for En in tqdm(champE):
    i += 1

    # psi = racine(gaussienne centree autour de En, de variance En/4)
    psi = np.sqrt(np.exp(-((x - 1000*En)**2) / (abs(1000*En) / 4))) * np.exp(1j * np.random.uniform(0, 2 * np.pi, len(x)))  # Random phase for each point
    psi /= np.linalg.norm(psi)  # Normalize the wavefunction
    psi_history[i, :] = psi.copy()  # Store the wavefunction at this time step

    psi_fonda = np.exp(-abs(x)) * np.exp(1j * np.random.uniform(0, 2 * np.pi, len(x)))  # Random phase for the fundamental wavefunction
    psi_fonda_history[i, :] = psi_fonda.copy()  # Store the fundamental state at this time step

    if save_with_buffer and (i + 1) % buffer_size == 0:
        # we do not save the buffer BUT we empty it
        if save_the_psi_history or buffer_number in (0, 1, 6):  # we do not save the buffer unless a few of them to debug
            print(f"[INFO] Saving buffer to files")


            with h5py.File(file_output, 'a') as f:
                f["psi_history"].create_dataset(f'psi_history_{buffer_number}', data=psi_history)
                f["psi_fonda_history"].create_dataset(f'psi_fonda_history_{buffer_number}', data=psi_fonda_history)

        if not save_the_psi_history:        # if we save psi_history, all computation can be done later
            # now we don t save the psi itself anymore, we could just save only the caracteristics of the wavepackets, such as: the dipole (projected onto psi_fonda and psi itsel), the momentum (same)
            print("[INFO] Computing characteristics of the wavepacket, might take some time...")
            dipole_on_fonda, dipole_on_itself, momentum_on_fonda, momentum_on_itself, scalar_product_fonda, kinetic_energy, stdx_fonda, stdx_itself, stdp_fonda, stdp_itself = calculate_caracterisitcs_wavepacket(psi_history, x, psi_fonda_history)
            with h5py.File(file_output, 'a') as f:
                f["wavepacket_characteristics/dipole_on_fonda"].create_dataset(f'dipole_on_fonda_{buffer_number}', data=dipole_on_fonda)
                f["wavepacket_characteristics/dipole_on_itself"].create_dataset(f'dipole_on_itself_{buffer_number}', data=dipole_on_itself)
                if not charact_only_save_dipole:
                    f["wavepacket_characteristics/momentum_on_fonda"].create_dataset(f'momentum_on_fonda_{buffer_number}', data=momentum_on_fonda)
                    f["wavepacket_characteristics/momentum_on_itself"].create_dataset(f'momentum_on_itself_{buffer_number}', data=momentum_on_itself)
                    f["wavepacket_characteristics/scalar_product_fonda"].create_dataset(f'scalar_product_fonda_{buffer_number}', data=scalar_product_fonda)
                    f["wavepacket_characteristics/kinetic_energy"].create_dataset(f'kinetic_energy_{buffer_number}', data=kinetic_energy)
                    f["wavepacket_characteristics/stdx_fonda"].create_dataset(f'stdx_fonda_{buffer_number}', data=stdx_fonda)
                    f["wavepacket_characteristics/stdx_itself"].create_dataset(f'stdx_itself_{buffer_number}', data=stdx_itself)
                    f["wavepacket_characteristics/stdp_fonda"].create_dataset(f'stdp_fonda_{buffer_number}', data=stdp_fonda)
                    f["wavepacket_characteristics/stdp_itself"].create_dataset(f'stdp_itself_{buffer_number}', data=stdp_itself)

        buffer_number += 1


        if buffer_number < nb_buffer_full:
            # and reset the buffers
            psi_history = np.zeros((buffer_size, len(x)), dtype=np.complex128)
            psi_fonda_history = np.zeros((buffer_size, len(x)), dtype=np.complex128)
        else: # this is the last buffer, so we do not take a full matrix buffer
            print("HEY", len(t) - nb_buffer_full*buffer_size)
            psi_history = np.zeros((len(t) - nb_buffer_full*buffer_size, len(x)), dtype=np.complex128)  # Store wavefunction history
            psi_fonda_history = np.zeros((len(t) - nb_buffer_full*buffer_size, len(x)), dtype=np.complex128)

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
    # with h5py.File(file_psi, 'a') as f:
    #     f.create_dataset(f'psi_history_{buffer_number}', data=psi_history)
    # with h5py.File(file_psi_fonda, 'a') as f_fonda:
    #     f_fonda.create_dataset(f'psi_fonda_history_{buffer_number}', data=psi_fonda_history)

    # for now we save the last buffer but we may have to remove it
    with h5py.File(file_output, 'a') as f:
        f["psi_history"].create_dataset(f'psi_history_{buffer_number}', data=psi_history)
        f["psi_fonda_history"].create_dataset(f'psi_fonda_history_{buffer_number}', data=psi_fonda_history)

    if not save_the_psi_history:        # if we save psi_history, all computation can be done later
            # now we don t save the psi itself anymore, we could just save only the caracteristics of the wavepackets, such as: the dipole (projected onto psi_fonda and psi itsel), the momentum (same)
            print("[INFO] Computing characteristics of the wavepacket, might take some time...")
            dipole_on_fonda, dipole_on_itself, momentum_on_fonda, momentum_on_itself, scalar_product_fonda, kinetic_energy, stdx_fonda, stdx_itself, stdp_fonda, stdp_itself = calculate_caracterisitcs_wavepacket(psi_history, x, psi_fonda_history)
            with h5py.File(file_output, 'a') as f:
                f["wavepacket_characteristics/dipole_on_fonda"].create_dataset(f'dipole_on_fonda_{buffer_number}', data=dipole_on_fonda)
                f["wavepacket_characteristics/dipole_on_itself"].create_dataset(f'dipole_on_itself_{buffer_number}', data=dipole_on_itself)
                if not charact_only_save_dipole:
                    f["wavepacket_characteristics/momentum_on_fonda"].create_dataset(f'momentum_on_fonda_{buffer_number}', data=momentum_on_fonda)
                    f["wavepacket_characteristics/momentum_on_itself"].create_dataset(f'momentum_on_itself_{buffer_number}', data=momentum_on_itself)
                    f["wavepacket_characteristics/scalar_product_fonda"].create_dataset(f'scalar_product_fonda_{buffer_number}', data=scalar_product_fonda)
                    f["wavepacket_characteristics/kinetic_energy"].create_dataset(f'kinetic_energy_{buffer_number}', data=kinetic_energy)
                    f["wavepacket_characteristics/stdx_fonda"].create_dataset(f'stdx_fonda_{buffer_number}', data=stdx_fonda)
                    f["wavepacket_characteristics/stdx_itself"].create_dataset(f'stdx_itself_{buffer_number}', data=stdx_itself)
                    f["wavepacket_characteristics/stdp_fonda"].create_dataset(f'stdp_fonda_{buffer_number}', data=stdp_fonda)
                    f["wavepacket_characteristics/stdp_itself"].create_dataset(f'stdp_itself_{buffer_number}', data=stdp_itself)
else:
    # with h5py.File(file_psi, 'w') as f:
    #     f.create_dataset('psi_history', data=psi_history)
    # with h5py.File(file_psi_fonda, 'w') as f_fonda:
    #     f_fonda.create_dataset('psi_fonda_history', data=psi_fonda_history)
    with h5py.File(file_output, 'w') as f:
        f["psi_history"].create_dataset('psi_history', data=psi_history)
        f["psi_fonda_history"].create_dataset('psi_fonda_history', data=psi_fonda_history)


print("JUNK FILE GENERATED SUCCESSFULLY")