"""
le fichier HHG simulation a pour but de simuler uniquement les données de HHG sans les analyser (calcul du dipole, de rho...)
plusieurs choses sont encore à ameliorer mais cela avance bien
Ce fichier a pour but d'analyser les resultats de la simulation HHG
:auteur: Maxence BARRE
"""

from annexe_analyze import *
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.constants import e, m_e, epsilon_0, c, h
t_au = 2.418884e-17  # s, constant for conversion to atomic units



# Ouverture du fichier
with h5py.File(r"C:\maxence_data_results\HHG_simulation\HHG_simulation_pulse_5.0000e-02_5.0000e-02_8.0000e+02_1.0000e+14.h5", "r") as f:
    parametres = f["simulation_parameters"].attrs
    I_wcm2 = parametres["I_wcm2"]  # in W/cm^2
    plot_direct_info(f)

    # calcul du dipole
    print("Calcul du dipole...")
    if "save_the_psi_history" not in parametres:
        print("No saved wavefunctions found. Computing dipole from wavepacket characteristics.")
        dipole, t = compute_dipole(f)
    elif parametres["save_the_psi_history"]:
        print("Using saved wavefunctions to compute the dipole.")
        dipole, t = compute_dipole(f)
    else:
        dipole = np.array([])
        for batch in sorted(f["wavepacket_characteristics/dipole_on_fonda"], key=lambda x: int(x.split('_')[-1])):
            # print("HELLLLLLO", f[f"wavepacket_characteristics/dipole_on_fonda/{batch}"])
            dipole = np.append(dipole, np.array(f[f"wavepacket_characteristics/dipole_on_fonda/{batch}"]))
        t = np.linspace(0, len(dipole) * parametres["dt"], len(dipole))
    plt.figure()
    plt.plot(t, dipole)
    # plt.plot(t, f["potentials_fields/champE"][:, 1][np.logical_and(f["potentials_fields/champE"][:, 0] > t[0], f["potentials_fields/champE"][:, 0] < t[-1])], label="Electric Field")
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Dipole (a.u.)")
    plt.title("Dipole over time")
    plt.grid()
    plt.tight_layout()

    # calcul du champ électrique emis
    print(dipole)
    E_emis = np.gradient(np.gradient(dipole, parametres["dt"]), parametres["dt"])
    plt.figure()
    plt.plot(t, E_emis)
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Electric Field (a.u.)")
    plt.title("Electric Field Emitted over time")
    plt.grid()
    plt.tight_layout()


    # calcul du spectre
    apply_fft_on = dipole.real[np.logical_and(t > 1200, True)]  # Keep only the central part of the emission
    N = len(apply_fft_on)
    apply_fft_on = np.pad(apply_fft_on, pad_width=(2**(17) - N)//2, mode='constant')  # Padding to the next power of 2 for FFT efficiency
    N = len(apply_fft_on)  # New length after padding
    frequencies = fftfreq(N, parametres["dt"])  # en a.u.^(-1)
    frequencies = frequencies / t_au
    # wv_plot = 3e8/frequencies
    spectrum = np.abs(fft(apply_fft_on))

    laser_IR_wv = parametres["wavelength"]  # in nm
    laser_IR_freq = 3e8 / (laser_IR_wv * 1e-9)  # Convert wavelength in nm to frequency in Hz


    # calculation of energu of the cut-off
    I_p = 13.6  # Ionization potential in eV
    U_p = e**2/(8*m_e*epsilon_0*c**3 * np.pi**2) * (I_wcm2 * 1e4) * ((laser_IR_wv*1e-9)**2)/e  # in eV, ponderomotive energy
    print(I_p, U_p,I_p + 3.17 * U_p)
    freq_cutoff = 1/h * (I_p + 3.17 * U_p)*e  # in Hz, cut-off frequency
    # Affichage
    plt.figure()
    # plt.xscale('log')
    plt.yscale('log')
    plt.plot(frequencies[:N//2]/laser_IR_freq, spectrum[:N//2])
    plt.axvline(x=1, color='r', linestyle='--', label='Fundamental Frequency (ω₀)')
    plt.axvline(x=freq_cutoff/laser_IR_freq, color='g', linestyle='--', label='Cut-off Frequency (ω_cutoff)')
    plt.xlabel("Harmonic order (ω/ω₀)")
    plt.ylabel("Spectre harmonique")
    plt.title("Spectre des harmoniques générées")
    plt.grid()




plt.show()