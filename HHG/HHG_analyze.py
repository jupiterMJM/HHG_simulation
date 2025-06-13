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
t_au = 2.418884e-17  # s, constant for conversion to atomic units



# Ouverture du fichier
with h5py.File(r"C:\maxence_data_results\HHG_simulation\HHG_simulation_1.0000e-01_5.0000e-02_8.0000e+02_1.0000e+14.h5", "r") as f:
    parametres = f["simulation_parameters"].attrs
    plot_direct_info(f)

    # calcul du dipole
    print("Calcul du dipole...")
    dipole, t = compute_dipole(f)
    plt.figure()
    plt.plot(t, dipole)
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Dipole (a.u.)")
    plt.title("Dipole over time")
    plt.grid()
    plt.tight_layout()

    # calcul du champ électrique emis
    E_emis = np.gradient(np.gradient(dipole, ), parametres["dt"])
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
    apply_fft_on = np.pad(apply_fft_on, pad_width=(2**(15) - N)//2, mode='constant')  # Padding to the next power of 2 for FFT efficiency
    N = len(apply_fft_on)  # New length after padding
    frequencies = fftfreq(N, parametres["dt"])  # en a.u.^(-1)
    frequencies = frequencies / t_au
    # wv_plot = 3e8/frequencies
    spectrum = np.abs(fft(apply_fft_on))

    laser_IR_wv = parametres["wavelength"]  # in nm
    laser_IR_freq = 3e8 / (laser_IR_wv * 1e-9)  # Convert wavelength in nm to frequency in Hz

    # Affichage
    plt.figure()
    # plt.xscale('log')
    plt.yscale('log')
    plt.plot(frequencies[:N//2]/laser_IR_freq, spectrum[:N//2])
    plt.xlabel("Harmonic order (ω/ω₀)")
    plt.ylabel("Spectre harmonique")
    plt.title("Spectre des harmoniques générées")
    plt.grid()




plt.show()