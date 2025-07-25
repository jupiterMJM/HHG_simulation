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
from scipy.signal.windows import hamming, blackmanharris
from scipy.signal import ShortTimeFFT
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add
from classical_view.generate_mask_classical import *

t_au = 2.418884e-17  # s, constant for conversion to atomic units

# for plateau: HHG_1064nm_1e14_1.0000e-02_5.0000e-02_1.0640e+03_1.0000e+14.h5 (mais pas tres beau)
# HHG_article1990_biggertimerange_psiinitfunc_2.5000e-01_1.8375e-01_1.0640e+03_1.0000e+14.h5
# Ouverture du fichier
with h5py.File(r"C:\maxence_data_results\HHG_simulation\HHG_new_initialwavefunction_gridincreased_2.5000e-01_1.3816e-01_1.0640e+03_1.0000e+14.h5", "r") as f:
    parametres = f["simulation_parameters"].attrs
    I_wcm2 = parametres["I_wcm2"]  # in W/cm^2
    plot_direct_info(f, plot_classical_on_top_of_rho=False)

    # plt.show()
    # calcul du dipole
    print("Calcul du dipole...")
    dipole_on_what = "itself"
    remove_all_cap = True
    if "save_the_psi_history" not in parametres:
        print("No saved wavefunctions found. Computing dipole from wavepacket characteristics.")
        dipole, t = compute_dipole(f, dipole_on_what=dipole_on_what, remove_all_cap=remove_all_cap, plot_not_summed_dipole=True)
    elif parametres["save_the_psi_history"]:
        print("Using saved wavefunctions to compute the dipole.")
        dipole, t = compute_dipole(f, dipole_on_what=dipole_on_what, remove_all_cap=remove_all_cap, plot_not_summed_dipole=True)
    else:
        dipole = np.array([])

        
        assert dipole_on_what in ("fonda", "itself"), "dipole_on_what must be either 'fonda' or 'itself'"
        for batch in sorted(f[f"wavepacket_characteristics/dipole_on_{dipole_on_what}"], key=lambda x: int(x.split('_')[-1])):
            # print("HELLLLLLO", f[f"wavepacket_characteristics/dipole_on_fonda/{batch}"])
            dipole = np.append(dipole, np.array(f[f"wavepacket_characteristics/dipole_on_{dipole_on_what}/{batch}"]))

        momentum = None
        if "momentum_on_itself" in f["wavepacket_characteristics"]:
            momentum = np.array([])
            for batch in sorted(f["wavepacket_characteristics/momentum_on_itself"], key=lambda x: int(x.split('_')[-1])):
                momentum = np.append(momentum, np.array(f[f"wavepacket_characteristics/momentum_on_itself/{batch}"]))


        t = f["potentials_fields/champE"][:, 0]     # more explicit
        if len(t) > len(dipole):
            t = t[:len(dipole)]

        if momentum is not None and len(t) > len(momentum):
            plt.figure()
            plt.plot(t, momentum, label="Momentum")
            e_field_to_plot = f["potentials_fields/champE"][:, 1][np.logical_and(f["potentials_fields/champE"][:, 0] >= t[0], f["potentials_fields/champE"][:, 0] <= t[-1])]
            plt.plot(t, e_field_to_plot/np.max(e_field_to_plot), label="Electric Field")
            plt.xlabel("Time (a.u.)")
            plt.ylabel("Momentum (a.u.)")
            plt.title("Momentum over time")
            plt.grid()
            plt.tight_layout()
            
        
        
    fig_dipole = plt.figure()
    print(f"[INFO] Dipole on {dipole_on_what} shape: {dipole.shape}, t shape: {t.shape}")
    plt.plot(t, dipole, label="Dipole")
    e_field_to_plot = f["potentials_fields/champE"][:, 1][np.logical_and(f["potentials_fields/champE"][:, 0] >= t[0], f["potentials_fields/champE"][:, 0] <= t[-1])]
    plt.plot(t, e_field_to_plot/np.max(e_field_to_plot) * 0.4, label="Electric Field", c="red", alpha=0.5)
    plt.xlabel("Time (a.u.)")
    plt.ylabel(f"Dipole on {dipole_on_what}(a.u.)")
    plt.title("Dipole over time")
    plt.grid()
    plt.legend(loc="upper right")
    plt.tight_layout()

    # calcul du champ électrique emis
    print(dipole)
    E_emis = np.gradient(np.gradient(dipole.real, parametres["dt"]*t_au), parametres["dt"]*t_au)
    plt.figure()
    plt.plot(t*t_au, E_emis)
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Electric Field (a.u.)")
    plt.title("Electric Field Emitted over time")
    plt.plot(t*t_au, e_field_to_plot/np.max(e_field_to_plot) * np.max(E_emis), label="Electric Field (from file)", c="red", alpha=0.5)
    plt.grid()
    plt.tight_layout()


    # calcul du spectre
    # apply_fft_on = dipole.real[np.logical_and(t > 1200, True)]  # Keep only the central part of the emission
    t_borne_inf = -250  # in a.u.
    t_borne_sup = 250  # in a.u.
    apply_fft_on = dipole.real[np.logical_and(t>=t_borne_inf, t<=t_borne_sup)]  # Use the entire dipole for FFT
    if apply_fft_on.size == 0:
        raise ValueError("No data available for FFT. Please check the dipole data. ( probably error of mask when choosing the time range )")
    N = len(apply_fft_on)
    apply_fft_on = blackmanharris(len(apply_fft_on)) * apply_fft_on  # Apply a Blackman-Harris window to reduce spectral leakage
    apply_fft_on = np.pad(apply_fft_on, pad_width=(2**(17) - N)//2, mode='constant')  # Padding to the next power of 2 for FFT efficiency
    N = len(apply_fft_on)  # New length after padding
    frequencies = fftfreq(N, parametres["dt"])  # en a.u.^(-1)
    frequencies = frequencies / t_au
    # wv_plot = 3e8/frequencies
    spectrum = fft(apply_fft_on)

    laser_IR_wv = parametres["wavelength"]  # in nm
    laser_IR_freq = 3e8 / (laser_IR_wv * 1e-9)  # Convert wavelength in nm to frequency in Hz

    # Plot vertical dashed grey lines at t_borne_inf and t_borne_sup on fig_dipole
    plt.figure(fig_dipole.number)
    plt.axvspan(t[0], t_borne_inf, color='grey', alpha=0.2)
    plt.axvspan(t_borne_sup, t[-1], color='grey', alpha=0.2)
    plt.axvline(x=t_borne_inf, color='grey', linestyle='--')
    plt.axvline(x=t_borne_sup, color='grey', linestyle='--')


    # calculation of energy of the cut-off
    I_p = 13.6  # Ionization potential in eV
    U_p = e**2/(8*m_e*epsilon_0*c**3 * np.pi**2) * (I_wcm2 * 1e4) * ((laser_IR_wv*1e-9)**2)/e  # in eV, ponderomotive energy
    print(I_p, U_p,I_p + 3.17 * U_p)
    freq_cutoff = 1/h * (I_p + 3.17 * U_p)*e  # in Hz, cut-off frequency
    # Affichage
    plt.figure()
    # plt.xscale('log')
    spectrum = spectrum[frequencies > 0]  # Keep only positive frequencies
    frequencies = frequencies[frequencies > 0]  # Keep only positive frequencies
    
    plt.plot(frequencies/laser_IR_freq, np.abs(spectrum)**2, label="Dipole Spectrum")
    plt.axvline(x=1, color='r', linestyle='--', label='Fundamental Frequency')
    plt.axvline(x=freq_cutoff/laser_IR_freq, color='g', linestyle='--', label='Theoretical Cut-off Frequency')
    plt.xlabel("Harmonic order (ω/ω₀)")
    plt.ylabel("Spectre harmonique")
    plt.title("Spectre des harmoniques générées")
    # plt.yscale('log')
    plt.legend()
    plt.grid()


    # # on essaye d'identifier les trajectoires longues et courtes
    # print("[INFO] Computing Gabor Transform...")
    # # retour, omegas = gabor_transform(dipole, t, 10, (0, 1000), 1)
    # win = gaussian(100, std=25, sym = True)
    # SFT = ShortTimeFFT(win, fs=1/(parametres["dt"]*t_au), hop=10, mfft=2**10)
    # Sx2 = SFT.spectrogram(dipole.real)
    # print(Sx2.shape)
    # plt.figure()
    # plt.imshow(np.abs(Sx2), aspect='auto', extent=SFT.extent(len(dipole.real)), origin='lower', cmap='turbo')
    # plt.colorbar(label='Amplitude')
    # print(list(SFT.extent(len(dipole.real))))
    # print("Frequencies:", SFT.f, "delta_f:", SFT.delta_f)

    # plt.figure()
    # plt.plot(t, np.unwrap(np.angle(dipole)))

    apply_fft_on_field = E_emis[np.logical_and(t>=t_borne_inf, t<=t_borne_sup)]  # Use the entire dipole for FFT
    if apply_fft_on_field.size == 0:
        raise ValueError("No data available for FFT. Please check the dipole data. ( probably error of mask when choosing the time range )")
    N = len(apply_fft_on_field)
    apply_fft_on_field = hamming(len(apply_fft_on_field)) * apply_fft_on_field  # Apply a Blackman-Harris window to reduce spectral leakage
    apply_fft_on_field = np.pad(apply_fft_on_field, pad_width=(2**(17) - N)//2, mode='constant')  # Padding to the next power of 2 for FFT efficiency
    N = len(apply_fft_on_field)  # New length after padding
    frequencies_field = fftfreq(N, parametres["dt"])  # en a.u.^(-1)
    frequencies_field = frequencies_field / t_au
    # wv_plot = 3e8/frequencies
    spectrum_field = fft(apply_fft_on_field)
    spectrum_field = spectrum_field[frequencies_field > 0]  # Keep only positive frequencies
    frequencies_field = frequencies_field[frequencies_field > 0]  # Keep only positive frequencies

    plt.plot(frequencies_field/laser_IR_freq, np.abs(spectrum_field)**2, label="Electric Field Spectrum")
    # plt.axvline(x=1, color='r', linestyle='--', label='Fundamental Frequency')
    # plt.axvline(x=freq_cutoff/laser_IR_freq, color='g', linestyle='--', label='Theoretical Cut-off Frequency')
    plt.xlabel("Harmonic order (ω/ω₀)")
    plt.ylabel("Spectre harmonique")
    plt.title("Spectre des harmoniques générées")
    # plt.yscale('log')
    plt.legend()
    plt.grid()

    omega = 2 * np.pi * frequencies  # Convert frequencies to angular frequencies
    expected_field_spectrum = 1j * omega * 1j * omega * spectrum
    plt.plot(frequencies/laser_IR_freq, np.abs(expected_field_spectrum)**2, label="Expected Field Spectrum (from dipole)")
    plt.xlabel("Harmonic order (ω/ω₀)")
    plt.ylabel("Spectre harmonique")
    plt.title("Expected Electric Field Spectrum from Dipole")
    plt.yscale('log')
    plt.legend()
    plt.grid()




plt.show()