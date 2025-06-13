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




# Ouverture du fichier
with h5py.File(r"C:\maxence_data_results\HHG_simulation\HHG_simulation_1.0000e-01_5.0000e-02_8.0000e+02_1.0000e+14.h5", "r") as f:
    plot_a_matrix(f["psi_fonda_history"], function_to_apply=compute_rho, title="Rho Matrix", xlabel="X-axis", ylabel="Y-axis")
    # fig.clim(0, 0.001)
    plt.clim(0, 0.001)  # Set color limits for better visibility
    plt.show()