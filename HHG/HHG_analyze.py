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
    parametres = f["simulation_parameters"].attrs
    x = np.arange(parametres["x_start"], parametres["x_end"], parametres["dx"])
    t = np.arange(parametres["t_start"], parametres["t_end"], parametres["dt"])
    champE = f["potentials_fields"]["champE"][()]
    potentiel_spatial = f["potentials_fields"]["potentiel_spatial"][()]

    
    


    plot_direct_info(f)


plt.show()