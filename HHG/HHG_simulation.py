"""
ce fichier permet de resoudre TDSE en 1D pour la simulation de HHG
:auteur: Maxence BARRE
TODO: modifier pr prendre en charge des simulations beaucoup plus longues (enregistrement au fur et à mesure de la simulation)
"""


# importation des modules
import numpy as np
from annexe_HHG import *
from core_calculation import *


#################################################################################
## CONFIGURATION
#################################################################################
# Space and time parameters
x = np.linspace(-100, 100, 1024)                    # in a.u. => in bohr units
dx = x[1] - x[0]                                    # in a.u. => should be well inferior to 1
dt = 0.05                                           # in a:u , if dt>0.05, we can t see electron that comes back to the nucleus
t = np.arange(-2000, 2000, dt)                      # also in a.u. => 1au=roughly 24as
N = len(x)
position_cap_abs = 75
epsilon = 0.0001                                    # small value to avoid division by zero in potential calculation


# Laser Parameters
wavelength = 800                                    # nm, NOT IN A.U. the conversion is done later
I_wcm2 = 1e14                                       # Intensity in W/cm^2, NOT IN A.U. the conversion is done later


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
###############################################################################



###############################################################################
## INITIALISATION DE LA SIMULATION
###############################################################################
psi_history = np.zeros((len(t), len(x)), dtype=np.complex128)  # Store wavefunction history
psi_fonda_history = np.zeros((len(t), len(x)), dtype=np.complex128)  # Store initial wavefunction history

psi = psi_init.copy()  # Initial wavefunction
psi_fonda = psi_init.copy()  # Initial fundamental wavefunction

psi_history[0] = psi.copy()  # Store initial wavefunction
psi_fonda_history[0] = psi_fonda.copy()  # Store initial fundamental wavefunction
###############################################################################



###############################################################################
## SIMULATION
###############################################################################
i = -1
for En in tqdm(champE):
    i += 1

    psi = evolve_crank_nikolson(psi, potentiel_spatial, En, dt, x)
    assert np.isclose(np.sum(np.abs(psi)**2) * dx, 1.0, atol=1e-6), f"Normalization condition not satisfied {np.sum(np.abs(psi)**2) * dx}"
    psi_history[i, :] = psi.copy()  # Store the wavefunction at this time step


    psi_fonda = evolve_crank_nikolson(psi_fonda, potentiel_spatial, 0, dt, x)       # Evolve the fundamental state without the laser field
    assert np.isclose(np.sum(np.abs(psi_fonda)**2) * dx, 1.0, atol=1e-6), f"Normalization condition not satisfied for fundamental state {np.sum(np.abs(psi_fonda)**2) * dx}"
    psi_fonda_history[i, :] = psi_fonda.copy()  # Store the fundamental state at this time step
##############################################################################