"""
ce fichier est issu de TDSE_hands_on.ipynb et rassemble bon nombre de fonctions utiles à la simulation 1D de HHG
:auteur: Maxence BARRE
TODO: modifier le stdx
"""



# importation des modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm
from scipy.fft import fft, fftfreq
import scipy.sparse.linalg as spla
import scipy.sparse as sparse
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


t_au = 2.418884e-17                                 # s, constant for conversion to atomic units


def envelope(t, periode_au):
    """
    génère une enveloppe temporelle pour le laser
    :param t: tableau numpy de temps en a.u.
    :param periode_au: période du laser en a.u.
    :return: tableau numpy de l'enveloppe temporelle

    :note: l'enveloppe est une rampe de montée puis un plateau constant
    """
    t = np.array(t)
    t1 = t[0] + periode_au * 4
    t2 = t[-1] - periode_au * 4
    
    env = np.zeros_like(t, dtype=float)
    # Ramp up
    mask1 = (t < t1)
    env[mask1] = (t[mask1] - t[0]) / (t1 - t[0])
    # Flat top
    # mask2 = (t >= t1) & (t <= t2)
    # env[mask2] = 1.0
    # # Ramp down
    # mask3 = (t > t2)
    # env[mask3] = 1 - (t[mask3] - t2) / (t[-1] - t2)
    mask2 = (t >= t1)
    env[mask2] = 1.0
    return env


def envelope_pulse(t, periode_au):
    """
    genere une enveloppe temporelle pour le laser en sin^2
    :param t: tableau numpy de temps en a.u.
    :param periode_au: période du laser en a.u.
    :return: tableau numpy de l'enveloppe temporelle
    :note: 
    """
    retour = np.sin(np.pi/ (t[-1] - t[0]) * (t-t[0])) ** 2
    return retour


def envelope_laser_labo_approx(t, periode_au, fwhm=15e-15/t_au):
    """
    generate a laser envelope that could approximates the one in the lab (the envelope is a gaussian, centered on 0)
    :param t: numpy array of time in a.u.
    :param periode_au: period of the laser in a.u.
    :param fwhm: full width at half maximum of the laser in a.u. (by default 22 fs in a.u.)
    """
    # Convert FWHM to standard deviation (sigma) for Gaussian
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    # Create the Gaussian envelope
    env = np.exp(-((t- 0) ** 2) / (2 * sigma ** 2))
    # Normalize the envelope
    env /= np.max(env)
    return env



def calculate_caracterisitcs_wavepacket(psi, x, psi_fonda):
    """
    the aim is to not save the whole wavefunction (1000*1000 complex nb) but only a few characteristics (10*1000 cplx nb rougly)
    :param psi: wavefunction at a given time
    :param x: position grid
    :param psi_fonda: fundamental state at a given time
    :return: a dictionary with the characteristics of the wavepacket
    """
    # dipole_on_fonda = np.sum(np.conj(psi_fonda) * x * psi, axis=1)*dx  # dipole moment projected onto the fundamental state
    # dipole_on_itself = np.sum(np.conj(psi) * x * psi, axis=1)*dx  # dipole moment projected onto the wavefunction itself
    # moment_quantity = -1j * np.gradient(psi, x)  # momentum quantity projected onto the wavefunction
    # momentum_on_fonda = np.sum(np.conj(psi_fonda) * moment_quantity * psi, axis=1)*dx  # momentum projected onto the fundamental state
    # momentum_on_itself = np.conj(psi) * moment_quantity * psi  # momentum projected onto the wavefunction itself
    # scalar_product_fonda = np.conj(psi_fonda) * psi  # scalar product with the fundamental state
    # kinetic_energy = -0.5 * 
    dx = x[1] - x[0]  # assuming uniform grid
    product_fonda = np.conj(psi_fonda) * psi * dx
    product_itself = np.conj(psi) * psi * dx
    print(product_fonda.shape, product_itself.shape)
    dipole_on_fonda = np.sum(product_fonda * x, axis=1)  # dipole moment projected onto the fundamental state
    dipole_on_itself = np.sum(product_itself * x, axis=1)  # dipole moment projected onto the wavefunction itself
    moment_quantity = -1j * np.gradient(psi, x, axis=1)  # momentum quantity projected onto the wavefunction
    momentum_on_fonda = np.sum(product_fonda*moment_quantity, axis=1)  # momentum projected onto the fundamental state
    momentum_on_itself = np.sum(product_itself*moment_quantity, axis=1)  # momentum projected onto the wavefunction itself
    scalar_product_fonda = np.sum(product_fonda, axis=1)  # scalar product with the fundamental state
    kinetic_energy = -0.5 * np.sum(product_itself * np.gradient(psi, x, axis=1), axis=1)  # kinetic energy projected onto the wavefunction itself
    # stdx_fonda = np.sqrt(np.sum(product_fonda * x**2, axis=1).real - dipole_on_fonda.real**2)  # standard deviation of position projected onto the fundamental state
    # stdx_itself = np.sqrt(np.sum(product_itself * x**2, axis=1).real - dipole_on_itself.real**2)  # standard deviation of position projected onto the wavefunction itself
    # stdp_fonda = np.sqrt(np.sum(product_fonda * moment_quantity**2, axis=1).real - momentum_on_fonda.real**2)  # standard deviation of momentum projected onto the fundamental state
    # stdp_itself = np.sqrt(np.sum(product_itself * moment_quantity**2, axis=1).real - momentum_on_itself.real**2)  # standard deviation of momentum projected onto the wavefunction itself
    return dipole_on_fonda, dipole_on_itself, momentum_on_fonda, momentum_on_itself, scalar_product_fonda, kinetic_energy, -1, -1, -1, -1