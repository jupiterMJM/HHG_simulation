"""
ce fichier rassemble l'ensemble des fonctions permettant de résoudre l'équation de Schrödinger dépendante du temps (TDSE) en 1D
l'idee ici de rassembler les fonctions qui peuvent etre utiles pour toutes simulations 1D de physique quantique
:auteur: Maxence BARRE

:note: toutes les fonction evolve (implicit_euler, explicit_euler, split operator) ne sont pas présentes ici car ne fonctionnent pas
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



def potentiel_CAP(x, x_start, x_end, eta_0):
    """
    Calcule le potentiel complexe absorbant (CAP) pour une grille donnée.

    :param x: tableau numpy des positions sur la grille
    :param x_start: limite inférieure de la grille
    :param x_end: limite supérieure de la grille
    :param eta_0: amplitude maximale du CAP
    :return: tableau numpy du potentiel complexe absorbant
    """
    # Initialisation du potentiel CAP
    eta = np.zeros_like(x, dtype=complex)

    # Définition de la région où le CAP commence à être appliqué
    x_cap_start = 1.2 * x_start 
    x_cap_end = 1.2 * x_end

    # Calcul du CAP
    mask_left = (x <= x_start) & (x >= x_cap_start)
    mask_right = (x >= x_end) & (x <= x_cap_end)

    eta[mask_left] = -1j * eta_0 * ((x[mask_left] - x_start) / (x_start - x_cap_start))**2
    # print(-1j * eta_0 * ((x[mask_left] - x_cap_start) / (x_start - x_cap_start))**2)
    eta[mask_right] = -1j * eta_0 * ((x[mask_right] - x_end) / (x_end - x_cap_end))**2

    # Assure que le CAP est constant au-delà de x_cap_start et x_cap_end
    mask_constant_left = (x < x_cap_start)
    mask_constant_right = (x > x_cap_end)

    eta[mask_constant_left] = -1j * eta_0
    eta[mask_constant_right] = -1j * eta_0

    return eta


def evolve_crank_nikolson(psi, V, E, dt, x):
    """
    Evolue la fonction d'onde psi en utilisant la méthode de Crank-Nicolson pour résoudre l'équation de Schrödinger dépendante du temps.
    En partant du temps t_n et en évoluant vers t_{n+1}.
    :param psi: fonction d'onde au temps t_n
    :param V: potentiel independant du temps en fonction de x, tableau numpy
    :param E: amplitude du champ électrique au temps t_n, scalaire
    :param dt: pas de temps en a.u.
    :param x: tableau numpy des positions sur la grille
    :return: fonction d'onde au temps t_{n+1}

    :note: on prefere resoudre TDSE pas à pas afin d'avoir plus de liberté sur les potentiels et sur les enregistrements, plots....
    """
    Nx = len(x)
    dx = x[1] - x[0]  # pas d'espace
    diagonals = [-2*np.ones(Nx), np.ones(Nx-1), np.ones(Nx-1)]
    L = sparse.diags(diagonals, [0, -1, 1], dtype=np.complex128) / dx**2
    potential = sparse.diags(V, 0, dtype=np.complex128) + sparse.diags(E * x, 0, dtype=np.complex128)       # TODO regarder s il faut un - ou pas dans le potentiel du champ electrique
    assert np.any(potential.data != 0), "Potential matrix should not be zero"
    H = -0.5 * L +  potential # Hamiltonian operator in sparse matrix form
    I = sparse.diags(np.ones(Nx), 0)
    A = (I + 1j * dt / 2 * H).tocsc()
    B = (I - 1j * dt / 2 * H).tocsc()
    b = B.dot(psi)
    psi = spla.spsolve(A, b)

    # psi /= np.linalg.norm(psi)  # normalisation, in theory not needed but good practice
    # print(psi.dtype, (np.sum(np.conj(psi) * psi)*dx).dtype, np.sum(np.conj(psi) * psi) * dx)
    assert (np.sum(np.conj(psi) * psi) * dx).imag < 1e-10, "Normalization condition not satisfied"
    # psi /= (np.sum(np.conj(psi) * psi) * dx).real  # Normalisation to ensure the wavefunction remains normalized, .real to avoid complex normalization (and thus phase shift)

    # in theory, crank-nikolson conserves the norm, BUT numerical algorithm introduc small errors that keep adding up and de-normalize the wavefunction
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx) # le sqrt est necessaire pour la normalisation
    if norm == 0:
        raise ValueError("Wavefunction norm is zero, cannot normalize.")
    psi /= norm 
    return psi