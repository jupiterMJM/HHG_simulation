"""
ce fichier est issu de TDSE_hands_on.ipynb et rassemble bon nombre de fonctions utiles à la simulation 1D de HHG
:auteur: Maxence BARRE
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



def envelope(t, periode_au):
    """
    génère une enveloppe temporelle pour le laser
    :param t: tableau numpy de temps en a.u.
    :param periode_au: période du laser en a.u.
    :return: tableau numpy de l'enveloppe temporelle

    :note: l'enveloppe est une rampe de montée puis un plateau constant
    """
    t1 = t[0] + periode_au * 4
    t2 = t[-1] - periode_au * 4
    t = np.array(t)
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