"""
this file is a prolongation of the file classical_view.py
instead of using the formula obtained by theoritical calculation of the three step model,
we integrate the classical equations of motion of the electron in the laser field
this way, we can extend the simulation to other things than a sinusoidal laser field
:author: Maxence BARRE
"""


# import of the modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c, hbar, epsilon_0
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.integrate import solve_ivp
from tqdm import tqdm




##########################################################################################
## CONFIGURATION
##########################################################################################
# parameters of the laser
wavelength = 800*1e-9      # m, NOT IN A.U. the conversion is done later
I_wcm2 = 1e14  # Intensity in W/cm^2, NOT IN A.U. the conversion is done later


# parameters of the simulation
time_range = np.linspace(0, 20, 4000)  # time range in fs, NOT IN A.U. the conversion is done later
step_ionization = .01  # fs, time step for ionization, NOT IN A.U. the conversion is done later
plot_in_au = True  # if True, the time is plotted in atomic units, if False, the time is plotted in fs

# constants in atomic units
E_h = 4.35974e-18  # J, Hartree energy
a0 = 5.29177210903e-11  # m, Bohr radius
# constants in SI units
c = 299792458  # m/s, speed of light
m_e = 9.10938356e-31  # kg, electron mass
e = 1.602176634e-19  # C, elementary charge

############################################################################################



#############################################################################################
## SIMULATION FUNCTIONS
#############################################################################################

def integration_laws_of_motion(t, E_laser, x0=0, v0=0):
    """ this function integrates the classical equations of motion of the electron in the laser field
    :note: the calculation are all done in atomic units, so the parameters are in SI units
    :param t: time in s (array)
    :param E_laser: laser field in V/m (function of time)
    :param x0: initial position in m (default: 0)
    :param v0: initial velocity in m/s (default: 0)
    :return: position and velocity of the electron in atomic units
    """
    def dXdt(t, x):
        """ function to solve the ode problem,
        X here represents the state of the system, which is a vector [position, velocity]
        :param t: time in s
        :param x: state of the system, a vector [position, velocity]
        :return: derivative of the state of the system, a vector [velocity, acceleration]
        """
        dxdt = x[1]  # velocity
        dvdt = - e * E_laser(t) / m_e  # acceleration, F = m*a, with F = -e*E_laser 
        return [dxdt, dvdt]
    
    return solve_ivp(dXdt, t_span = [t[0], t[-1]], y0=[x0, v0], t_eval=t)


def envelope(t, periode_au):
    """
    génère une enveloppe temporelle pour le laser
    :param t: tableau numpy de temps en a.u.
    :param periode_au: période du laser en a.u.
    :return: tableau numpy de l'enveloppe temporelle

    :note: l'enveloppe est une rampe de montée puis un plateau constant
    """
    t1 = time_range[0] + periode_au * 4
    env = np.zeros_like(t, dtype=float)
    
    if type(t) is np.ndarray:
        retour = np.zeros_like(t, dtype=float)
        retour[t < t1] = (t[t < t1] - time_range[0]) / (t1 - time_range[0])
        retour[t >= t1] = 1.0
        return retour
    else:
        if t < t1:
            env = (t - time_range[0]) / (t1 - time_range[0])
        else:
            env = 1.0
    return env


def E_laser(t):
    """ this function returns the laser field at time t in V/m
    :param t: time in s
    :return: laser field in V/m
    """

    omega = 2 * np.pi * c / wavelength  # angular frequency in rad/s
    E0 = np.sqrt(I_wcm2 * 1e4 / (c*epsilon_0))  # electric field amplitude in V/m
    return E0 * np.sin(omega * t)*envelope(t, periode_au=2*np.pi/omega)  # sinusoidal laser field
    


############################################################################################
    





###########################################################################################
## SIMULATION AND PLOTTING
###########################################################################################
# Colormap: on normalise entre les bornes min/max attendues
cmap = cm.turbo
norm = mcolors.Normalize(vmin=0, vmax=10)

omega = 2 * np.pi * c / wavelength  # angular frequency in rad/s

fig, ax = plt.subplots(figsize=(10, 6))

for time_ionization in tqdm(np.arange(0, time_range[-1], step_ionization)):
    # print(f'Ionization at {time_ionization} fs')
    # calculate the electron's initial velocity after ionization
    v_initial = 0  # m/s, assuming the electron is initially at rest, see Robin Weissenbilder's thesis

    t_i = time_ionization * 1e-15  # convert time to seconds
    t = time_range[time_range >= time_ionization] * 1e-15  # convert time to seconds
    

    retour = integration_laws_of_motion(t, E_laser, x0=0, v0=v_initial)
    x = retour.y[0]  # position of the electron in m, first row
    v = retour.y[1]  # velocity of the electron in m/s, second row




    # lets find out electron that are recombining with the ion
    if np.all(x[1:] * x[1] > 0): # if the electron is always on the same side of the ion
        # the electron is not recombining with the ion
        if not plot_in_au:
            ax.plot(t * 1e15, x * 1e9, label=f'Ionization at {time_ionization} fs', color='lightblue', alpha=0.3)  # convert x to nm for plotting
        else:
            ax.plot(t / 2.418884e-17, x / 5.29177210903e-11, color='lightblue', alpha=0.3)  # time in a.u., position in a.u.
    else:
        t_recombination = t[np.where(np.diff(np.sign(x[1:])))[0][0]]  # find the time of recombination
        # print(f'Recombination at {t_recombination * 1e15:.2f} fs for ionization at {time_range[time_ionization]} fs')
        x = x[t <= t_recombination]  # only keep the position before recombination
        t = t[t <= t_recombination]  # only keep the time before recombination
        E0 = np.sqrt(I_wcm2 * 1e4 / (c*epsilon_0))
        u_p = e**2*E0**2 / (4*m_e * omega**2)          # ponderomotive energy in J
        e_k = 2*u_p*(np.cos(omega * t_recombination) - np.cos(omega * t_i))**2/(1.6e-19)  # kinetic energy of the electron at recombination, see R.W's thesis
        if not plot_in_au:
            ax.plot(t * 1e15, x * 1e9, label=f'Ionization at {time_ionization} fs', color=cmap(norm(e_k)))  # convert x to nm for plotting, color by kinetic energy
        else:
            ax.plot(t / 2.418884e-17, x / 5.29177210903e-11, color=cmap(norm(e_k)))  # time in a.u., position in a.u., color by kinetic energy

    # plt.show()

if not plot_in_au:
    plt.xlabel('Time (fs)')
    plt.ylabel('Position (nm)')
else:
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Position (a.u.)')
plt.title('Electron Position in the Laser Field (via Integration)')
# plt.legend()
plt.grid()

# Ajouter une colorbar fictive (avec mappable)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # pas besoin de données
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Return Energy (eV)")

if not plot_in_au:
    ax.plot(time_range, E_laser(time_range*1e-15), 'r--', label='Laser Field')  # convert time to seconds for plotting, amplitude is arbitrary for visualization
else:
    # ax.plot(time_range * 1e-15 / 2.418884e-17, E_laser(time_range*1e-15)/(E_h/(e*a0)), 'r--', label='Laser Field')
    ax.plot(time_range * 1e-15 / 2.418884e-17, E_laser(time_range*1e-15)/np.max( E_laser(time_range*1e-15))*50, 'r--', label='Laser Field')
plt.show()