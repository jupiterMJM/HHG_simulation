"""
in order to compare with the TDSE calculation, we try to generate a mask that can be superposed to the TDSE plot
:author: Maxence BARRE
"""


import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c, hbar, epsilon_0
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.integrate import solve_ivp
from tqdm import tqdm






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
        dvdt = -E_laser(t) # acceleration, F = m*a, with F = -e*E_laser in AU units (e=1, m=1 in AU)
        return [dxdt, dvdt]
    
    return solve_ivp(dXdt, t_span = [t[0], t[-1]], y0=[x0, v0], t_eval=t)



################################################################################
## MAIN FUNCTION
################################################################################
def plot_classical_trajectories(E_laser, time_grid, ax, time_grid_for_plot, position_grid_for_plot, plot_vertically=True):
    """ this function plots the classical trajectories of the electron in the laser field
    :param E_laser: laser field in V/m (function of time)
    :param time_grid: time grid in s (array)
    :param ax: axis to plot on
    :param time_grid_for_plot: time grid for plotting in fs
    :param position_grid_for_plot: position grid for plotting in a.u.
    """
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.turbo
    for time_ionization in tqdm(time_grid[time_grid<time_grid_for_plot[-1]][::20], desc="Integrating classical trajectories"):
        # integrate the equations of motion
        time_to_integrate_on = time_grid[time_grid>=time_ionization]
        result = integration_laws_of_motion(time_to_integrate_on, E_laser, x0=0, v0=0)
        # get the position and velocity of the electron
        position = result.y[0]
        velocity = result.y[1]

        if np.all(position[1:] * position[1] > 0):  # if the electron is always on the same side of the ion
            # the electron is not recombining with the ion
            time_out_of_scope = time_grid[np.where(np.logical_and(position > position_grid_for_plot[0], position < position_grid_for_plot[-1]))][-1]
            # print(time_out_of_scope, time_grid_for_plot[0])
            if time_out_of_scope < time_grid_for_plot[0]:
                # if the last time is before the plot range, skip this ionization time
                continue
            end_of_plot = min(time_grid_for_plot[-1], time_out_of_scope)
            position = position[np.logical_and(time_to_integrate_on >= time_grid_for_plot[0], time_to_integrate_on <= end_of_plot)]
            time_to_plot = time_to_integrate_on[np.logical_and(time_to_integrate_on >= time_grid_for_plot[0], time_to_integrate_on <= end_of_plot)]
            np.clip(position, position_grid_for_plot[0], position_grid_for_plot[-1], out=position)
            if plot_vertically:
                ax.plot(position, time_to_plot, alpha=0.5, color='lightblue', label=f'Ionization at {time_ionization:.2f} fs')
            else:
                ax.plot(time_to_plot, position, alpha=0.5, color='lightblue', label=f'Ionization at {time_ionization:.2f} fs')

        else:
            # print(time_to_integrate_on.shape, np.where(np.diff(np.sign(position[1:])))[0][0])
            t_recombination = time_to_integrate_on[np.where(np.diff(np.sign(position[1:])))[0][0]]  # find the time of recombination

            # print(t_recombination, velocity[time_to_integrate_on == t_recombination])
            position = position[time_to_integrate_on <= t_recombination]  # only keep the position before recombination
            velocity_recomb = velocity[time_to_integrate_on == t_recombination][0]
            time_to_integrate_on = time_to_integrate_on[time_to_integrate_on <= t_recombination]  # only keep the time before recombination

            if time_to_integrate_on[-1] < time_grid_for_plot[0] or time_to_integrate_on[0] > time_grid_for_plot[-1]:
                # if the recombination time is outside the plot range, skip this ionization time
                continue


            # print(position.shape, velocity.shape, time_to_integrate_on.shape)
            energy_recomb = 0.5 * velocity_recomb**2
            # print(energy_recomb)
            # plot the trajectory
            np.clip(position, position_grid_for_plot[0], position_grid_for_plot[-1], out=position)
            position = position[np.logical_and(time_to_integrate_on >= max(time_grid_for_plot[0], time_ionization), time_to_integrate_on <= time_grid_for_plot[-1])]
            # print(time_grid_for_plot[np.logical_and(time_grid_for_plot>=time_ionization, time_grid_for_plot<=t_recombination)].shape, position.shape, time_grid_for_plot.shape)
            if plot_vertically:
                ax.plot(position, time_grid_for_plot[np.logical_and(time_grid_for_plot>=time_ionization, time_grid_for_plot<=t_recombination)], alpha=0.7, color=cmap(norm(energy_recomb)), label=f'Ionization at {time_ionization:.2f} fs')
            else:
                ax.plot(time_grid_for_plot[np.logical_and(time_grid_for_plot>=time_ionization, time_grid_for_plot<=t_recombination)], position, alpha=0.7, color=cmap(norm(energy_recomb)), label=f'Ionization at {time_ionization:.2f} fs')


if __name__ == "__main__":
    # constants (DO NOT CHANGE THESE VALUES)
    c = 2.99792458e8                                    # m/s, speed of light
    e = 1.602176634e-19                                 # C, elementary charge
    m = 9.10938356e-31                                  # kg, electron mass
    a0 = 5.291772108e-11                                # m, Bohr radius
    hbar = 1.0545718e-34                                # J.s, reduced Planck's constant
    t_au = 2.418884e-17                                 # s, constant for conversion to atomic units
    epsilon_0 = 8.854187817e-12                         # F/m, vacuum permittivity
    E_h = 4.3597447222071e-18                           # J, Hartree energy


    fig, ax = plt.subplots(figsize=(10, 6))

    dx = 0.01                                         # in a.u. => should be small enough to resolve well the wave function => dx < 0.1 is a good value
    x = np.arange(-120, 120, dx)                     # in a.u. => 1au=roughly 24as, this is the space grid for the simulation
    dt = 0.05                                           # in a:u , if dt>0.05, we can t see electron that comes back to the nucleus
    t = np.arange(-1000, 1000, dt)                      # also in a.u. => 1au=roughly 24as

    # Laser Parameters
    wavelength = 800                                    # nm, NOT IN A.U. the conversion is done later
    I_wcm2 = 1e14                                       # Intensity in W/cm^2, NOT IN A.U. the conversion is done later
    freq = 3e8 / (wavelength * 1e-9)                    # Frequency in Hz, converting nm to m
    omega_au = 2*np.pi*freq*t_au
    periode_au = 2*np.pi / omega_au                     # Period in atomic units
    pulse_duration = 25 * periode_au                    # Pulse duration in atomic units
    E0_laser = np.sqrt(2/(epsilon_0*c)) * np.sqrt(I_wcm2*1e4) / (E_h/(e*a0))       # from intensity to eletric field in a.u.
    champE_func = lambda t: E0_laser*np.cos(omega_au * t)


    plot_classical_trajectories(champE_func, t, ax, time_grid_for_plot=t[np.logical_and(t>=0, t<=750)], position_grid_for_plot=x)
    plt.show()