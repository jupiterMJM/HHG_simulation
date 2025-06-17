"""
the HHG can be seen as a semi-classical system through the 3-step model
in this model, the electron is first ionized, then accelerated by the laser field, and finally recombines with the ion to emit high-energy photons
during the time the electron is free, it can be treated as a classical particle
:author: Maxence BARRE
"""


# import of the modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c, hbar, epsilon_0
import matplotlib.cm as cm
import matplotlib.colors as mcolors



##########################################################################################
## CONFIGURATION
##########################################################################################
# parameters of the laser
wavelength = 800      # nm, NOT IN A.U. the conversion is done later
I_wcm2 = 1e14  # Intensity in W/cm^2, NOT IN A.U. the conversion is done later

# parameters of the atom (especially the ionization potential)
ionization_potential = 15.76  # eV, NOT IN A.U. the conversion is done later, why did i put it here? doesn t seem to be used in the code

# parameters of the simulation
time_range = np.linspace(0, 10, 2000)  # time range in fs, NOT IN A.U. the conversion is done later
step_ionization = .01  # fs, time step for ionization, NOT IN A.U. the conversion is done later
plot_in_au = True  # if True, the time is plotted in atomic units, if False, the time is plotted in fs
############################################################################################



###########################################################################################
## SIMULATION AND PLOTTING
###########################################################################################
# Colormap: on normalise entre les bornes min/max attendues
cmap = cm.turbo
norm = mcolors.Normalize(vmin=0, vmax=10)


# calculate the laser field at the time of ionization
omega = 2 * np.pi * c / (wavelength * 1e-9)  # angular frequency in rad/s
E_laser = np.sqrt(I_wcm2 * 1e4 / (c*epsilon_0))

fig, ax = plt.subplots(figsize=(10, 6))

for time_ionization in np.arange(0, time_range[-1], step_ionization):
    # print(f'Ionization at {time_ionization} fs')
    # calculate the electron's initial velocity after ionization
    v_initial = 0  # m/s, assuming the electron is initially at rest, see Robin Weissenbilder's thesis

    t_i = time_ionization * 1e-15  # convert time to seconds
    t = time_range[time_range >= time_ionization] * 1e-15  # convert time to seconds
    x = e*E_laser / (m_e * omega**2) * (np.sin(omega*t) - np.sin(omega * t_i) - omega* (t - t_i) * np.cos(omega * t_i))
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
        u_p = e**2*E_laser**2 / (4*m_e * omega**2)          # ponderomotive energy in J
        e_k = 2*u_p*(np.cos(omega * t_recombination) - np.cos(omega * t_i))**2/(1.6e-19)  # kinetic energy of the electron at recombination, see R.W's thesis
        if not plot_in_au:
            ax.plot(t * 1e15, x * 1e9, label=f'Ionization at {time_ionization} fs', color=cmap(norm(e_k)))  # convert x to nm for plotting, color by kinetic energy
        else:
            ax.plot(t / 2.418884e-17, x / 5.29177210903e-11, color=cmap(norm(e_k)))  # time in a.u., position in a.u., color by kinetic energy

if not plot_in_au:
    plt.xlabel('Time (fs)')
    plt.ylabel('Position (nm)')
else:
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Position (a.u.)')
plt.title('Electron Position in the Laser Field')
# plt.legend()
plt.grid()

# Ajouter une colorbar fictive (avec mappable)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # pas besoin de données
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Return Energy (eV)")

if not plot_in_au:
    ax.plot(time_range, 3*np.sin(omega * time_range * 1e-15), 'r--', label='Laser Field')  # convert time to seconds for plotting, amplitude is arbitrary for visualization
else:
    ax.plot(time_range * 1e-15 / 2.418884e-17, 3*np.sin(omega * time_range * 1e-15) *1e-9/ 5.29177210903e-11, 'r--', label='Laser Field')
plt.show()