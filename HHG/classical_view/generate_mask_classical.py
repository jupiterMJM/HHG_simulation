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



############################################################################################
## PURE PLOTTING FUNCTIONS
## in order to go from curves to a matrix of mask
############################################################################################

def blend_pixel_matrices(matrix1:np.array, matrix2:np.array) -> np.array:
    """
    Blends two matrices pixel by pixel, returning a new matrix where each pixel is taken from matrix2 if it is non-zero,
    non-NaN, and not None; otherwise, it takes the pixel from matrix1.
    :param matrix1: First input matrix.
    :param matrix2: Second input matrix.
    :return: Blended matrix.
    """
    retour = np.where(np.logical_and(matrix2 != 0, matrix2 != None), matrix2, matrix1)
    return np.array(retour, dtype=float)






def genere_array_modified(x_grid, y_grid, x, value):
    """
    Generate a binary matrix where only the closest elements to the curve defined by value are set to 1.

    Parameters:
    x_grid (numpy.ndarray): Grid of x values.
    y_grid (numpy.ndarray): Grid of y values.
    value (numpy.ndarray): Precomputed values of the function f(x) corresponding to x_grid.

    Returns:
    numpy.ndarray: Binary matrix with 1s for points closest to the curve.
    """
    # Create meshgrid and compute grid points
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))

    # Compute curve points using the precomputed value
    # print(x.shape, value.shape)
    curve_points = np.column_stack((x, value))

    # Calculate distances from grid points to curve points
    distances = cdist(grid_points, curve_points)

    # Find the minimum distance for each grid point
    min_distances = distances.min(axis=1)

    # Determine the closest points to the curve
    closest_indices = np.argmin(distances, axis=0)

    # Initialize the matrix with zeros
    matrix = np.zeros_like(X_grid, dtype=bool)

    # Set the closest grid points to True (1)
    for idx in closest_indices:
        # print(idx)
        matrix.flat[idx] = True

    # matrix[matrix == False] = None

    return matrix

###############################################################################



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



################################################################################
## MAIN FUNCTION
################################################################################
def generate_mask_from_classical(time_grid, position_grid, E_field, time_grid_zoomed, position_grid_zoomed):
    """
    Generate a mask from classical trajectories based on the electric field.

    Parameters:
    time_grid (numpy.ndarray): Grid of time values. IN SECONDS
    position_grid (numpy.ndarray): Grid of position values. IN METERS
    E_field (function): Function representing the electric field as a function of time.
    Returns:
    numpy.ndarray: Final mask representing the classical trajectories.
    """
    # cmap = cm.turbo
    norm = mcolors.Normalize(vmin=0, vmax=10)
    retour = np.zeros((len(position_grid_zoomed), len(time_grid_zoomed)), dtype=float)
    # retour[retour==0] = None  # Initialize with None for empty pixels
    # alpha_retour = .5

    for time_ionization in tqdm(time_grid[1:]):
        # print(f"Processing ionization time: {time_ionization} s")
        t = time_grid[time_grid >= time_ionization]

        # Integrate the equations of motion for the electron
        result = integration_laws_of_motion(t, E_field, x0=0, v0=0)
        # print(f"Integration result got")
        if not result.success or len(result.y) == 0:
            # print(f"Integration failed for time {time_ionization}. Skipping this trajectory.")
            continue
        # Extract position and velocity
        x = result.y[0]
        v = result.y[1]

        if np.all(x[1:] * x[1] > 0): # if the electron is always on the same side of the ion
            # alpha = 0.3 # transparency for the mask
            color = norm(0.1)
        else:
            
            t_recombination = t[np.where(np.diff(np.sign(x[1:])))[0][0]]  # find the time of recombination
            x = x[t <= t_recombination]  # only keep the position before recombination
            v_recomb = v[t== t_recombination][0]  # velocity at recombination
            t = t[t <= t_recombination]  # only keep the time before recombination
            
            energy_recomb = 0.5 * m_e * v_recomb**2/e  # kinetic energy at recombination

            color = norm(energy_recomb)+0.1 # TODO modify
            print(f"Electron recombining at time {time_ionization} s, energy at recombination: {energy_recomb:.2e} J, velocity: {v_recomb:.2e} m/s, color: {color:.2f}")
            # alpha = 0.5
        
        # plt.figure()
        # plt.plot(t, x, label=f'Ionization at {time_ionization:.2f} fs')

        # t = t[np.logical_and(x >= position_grid_zoomed[0], x <= position_grid_zoomed[-1])]
        # x = x[np.logical_and(x >= position_grid_zoomed[0], x <= position_grid_zoomed[-1])]
        # print(t)
        x = x[np.logical_and(t >= time_grid_zoomed[0], t <= time_grid_zoomed[-1])]
        t = t[np.logical_and(t >= time_grid_zoomed[0], t <= time_grid_zoomed[-1])]
        if len(t) == 0 or len(x) == 0:
            # print(f"No valid trajectory points found for time {time_ionization}. Skipping this trajectory.")
            continue
        # print(alpha, color)
        # print(t, x)
        # plt.figure()
        # plt.scatter(t, x, color=color[:3], alpha=alpha, s=1, label=f'Ionization at {time_ionization:.2f} fs')
        # plt.show()

        # Create a mask for the current trajectory
        mask = genere_array_modified(time_grid_zoomed, position_grid_zoomed, t, x)
        # plt.figure()
        # plt.imshow(mask, cmap="grey", aspect='auto', extent=(time_grid_zoomed[0], time_grid_zoomed[-1], position_grid_zoomed[0], position_grid_zoomed[-1]))

        mask_color = np.zeros((len(position_grid_zoomed), len(time_grid_zoomed)), dtype=float)
        mask_color[mask] = color
        # mask_color[~mask] = 0
        # plt.figure()
        # plt.imshow(mask_color, aspect='auto', extent=(time_grid_zoomed[0], time_grid_zoomed[-1], position_grid_zoomed[0], position_grid_zoomed[-1]), cmap='grey')
        # plt.title(f"Mask for ionization at {time_ionization:.2f} s")
        # plt.colorbar()
        # # plt.show()
        # Blend the mask with the existing matrix

        # plt.figure()
        # plt.imshow(retour, aspect='auto', extent=(time_grid_zoomed[0], time_grid_zoomed[-1], position_grid_zoomed[0], position_grid_zoomed[-1]), cmap='grey')
        # plt.title("Current state of the mask")
        # plt.colorbar()
        # plt.show()

        retour = blend_pixel_matrices(retour, mask_color)
        # plt.figure()
        # plt.imshow(retour, aspect='auto', extent=(time_grid_zoomed[0], time_grid_zoomed[-1], position_grid_zoomed[0], position_grid_zoomed[-1]), cmap='grey')
        # plt.title("Blended mask after processing current trajectory")
        # plt.colorbar()
        # plt.show()

    return retour


if __name__ == "__main__":


    # Example usage
    time_grid = np.linspace(-10, 10, 1000)*1e-15  # Time grid in seconds
    position_grid = np.linspace(-15e-9, 15e-9, 100)  # Position grid in meters

    def E_field(t):
        """ Example electric field function. Replace with actual field calculation. """
        wavelength = 800*1e-9      # m, NOT IN A.U. the conversion is done later
        I_wcm2 = 1e14  # Intensity in W/cm^2, NOT IN A.U. the conversion is done later
        omega = 2 * np.pi * c / wavelength  # angular frequency in rad/s
        E0 = np.sqrt(I_wcm2 * 1e4 / (c*epsilon_0))  # electric field amplitude in V/m
        return E0 * np.sin(omega * t)

    time_grid_zoomed = time_grid[250:750]  # Zoomed time grid (middle part)
    position_grid_zoomed = position_grid[25:75]  # Zoomed position grid (middle part)

    print("Generating mask from classical trajectories...")
    mask = generate_mask_from_classical(time_grid, position_grid, E_field, time_grid_zoomed, position_grid_zoomed)
    print("Mask generation complete.")
    plt.imshow(mask, cmap='turbo', aspect='auto',
               extent=(time_grid_zoomed[0], time_grid_zoomed[-1], position_grid_zoomed[0], position_grid_zoomed[-1]))
    plt.colorbar(label='Mask Value')
    plt.title("Generated Mask from Classical Trajectories")
    plt.show()