"""
run this file prior to running HHG_simulation in order to generate the initial wavefunction
:author: Maxence BARRE
:note: this file is used to generate the initial wavefunction for the HHG simulation
"""
import numpy as np
from scipy.sparse import diags
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import os
import json

###############################################################################
## CONFIGURATION
###############################################################################
# Parameters
dx = 0.25                                         # in a.u. => should be small enough to resolve well the wave function => dx < 0.1 is a good value
x = np.arange(-200, 200, dx)                     # in a.u. => 1au=roughly 24as, this is the space grid for the simulation
epsilon = 1.41                               # small value to avoid singularity at x=0, epsilon=1 is a good value 
# WARNING : play with the value of epsilon to get at the end the ground state of energy = to the iniosation potential of the atom you are looking in
# decrease epsilon => increase the absolute value of the energy of the ground state
# WARNING: dx MUST be the same as the one used in HHG_simulation.py (the grid can be a sub-grid of the one used in HHG_simulation.py, but the step must be the same)


N = len(x)                                       # number of points in the grid

# 1D Hydrogen-like potential (softened to avoid singularity at x=0)
Z = 1.0
V = -Z / np.sqrt(x**2 + epsilon**2)

# Kinetic energy operator (finite difference, central)
main_diag = np.full(N, -2.0)
off_diag = np.full(N-1, 1.0)
laplacian = diags([off_diag, main_diag, off_diag], [-1, 0, 1]) / dx**2

# Hamiltonian
H = -0.5 * laplacian.toarray() + np.diag(V)

# Solve eigenvalue problem
num_states = 15
eigvals, eigvecs = eigh(H)
eigvals = eigvals[:num_states]
eigvecs = eigvecs[:, :num_states]

print("IONISATION POTENTIAL OF THE ATOM SIMULATED (in a.u): ", eigvals[0])
print("IONISATION POTENTIAL OF THE ATOM SIMULATED (in eV): ", eigvals[0] * 27.2114)  # conversion from a.u. to eV


# Normalize eigenfunctions
eigvecs /= np.sqrt(np.sum(np.abs(eigvecs)**2, axis=0) * dx)

# Plot
plt.figure(figsize=(8,6))
for i in range(num_states):
    plt.plot(x, eigvecs[:, i], label=f'n={i+1}, E={eigvals[i]:.3f}')
plt.xlabel('x')
plt.ylabel('Wavefunction + Energy')
plt.title('1D Hydrogen Atom Eigenstates')
plt.legend()
plt.grid()

plt.figure()
plt.plot(x, eigvecs[:, 0])
plt.plot(x, np.exp(-np.abs(x)), label='Analytical 1s state', linestyle='--', color='red')


##################################################################################
## SAVING ALL THE WAVEFUNCTIONS IN A FOLDER
##################################################################################
folder_name = f"wavefunctions_{dx}_{epsilon}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name, exist_ok=True)
    for i in range(num_states):
        filename = os.path.join(folder_name, f"eigenstate_{i+1}.npy")
        np.save(filename, np.vstack((x, eigvecs[:, i])))

        # Prepare data for JSON
        wavefunctions_info = {
            "epsilon": epsilon,
            "dx": dx,
            "wavefunctions": {}
        }
        for i in range(num_states):
            filename = f"eigenstate_{i+1}.npy"
            wavefunctions_info["wavefunctions"][filename] = eigvals[i].item()

        # Save JSON file in the same folder
        json_path = os.path.join(folder_name, "wavefunctions_info.json")
        with open(json_path, "w") as f:
            json.dump(wavefunctions_info, f, indent=4)

plt.show()

