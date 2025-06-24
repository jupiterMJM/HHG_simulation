import numpy as np
from scipy.sparse import diags
from scipy.linalg import eigh


import matplotlib.pyplot as plt

# Parameters
dx = 0.05                                         # in a.u. => should be small enough to resolve well the wave function => dx < 0.1 is a good value
x = np.arange(-120, 120, dx)                     # in a.u. => 1au=roughly 24as, this is the space grid for the simulation
N = len(x)                                       # number of points in the grid

# 1D Hydrogen-like potential (softened to avoid singularity at x=0)
Z = 1.0
epsilon = 1e-9
V = -Z / np.sqrt(x**2 + epsilon**2)

# Kinetic energy operator (finite difference, central)
main_diag = np.full(N, -2.0)
off_diag = np.full(N-1, 1.0)
laplacian = diags([off_diag, main_diag, off_diag], [-1, 0, 1]) / dx**2

# Hamiltonian
H = -0.5 * laplacian.toarray() + np.diag(V)

# Solve eigenvalue problem
num_states = 5
eigvals, eigvecs = eigh(H)
eigvals = eigvals[:num_states]
eigvecs = eigvecs[:, :num_states]

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
plt.plot(x, eigvecs[:, 1])
plt.plot(x, np.exp(-np.abs(x)), label='Analytical 1s state', linestyle='--', color='red')
np.save('1D_hydrogen_eigenstates.npy', eigvecs[:, 1])
plt.show()

