import numpy as np
import matplotlib.pyplot as plt



###############################################################################
## CONFIGURATION
###############################################################################
N = 2048
x = np.linspace(-100, 100, N)        # in a.u. => in bohr units
epsilon = 0.001
V = -1.0 / np.sqrt(x**2 + epsilon)
###############################################################################



###############################################################################
## CALCULATION TO FIND EIGENVALUES
###############################################################################
H = np.zeros((N, N))
dx = x[1] - x[0]
kinetic = -0.5 * (np.diag(np.ones(N-1), 1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), -1)) / dx**2
potential = np.diag(V)
H = kinetic + potential


valeurs_propres, vecteurs_propres = np.linalg.eigh(H)
###############################################################################

###############################################################################
## PLOT EIGENVALUES
###############################################################################
plt.figure(figsize=(10, 6))
plt.plot(x, V, label='Potential V(x)', color='blue')
for i in range(len(valeurs_propres)):
    if valeurs_propres[i] < 0:  # Only plot bound states
        plt.plot(x, vecteurs_propres[:, i], label=f'Eigenstate {i+1}', alpha=0.7)
plt.title('Eigenstates of a Single Atom Model')
plt.xlabel('Position (a.u.)')
plt.ylabel('Energy (a.u.)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(x, vecteurs_propres[:, np.argmin(valeurs_propres)], label='Fundamental State', color='red')
plt.plot(x, V, label='Potential V(x)', color='blue')
plt.title('Fundamental State of the Single Atom Model')
plt.plot(x, 1/np.sqrt(np.pi) * np.exp(-np.abs(x)), label='Analytical Solution', color='green', linestyle='--')
plt.xlabel('Position (a.u.)')
plt.ylabel('Wavefunction')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()


E1 = 0.5  # Energy of the fundamental state in a.u. (see theory)
plt.figure(figsize=(10, 6))
plt.plot(valeurs_propres, label='Eigenvalues', color='purple')
plt.plot(-E1/np.array(range(1, len(valeurs_propres)+1))**2, label='Analytical Energy Levels', color='orange', linestyle='--')
plt.title('Eigenvalues of the Single Atom Model')
plt.xlabel('State Index')
plt.ylabel('Energy (a.u.)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()

# Check normalization of the fundamental state in SI units

# Constants
a0 = 5.29177210903e-11  # Bohr radius in meters

# Fundamental state (ground state)
psi = vecteurs_propres[:, np.argmin(valeurs_propres)]

# Normalize in SI units
dx_SI = dx * a0
norm_SI = np.sum(np.abs(psi)**2) * dx_SI
print(f"Normalization of the fundamental state in SI units: {norm_SI:.6e}")

plt.show()