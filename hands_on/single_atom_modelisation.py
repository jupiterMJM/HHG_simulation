"""
Single Atom Modelisation
Ce script permet de modéliser l'atome d'hydrogène en COORDONNEES SPHERIQUES en utilisant la méthode des différences finies pour résoudre l'équation de Schrödinger.
PAR CONSEQUENT, la fonction d'onde est séparée en 3: psi(r, theta, phi) = R(r) * Y(theta, phi), où R(r) est la fonction radiale et Y(theta, phi) est l'harmonique sphérique.
Nous on ne s'intéresse qu'à la fonction radiale R(r) car on est en 1D.
"""



import numpy as np
import matplotlib.pyplot as plt



###############################################################################
## CONFIGURATION
###############################################################################
N = 4000
x = np.linspace(1e-5, 50, N)        # in a.u. => in bohr units
epsilon = 0.
V = -1.0 / np.sqrt(x**2 + epsilon)
###############################################################################



###############################################################################
## CALCULATION TO FIND EIGENVALUES
###############################################################################
H = np.zeros((N, N))
dx = x[1] - x[0]
kinetic = -0.5 * (np.diag(np.ones(N-1), 1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), -1)) / dx**2
print(kinetic*dx**2)
potential = np.diag(V)
H = kinetic + potential


# ici les VP correspondent donc à u(r) = r * R(r) car on est en sperique!!!!!
valeurs_propres, vecteurs_propres_u = np.linalg.eigh(H)
vecteurs_propres_R = vecteurs_propres_u / x[:, np.newaxis]  # R(r) = u(r)/r
print(f"Eigenvalues: {valeurs_propres}")

# # some weird shit happening
vecteurs_propres_R = vecteurs_propres_R[:, valeurs_propres > -2]
valeurs_propres = valeurs_propres[valeurs_propres > -2]


# for i in range(len(valeurs_propres)):

#     vecteurs_propres[:, i] /= np.sqrt(np.sum(np.abs(vecteurs_propres[:, i])**2) * dx)
###############################################################################

###############################################################################
## PLOT EIGENVALUES
###############################################################################
plt.figure(figsize=(10, 6))
plt.plot(x, V, label='Potential V(x)', color='blue')
for i in range(len(valeurs_propres)):
    if valeurs_propres[i] < 0:  # Only plot bound states
        plt.plot(x, vecteurs_propres_R[:, i], label=f'Eigenstate {i+1}', alpha=0.7)
plt.title('Eigenstates of a Single Atom Model')
plt.xlabel('Position (a.u.)')
plt.ylabel('Energy (a.u.)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(x, vecteurs_propres_R[:, 0]/x, label=f'Fundamental State: {valeurs_propres[0]}', color='red')
plt.plot(x, V, label='Potential V(x)', color='blue')
plt.title('Fundamental State of the Single Atom Model')
plt.plot(x, np.exp(-np.abs(x)), label='Analytical Solution', color='green', linestyle='--')
# plt.plot(x, 1/np.pi * np.exp(-2*np.abs(x)), label='Analytical Solution bis2', color='green', linestyle='--')
plt.xlabel('Position (a.u.)')
plt.ylabel('Wavefunction')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()


analytical_theory = -0.5/np.array(range(1, len(valeurs_propres[valeurs_propres<0])+1))**2
plt.figure(figsize=(10, 6))
plt.plot(valeurs_propres[valeurs_propres<0], label='Eigenvalues', color='purple')
plt.plot(range(1, len(analytical_theory)+1), analytical_theory, label='Analytical Energy Levels', color='orange', linestyle='--')
plt.title('Eigenvalues of the Single Atom Model')
plt.xlabel('State Index')
plt.ylabel('Energy (a.u.)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()

plt.figure()
plt.plot(x, V, label='Potential V(x)', color='blue')
plt.hlines(valeurs_propres, xmin=x[0], xmax=x[-1], label = "Eigenvalues")

plt.show()