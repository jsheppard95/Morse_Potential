"""
Python Script employing the finite difference method to solve Schrodinger's
Equation including the Morse Potential for the energy eigenavalues and
corresponding eigenstates.

Author: Jackson Sheppard

Sources:
https://pubs.acs.org/doi/pdf/10.1021/acs.jchemed.7b00003
https://www.youtube.com/watch?v=9Wzdwxwrcnw
"""
import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

# define physical constants
hbar_c = 1973  # hbar*c in eV*Angstroms
m_cl_amu = 35  # mass of chlorine in amu
amu_MeV_by_c2 = 931.49432  # conversion factor 1 amu = 931.49432 MeV/c^2
m_cl = m_cl_amu * amu_MeV_by_c2 * 1e6  # mass of chlorine in eV

De = 4.618  # Morse potential well depth in eV
a = 1.869  # Morse potential "a" parameter in Angstroms^-1
# (1 Angstrom = 10^-10 m)
r_e = 1.275  # Chlorine equilibrium separation in Angstroms
#r_e = 0.1

# Define r values in angrstoms:
r_min = 0.01
r_max = 10
n_steps = 1000
r = np.linspace(r_min, r_max, n_steps)  # grid of r values in Angrstoms

V_r = De*(np.exp(-2*a*(r - r_e)) - 2*np.exp(-a*(r - r_e)))  # Morse potential in eV

# Create potential matrix V
V = np.diag(V_r)

# Create potential energy matrix K
# K = -hbar^2/(2md^2)*diagonal_matrix_-2_1 (see derivation)
# const C = =hbar^2/(2md^2) = (hbar*c)^2(2mc^2d^2), mc^2 has energy units, eV)
d = r[1] - r[0]  # incremental step size
ke_factor = -(hbar_c**2)/(2*m_cl*(d**2))
# first create diagonal matrix of -2
K = np.diag(-2*np.ones(n_steps))
# Now fill elements above and below diagonal with 1
for i in range(n_steps - 1):
    K[i, i+1] = 1
    K[i+1, i] = 1
K = ke_factor * K

# Define Hamilitonian
H = K + V

# Diagonalize, i.e find wavefunctions (eigenstates) and energy eigenvalues for
# Schrodinger's Equation:
# HU = EU, U = wave function
[E_u, U] = spla.eigh(H)  # E_u = eigenenergies, U = eigenvectors

print(E_u.shape)
print(U.shape)

print(E_u[:5])

# Plot energy eigenvalues alongside the Morse Potential

# Plot Morse Potential in interesting region
#f, ax = plt.subplots()
#ax.plot(r, V_r)
#ax.set_xlim(0, 5)
#ax.set_ylim(-7, 2)
#f.show()
#input("Press <Return> to close")
plt.plot(r, V_r)
plt.show()