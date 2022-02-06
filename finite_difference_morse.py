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

# Plot energy eigenvalues alongside the Morse Potential
# Plot every 20th energy level from 0 to 200 alongside full Morse well
f1, ax1 = plt.subplots()
# Plot Morse Potential
morse, = ax1.plot(r, V_r)

# Plot Energy Eigenvalues
for i in range(0, 141, 20):
    energy, = ax1.plot(E_u[i]*np.ones(n_steps), "r--")
    energy_level_string = "n = " + str(i)
    ax1.text(3.5, E_u[i] + 0.05, energy_level_string)

# Add legend
ax1.legend([morse, energy], ["Morse Potential", "Energy Eigenvalue"],
           loc="center left", bbox_to_anchor=(.75, 0.5))

# Set axes limits/labels
ax1.set_xlim(0.75, 5)
ax1.set_ylim(-5, 0.1)
ax1.set_xlabel(r"Internuclear separation ($\AA$)")
ax1.set_ylabel("Energy (eV)")
ax1.set_title("Morse potential and Energy Eigenvalues (Full Well)")
plt.savefig("full_morse_and_energies.png")
f1.show()

# Plot of first few energy eigenvalues
f2, ax2 = plt.subplots()
# Plot Morse Potential
morse, = ax2.plot(r, V_r)
# Plot Energy Eigenvalues
for i in range(0, 7):
    energy, = ax2.plot(E_u[i]*np.ones(n_steps), "r--")
    energy_level_string = "n = " + str(i)
    ax2.text(1.27, E_u[i] + 0.01, energy_level_string)

# Add legend
ax2.legend([morse, energy], ["Morse Potential", "Energy Eigenvalue"],
           loc="best")

# Set axes limits/labels
ax2.set_xlim(1.1, 1.5)
ax2.set_ylim(-4.65, -4.2)
ax2.set_xlabel(r"Internuclear separation ($\AA$)")
ax2.set_ylabel("Energy (eV)")
ax2.set_title("Morse potential and Energy Eigenvalues (Bottom of Well)")
plt.savefig("lower_well_morse_energies.png")
f2.show()

plt.show()