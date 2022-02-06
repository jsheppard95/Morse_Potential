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
m_h_amu = 1  # mass of hydrogen in amu
m = m_cl_amu*m_h_amu/(m_cl_amu + m_h_amu)  # reduced mass
amu_MeV_by_c2 = 931.49432  # conversion factor 1 amu = 931.49432 MeV/c^2
m_c2 = m * amu_MeV_by_c2 * 1e6  # mass of chlorine in eV

De = 4.618  # Morse potential well depth in eV
a = 1.869  # Morse potential "a" parameter in Angstroms^-1
# (1 Angstrom = 10^-10 m)
r_e = 1.275  # Chlorine equilibrium separation in Angstroms

# Define r values in angrstoms:
r_min = 0.01
r_max = 10
n_steps = 4000
r = np.linspace(r_min, r_max, n_steps)  # grid of r values in Angrstoms

V_r = De*(np.exp(-2*a*(r - r_e)) - 2*np.exp(-a*(r - r_e)))  # Morse potential in eV

# Create potential matrix V
V = np.diag(V_r)

# Create potential energy matrix K
# K = -hbar^2/(2md^2)*diagonal_matrix_-2_1 (see derivation)
# const C = =hbar^2/(2md^2) = (hbar*c)^2(2mc^2d^2), mc^2 has energy units, eV)
d = r[1] - r[0]  # incremental step size
ke_factor = -(hbar_c**2)/(2*m_c2*(d**2))
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
for i in range(0, 19, 2):
    energy, = ax1.plot(E_u[i]*np.ones(n_steps), "r--")
    energy_level_string = "n = " + str(i)
    ax1.text(-0.1, E_u[i] + 0.05, energy_level_string)

# Display equilibrium radius
re_line = ax1.axvline(x=r_e, color="green", linestyle="dotted")

# Add legend
ax1.legend([morse, energy, re_line],
           ["Morse Potential V(r)",
            "Energy Eigenvalue",
            "Equilibrium Separation"],
           loc="center left", bbox_to_anchor=(.5, 0.5))


# Set axes limits/labels
ax1.set_xlim(-0.2, 5.2)
ax1.set_ylim(-4.75, 0.1)
ax1.set_xlabel(r"Interatomic Separation, r, ($\AA$)")
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

# Display equilibrium radius
re_line = ax2.axvline(x=r_e, color="green", linestyle="dotted")

# Add legend
ax2.legend([morse, energy, re_line],
           ["Morse Potential V(r)",
            "Energy Eigenvalue",
            "Equilibrium Separation"],
           loc="best")

# Set axes limits/labels
ax2.set_xlim(0.9, 2)
ax2.set_ylim(-4.65, -2.4)
ax2.set_xlabel(r"Interatomic Separation, r, ($\AA$)")
ax2.set_ylabel("Energy (eV)")
ax2.set_title("Morse potential and Energy Eigenvalues (Bottom of Well)")
plt.savefig("lower_well_morse_energies.png")
f2.show()

# Calculate probability distribution for first few energy states
# Normalize wave functions
U0 = U[:, 0]
mod_U0 = np.multiply(np.conj(U0), U0)
mod_U0_norm = mod_U0/np.linalg.norm(mod_U0)

U1 = U[:, 1]
mod_U1 = np.multiply(np.conj(U1), U1)
mod_U1_norm = mod_U1/np.linalg.norm(mod_U1)

U2 = U[:, 2]
mod_U2 = np.multiply(np.conj(U2), U2)
mod_U2_norm = mod_U2/np.linalg.norm(mod_U2)


f3, ax3 = plt.subplots()
ground, = ax3.plot(r, mod_U0_norm)
first_excited, = ax3.plot(r, mod_U1_norm)
second_excited, = ax3.plot(r, mod_U2_norm)

# Display equilibrium radius
re_line = ax3.axvline(x=r_e, color="red", linestyle="dotted")

# Add legend
ax3.legend([ground, first_excited, second_excited, re_line],
           ["n = 0", "n = 1", "n = 2", "Equilibrium Separation"],
           loc="best")

# Set axes limits/labels
ax3.set_xlim(0.9, 1.7)
ax3.set_ylim(0, 0.14)
ax3.set_xlabel(r"Interatomic Separation, r, ($\AA$)")
ax3.set_ylabel(r"Probability of Separation (|$U(r)^2$|)")
ax3.set_title("Probability Distributions for Ground State and First Two Excited States")
plt.savefig("morse_prob_distribs.png")
f3.show()

print("Calculated Ground State Energy (eV):", E_u[0])

# Analytical solution for eigenenergies
# En = hv0(n + 1/2) - [hv0(n + 1/2)]^2/4De
# v0 = (a/2pi)*sqrt(2De/m)
# write out hv0*c/c:
# hv0 = a*hbarc*sqrt(2De/mc^2)
# Then shift energies down by De to match location of Morse
n_vals = np.arange(n_steps)
hv0 = a*hbar_c*np.sqrt(2*De/m_c2)
En_act = (hv0*(n_vals + 0.5) - (((hv0*(n_vals + 0.5))**2)/(4*De))) - De
print("Actual Ground State Energy (eV):", En_act[0])

plt.show()