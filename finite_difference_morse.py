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
import matplotlib.pyplot as plt

# define physical constants
hbar_c = 1973  # hbar*c in eV*Angstroms
m_cl_amu = 35  # mass of chlorine in amu
amu_MeV_by_c2 = 931.49432  # conversion factor 1 amu = 931.49432 MeV/c^2
m_cl = m_cl_amu * amu_MeV_by_c2 * 1e-6  # mass of chlorine in eV

De = 4.618  # Morse potential well depth in eV
a = 1.869  # Morse potential "a" parameter in Angstroms^-1
# (1 Angstrom = 10^-10 m)
r_e = 1.275  # Chlorine equilibrium separation in Angstroms

# Define r values in angrstoms:
r = np.linspace(0.01, 10, 1000)
V = De*(np.exp(-2*a*(r - r_e)) - 2*np.exp(-a*(r - r_e)))

# Plot Morse Potential in interesting region
f, ax = plt.subplots()
ax.plot(r, V)
ax.set_xlim(0, 5)
ax.set_ylim(-7, 2)
f.show()
input("Press <Return> to close")
#plt.plot(r, V)
#plt.show()