# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:20:24 2026

@author: cbnor
"""

import matplotlib.pyplot as plt
import numpy as np

# Radiation doses in Gray (Gy)
dose = np.linspace(0, 10, 100)

# Linear-quadratic model parameters (typical for mammalian cells)
alpha = 0.3  # Gy^-1
beta = 0.03  # Gy^-2

# Calculate surviving fraction (SF)
surviving_fraction = np.exp(-(alpha * dose + beta * (dose ** 2)))

# Plot
plt.figure(figsize=(7, 5))
plt.plot(dose, surviving_fraction, 'b-', linewidth=2, label='LQ Model Fit')
plt.yscale('log')

plt.xlabel('Radiation Dose (Gy)')
plt.ylabel('Surviving Fraction (S)')
#plt.title('Mammalian Cell Survival Curve')
#plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.ylim(1e-4, 1.1)
#plt.legend()
plt.tight_layout()
plt.show()