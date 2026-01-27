# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 14:13:49 2026

@author: Cassidy.Northway
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
import matplotlib.lines as lines

#%%Load in data
Frac1 = np.load("1FracAbstract.npy")
Frac1 = Frac1[:,0]

# Fit the Gamma distribution to the data
shape, loc, scale = gamma.fit(Frac1)

print(f"Fitted shape (α): {shape:.2f}")
print(f"Fitted scale (θ): {scale:.2f}")

mean = shape *scale
variance = shape * (scale**2)
std = np.sqrt(variance)

# Create a range of x-values for the plot
x = np.linspace(0, 7, 1000)
pdf_fitted = gamma.pdf(x, a=shape, loc=loc, scale=scale)

# Plot histogram and fitted curve
plt.hist(Frac1, bins=30, density=True, alpha=0.5, label='Dose to Blood Volumes')

plt.plot(x, pdf_fitted, 'r-', label='Fitted Gamma PDF')
plt.axvline(mean, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f} Gy')
plt.title('Dose to Circulating Blood Volumes after One Fraction')
plt.xlabel('Dose to Blood Volume (Gy)')
plt.ylabel('Density')
plt.legend()


handles, labels = plt.gca().get_legend_handles_labels()
proxy_patch = lines.Line2D([0],[0], label= f'Std: {std:.2f} Gy', color = "white")
handles.append(proxy_patch)
plt.legend(handles=handles, labels=[h.get_label() for h in handles])
plt.show()

print(f"Mean of dist: {mean:.2f}")
print(f"Standard deviation of dist: {std:.2f}")

#%% All 6 fracs
Frac1 = np.load("1FracAbstract.npy")
Frac2 = np.load("1FracAbstract_2.npy")
Frac3 = np.load("1FracAbstract_3.npy")
Frac4 = np.load("1FracAbstract_4.npy")
Frac5 = np.load("1FracAbstract_5.npy")
Frac6 = np.load("1FracAbstract_6.npy")

FullFrac = Frac1[:,0] + Frac2[:,0] + Frac3[:,0] + Frac4[:,0] + Frac5[:,0] + Frac6[:,0]

# Fit the Gamma distribution to the data
mu, std = norm.fit(FullFrac)

# Create a range of x-values for the plot
x = np.linspace(0, 20, 1000)
pdf_fitted = norm.pdf(x, mu, std)

# Plot histogram and fitted curve
plt.hist(FullFrac, bins=30, density=True, alpha=0.5, label='Dose to Blood Volumes')

plt.plot(x, pdf_fitted, 'r-', label='Fitted Norm PDF')
plt.axvline(mu, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mu:.2f} Gy')
plt.title('Dose to Circulating Blood Volumes after All Fractions')
plt.xlabel('Dose to Blood Volume (Gy)')
plt.ylabel('Density')
plt.legend()


handles, labels = plt.gca().get_legend_handles_labels()
proxy_patch = lines.Line2D([0],[0], label= f'Std: {std:.2f} Gy', color = "white")
handles.append(proxy_patch)
plt.legend(handles=handles, labels=[h.get_label() for h in handles])
plt.show()

print(f"Mean of dist: {mu:.2f}")
print(f"Standard deviation of dist: {std:.2f}")
