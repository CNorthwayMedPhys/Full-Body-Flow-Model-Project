# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 14:13:49 2026

@author: Cassidy.Northway
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
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
# mean = np.mean(Frac1)
# std_dev = np.std(Frac1)

# plt.hist(Frac1, bins=50, alpha=0.7, color='skyblue', edgecolor='black', label='Blood Volume Dose')

# # Add vertical lines for mean and standard deviation
# plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2E}')
# plt.axvline(mean - std_dev, color='green', linestyle='dotted', linewidth=2, label=f'1 Std Dev: {std_dev:.2E}')
# plt.axvline(mean + std_dev, color='green', linestyle='dotted', linewidth=2)

# # Add labels and title
# plt.xlabel('Dose (Gy)')
# plt.ylabel('Blood Volume Count')
# plt.title('Single Fracation Results for Co-60 Sweeping TBI')
# plt.legend()
# plt.grid(axis='y', alpha=0.75)

# # Display the plot
# plt.show()



