# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 13:39:14 2025

@author: Cassidy.Northway
"""
import numpy as np
import matplotlib.pyplot as plt
import os

Tfield = round(6.81 * 60, 3)
cd = os.getcwd()
BV_data = np.load(os.path.join(cd,'5E2BVSim.npy'))
#%%
# Normalize the data by exposure time
scale = (2*Tfield) / BV_data[:,1]
norm_dose = BV_data[:,0] * scale
dose = BV_data[:,0]
# Calculate mean and standard deviation
mean = np.mean(norm_dose)
std_dev = np.std(norm_dose)

# Create the histogram
plt.hist(norm_dose, bins=15, alpha=0.7, color='skyblue', edgecolor='black', label='Blood Volume Dose')

# Add vertical lines for mean and standard deviation
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2E}')
plt.axvline(mean - std_dev, color='green', linestyle='dotted', linewidth=2, label=f'1 Std Dev: {std_dev:.2E}')
plt.axvline(mean + std_dev, color='green', linestyle='dotted', linewidth=2)

# Add labels and title
plt.xlabel('Scaled Dose (Gy)')
plt.ylabel('Blood Volume Count')
plt.title('Preliminary Results for Co-60 Sweeping TBI')
plt.legend()
plt.grid(axis='y', alpha=0.75)

# Display the plot
plt.show()

# Create the histogram
plt.hist(dose, bins=15, alpha=0.7, color='skyblue', edgecolor='black', label='Blood Volume Dose')
mean = np.mean(dose)
std_dev = np.std(dose)
# Add vertical lines for mean and standard deviation
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2E}')
plt.axvline(mean - std_dev, color='green', linestyle='dotted', linewidth=2, label=f'1 Std Dev: {std_dev:.2E}')
plt.axvline(mean + std_dev, color='green', linestyle='dotted', linewidth=2)

# Add labels and title
plt.xlabel('Unscaled Dose (Gy)')
plt.ylabel('Blood Volume Count')
plt.title('Preliminary Results for Co-60 Sweeping TBI')
plt.legend()
plt.grid(axis='y', alpha=0.75)

# Display the plot
plt.show()