# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 13:29:53 2026

@author: cbnor
"""

import numpy as np
import matplotlib.pyplot as plt
import os

#%%
filename = "BVResults200.npy"
cd = os.getcwd()
stacked_results = np.load(os.path.join(cd,filename))

#Flatten the results
results = stacked_results.flatten()
BV_data = results
# Calculate mean and standard deviation
mean = np.mean(BV_data)
std_dev = np.std(BV_data)

# Create the histogram
plt.hist(BV_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black', label='Blood Volume Dose')

# Add vertical lines for mean and standard deviation
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2E}')
plt.axvline(mean - std_dev, color='green', linestyle='dotted', linewidth=2, label=f'1 Std Dev: {std_dev:.2E}')
plt.axvline(mean + std_dev, color='green', linestyle='dotted', linewidth=2)

# Add labels and title
plt.xlabel('Dose (Gy)')
plt.ylabel('Blood Volume Count')
plt.title('Preliminary Results for Co-60 Sweeping TBI')
plt.legend()
plt.grid(axis='y', alpha=0.75)

# Display the plot
plt.show()
