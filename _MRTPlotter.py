# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 10:03:37 2025

@author: Cassidy.Northway

Intent: Create figure of MRT data were we have with mean and error bars with 
the Shin values overlaid
"""

import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
y = np.array([39.1, 40.0, 155.9, 202.1, 189.9, 173.2, 193.7, 122.0, 144.7, 184.7, 38.6 ])
y_error = np.array([15.6, 14.4, 101.5, 133.6, 133.5, 128.7, 133.6, 103.9, 127.5, 141.58, 15.0]) # Symmetric y-error

HEDOS = np.array([44, 44.2, 203.7, 191.4, 196.5, 164.3, 188.1, 114.4, 124.1, 145.4, 41.3])
x_labels = ['Left Heart',
'Right Heart',
'Pancreas',
'Spleen',
'Stomach',
'S. Intestine',
'L. Intestine',
'Liver',
'Kidneys',
'Brain',
'Lungs'
]

# Create the plot with error bars
fig, ax = plt.subplots()
ax.scatter(x, HEDOS,marker='o', color = '1', edgecolors = '0', zorder = 3, label = 'Shin et al. (2021) ')
ax.errorbar(x, y,yerr = y_error, linestyle = 'None', marker = 'o', color= ' 0.2', capsize = 4, elinewidth=2, label = 'Presented Work')

# Customize the plot
ax.set_xlabel('Compartment')
ax.set_ylabel('Mean Return Time (s)')

ax.tick_params(left=False, bottom = False)
ax.legend()
plt.grid(True, axis='y', color='gray', linewidth=0.25)


plt.xticks(x, x_labels, rotation=45, ha='right')

# Display the plot
plt.show()