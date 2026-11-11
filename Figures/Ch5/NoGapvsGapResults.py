# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:49:13 2026

@author: Cassidy.Northway
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

#%%FCN: Calculate the HI 
def calculate_HI(data):
    x = np.sort(data)                           # X-axis values sorted
    y = np.arange(1, len(data) + 1) / len(data) # Y-axis proportions (0 to 1)
    y = np.flip(y)
    i2 = np.where(y==0.02)[0]
    i98 = np.where(y==0.98)[0]
    i50 = np.where(y == 0.5)[0]
    HI = (x[i2]-x[i98])/x[i50]
    return HI[0]


#%% Load in data
cd = '//phsahome2.phsabc.ehcnet.ca/Cassidy.Northway/Remote Git/DoseResults'

#Load Gap Data 
Gap = np.load(os.path.join(cd,"BVResults_AllVMAT0.npy"))

#Load no Gap data
noGap = np.load(os.path.join(cd,"BVResults_AllVMAT_noGaps0.npy"))

#Combine the two arrays for prepping plot
allData = np.concat((Gap,noGap))
max_dose = np.max(allData)

#%%Plot the DVH
fig, ax = plt.subplots()

# Compute histogram counts for Gap data
bins = np.linspace(0, np.max(Gap), 100)
counts, bin_edges = np.histogram(Gap, bins=bins)
dvh_counts = np.cumsum(counts[::-1])[::-1]
dvh_volume_percentGap = (dvh_counts / len(Gap)) * 100
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax.plot(bin_centers, dvh_volume_percentGap, linewidth=1, color='black', label='Including beam-off time')

# Compute histogram counts for noGap data
bins = np.linspace(0, np.max(Gap), 100)
counts, bin_edges = np.histogram(noGap, bins=bins)
dvh_counts = np.cumsum(counts[::-1])[::-1]
dvh_volume_percentnoGap = (dvh_counts / len(Gap)) * 100
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax.plot(bin_centers, dvh_volume_percentnoGap, linewidth=1, linestyle = '--', color='black', label='Excluding beam-off time')
plt.legend()

ax2 = ax.twinx()
diff = dvh_volume_percentGap - dvh_volume_percentnoGap

ax2.plot(bin_centers, diff, linewidth=1, color='red')


plt.xlabel('Dose (Gy)')

ax.set_ylabel('Volume (%)')
ax2.set_ylabel('Difference (%)', color ='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.legend()
plt.xlim(0, max_dose)
plt.show()

#%%Determine the HI
print("Gap HI is " + f"{calculate_HI(Gap):.2f}")
print("no Gap HI is " + f"{calculate_HI(noGap):.2f}")


#%%What about all fracs?

#Load Gap Data 
Gap = Gap+np.load(os.path.join(cd,"BVResults_AllVMAT1.npy"))+np.load(os.path.join(cd,"BVResults_AllVMAT2.npy"))+np.load(os.path.join(cd,"BVResults_AllVMAT3.npy"))+np.load(os.path.join(cd,"BVResults_AllVMAT4.npy"))+np.load(os.path.join(cd,"BVResults_AllVMAT5.npy"))

#Load no Gap data
noGap = noGap + np.load(os.path.join(cd,"BVResults_AllVMAT_noGaps1.npy"))+ np.load(os.path.join(cd,"BVResults_AllVMAT_noGaps2.npy"))+ np.load(os.path.join(cd,"BVResults_AllVMAT_noGaps3.npy"))+ np.load(os.path.join(cd,"BVResults_AllVMAT_noGaps4.npy"))+ np.load(os.path.join(cd,"BVResults_AllVMAT_noGaps5.npy"))

#%%Plot the DVH
fig, ax = plt.subplots()

# Compute histogram counts for Gap data
bins = np.linspace(0, np.max(Gap), 100)
counts, bin_edges = np.histogram(Gap, bins=bins)
dvh_counts = np.cumsum(counts[::-1])[::-1]
dvh_volume_percentGap = (dvh_counts / len(Gap)) * 100
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax.plot(bin_centers, dvh_volume_percentGap, linewidth=1, color='black', label='Including beam-off time')

# Compute histogram counts for noGap data
bins = np.linspace(0, np.max(Gap), 100)
counts, bin_edges = np.histogram(noGap, bins=bins)
dvh_counts = np.cumsum(counts[::-1])[::-1]
dvh_volume_percentnoGap = (dvh_counts / len(Gap)) * 100
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax.plot(bin_centers, dvh_volume_percentnoGap, linewidth=1, linestyle = '--', color='black', label='Excluding beam-off time')
plt.legend()

ax2 = ax.twinx()
diff = dvh_volume_percentGap - dvh_volume_percentnoGap

ax2.plot(bin_centers, diff, linewidth=1, color='red')


plt.xlabel('Dose (Gy)')

ax.set_ylabel('Volume (%)')
ax2.set_ylabel('Difference (%)', color ='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.legend()
plt.xlim(0, np.max(Gap))
plt.show()

#%%Determine the HI
print("Gap HI all frac is " + f"{calculate_HI(Gap):.2f}")
print("no Gap HI all frac is " + f"{calculate_HI(noGap):.2f}")