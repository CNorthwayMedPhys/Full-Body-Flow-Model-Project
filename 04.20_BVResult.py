# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 13:29:53 2026

@author: cbnor
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
#%%
filename = "BVResults20.npy"
cd = os.getcwd()
results = np.load(os.path.join(cd,filename))

mean = results[:,0]
std = results [:,1]
BVnum = results[:,2]

fig, ax1 = plt.subplots()

ax1.set_xlabel("Number of Blood Volumes Simulated")
ax1.set_ylabel("Mean Dose (Gy)")
ax1.plot(BVnum, mean, linestyle = "solid")
ax1.tick_params(axis = "y")
ax1.set_ylim(1.7,1.8)

ax2 = ax1.twinx()
ax2.set_ylabel("Standard Deviaiton of Dose (Gy)")
ax2.plot(BVnum, std , linestyle = "--")
ax2.tick_params(axis='y')
ax2.set_ylim(0.29,0.33)

plt.show()

fig, ax = plt.subplots()
ax.fill_between(BVnum, mean-std, mean+std)

resMean = stats.spearmanr(mean,BVnum)
resStd = stats.spearmanr(std,BVnum)