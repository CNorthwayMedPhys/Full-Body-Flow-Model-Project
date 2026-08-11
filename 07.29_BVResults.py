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
filename = "BVnumResults_VMAT.npy"
filename2 = "BVnumResults_VMAT_2ndPortion.npy"
#filename3 = "BVnumResults_VMAT_3rdPortion.npy"
cd = os.getcwd()
results = np.load(os.path.join(cd,filename))
results2 = np.load(os.path.join(cd,filename2))
#results3 = np.load(os.path.join(cd,filename3))
results = np.append(results,results2, axis = 0)
#results = np.append(results,results3, axis = 0)

mean = results[:,0]
std = results [:,1]
BVnum = results[:,2]

fig, ax1 = plt.subplots()

ax1.set_xlabel("Number of Blood Volumes Simulated")
ax1.set_ylabel("Mean Dose (Gy)")
ax1.plot(BVnum, mean, linestyle = "solid")
ax1.tick_params(axis = "y")
ax1.set_ylim(0.8,1.1)

ax2 = ax1.twinx()
ax2.set_ylabel("Standard Deviaiton of Dose (Gy)")
ax2.plot(BVnum, std , linestyle = "--")
ax2.tick_params(axis='y')
ax2.set_ylim(0.2,0.4)

mean= mean[mean != 0]
BVnum = BVnum[BVnum != 0]
std = std[std!=0]

resMean = stats.spearmanr(mean,BVnum)
resStd = stats.spearmanr(std,BVnum)