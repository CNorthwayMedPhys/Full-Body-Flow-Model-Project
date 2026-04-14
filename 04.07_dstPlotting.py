# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:46:42 2024

@author: cbnor

Linearly interpolates for event frequency from 0.5s histogram data
"""

import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
#%% Load in data


cd = os.getcwd()

dst_results1 = np.load(os.path.join(cd,"dstResults.npy"))
dst_results2 = np.load(os.path.join(cd,"dstResultsLarge.npy"))

dst_results = np.append(dst_results2,dst_results1[1:,:],axis=0)
 
mean = dst_results[:,0]   
std_dev = dst_results[:,1]
dst = dst_results[:,2]
x=mean
y=dst
#%%Run stats test
resMean = stats.spearmanr(mean,dst)
resStd = stats.spearmanr(std_dev,dst)

#%%Follow up question if we ignore the larger dst values (>= 1sec) what happens?

resMeanSmall = stats.spearmanr(dst_results1[1:,0],dst_results1[1:,2])

#%%Plot results for both
fig,(ax1,ax2) = plt.subplots(2,1,layout = "constrained")

dst = dst_results1[1:,2]
mean = dst_results1[1:,0]
std_dev = dst_results1[1:,1]

ax1.semilogx(dst, mean)
ax1.set_ylabel("Mean of Dose \n to Blood Volumes \n (Gy)")

# Plot on the second axes
ax2.semilogx(dst, std_dev)
ax2.set_ylabel("Standard Deviation \n of Dose to Blood \n Volumes (Gy)")

fig.supxlabel("Log of Dose Rate Sampling Time Step Size (s)")
fig.suptitle("Impact of Dose Sampling Time Step Size on \n Blood Volume Dose during Sweeping Cobalt-60 TBI")
# Automatically adjust spacing between plots
#plt.tight_layout()
plt.show()


#%%
def statistic(x): # permute only `x`
    return stats.spearmanr(x, y).statistic
res_exact = stats.permutation_test((x,), statistic,
    permutation_type='pairings')
res_asymptotic = stats.spearmanr(x, y)
res_exact.pvalue, res_asymptotic.pvalue # asymptotic pvalue is too low