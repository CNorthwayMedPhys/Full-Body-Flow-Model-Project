# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 15:58:10 2026

@author: Cassidy.Northway

I want to plot the mean and std of dose as a fcn of BV number

"""

import numpy as np
import matplotlib.pyplot as plt

#%% Data

mean = np.array([3.26, 3.0, 2.7, 2.66, 2.69, 2.68, 2.62, 2.64, 2.66, 2.73])#, 2.67, 2, 2])
std = np.array([0.37, 0.37, 0.56, 0.52,0.50,0.52,0.51,0.51,0.51,0.52])#,0.52,0.5,0.5])
BVnumb = np.array([1,1E1,1E2,5E2,1E3,2.5E3,5E3,7.5E3,1E4,2.5E4])#,5E4,7.5E4,1E5])

#%% Plot

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
line1 = ax1.semilogx(BVnumb,mean,color = 'k', label= "Mean Dose" )
line2 = ax2.semilogx(BVnumb,std, color = 'k', linestyle = '--', label = "Standard Deviation")

ax1.set_xlabel("Number of Blood Volumes Simulated")
ax1.set_ylabel("Mean Dose (Gy)")


ax2.set_ylabel("Standard Deviation (Gy)")
ax2.set_ylim(bottom = 0.3)

lns = line1+line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

plt.show()