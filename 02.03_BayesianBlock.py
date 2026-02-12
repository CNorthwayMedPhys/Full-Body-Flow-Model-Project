# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:07:20 2025

@author: Cassidy.Northway
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import astropy
from sklearn.cluster import DBSCAN

#%%Files names 
cd = os.getcwd()
filename = "timeStepQuery\XCAT_AP_100ms15E9.npy"
filepath = os.path.join(cd,filename)

data = np.load(filepath)

#%%Find the VOI with the most events
VOI = data[1, 100] #random
mask = np.where(data[1,:] == VOI)
single_VOI_data = data[:,mask]
single_VOI_data = single_VOI_data.reshape(4,1185)
dose = single_VOI_data[0,:]
mask = np.where(dose == 0)
time = single_VOI_data[2,:]
#%%
#If we use 0.002 s binning and just look at number of events
plt.hist(time, bins=1185, alpha=0.5, density = True, label='0.002', color='blue')
plt.hist(time, bins = 500, alpha=0.5, density = True, label='0.1', color='orange')

plt.xlabel('Time')
plt.ylabel('Event Frequency')
plt.title('Overlapping Histograms with Matplotlib')
plt.legend(loc='upper right')
plt.show()

#%% Bayesian Block Binning

norm_dose = (dose - dose.min()) / (dose.max() - dose.min())

bin_edges = astropy.stats.bayesian_blocks(time,fitness = 'regular_events', dt =300 )
plt.hist(time, bins=1185, histtype='stepfilled',
          alpha=0.2)
plt.hist(time, bins= bin_edges, color='black',
          histtype='step' )
plt.show()




    

