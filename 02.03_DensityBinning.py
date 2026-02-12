# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:07:20 2025

@author: Cassidy.Northway
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from optbinning import ContinuousOptimalBinning
#%%Files names 
cd = os.getcwd()
filename = "timeStepQuery\XCAT_AP_2ms15E9.npy"
filepath = os.path.join(cd,filename)

data = np.load(filepath)

#%%Find the VOI with the most events
VOI = data[1, 100] #random
mask = np.where(data[1,:] == VOI)
single_VOI_data = data[:,mask]
single_VOI_data = single_VOI_data.reshape(4,1500)
single_VOI_data = single_VOI_data[:,single_VOI_data[2, :].argsort()]
dose = single_VOI_data[0,:]
mask = np.where(dose == 0)
time = single_VOI_data[2,:]
rel_error = single_VOI_data[3,:]
#%%
#If we use 0.002 s binning and just look at number of events
plt.hist(time, bins=1500, alpha=0.5, weights = dose, label='0.002', color='blue')
plt.xlabel('Time(s)')
plt.ylabel('Dose/Particle')
plt.title('Uniform binning of 2 ms')
plt.legend(loc='upper right')
plt.show()

#%% Wait let's bin based on uncertainity 
bin_edges = [time[0]]
sum_dose = 0
abs_error = 0
new_array = np.array([[0],[0],[0]])

for i in range(len(dose)):
    sum_dose = sum_dose + dose[i]
    abs_error = np.sqrt(abs_error**2 + (dose[i]*rel_error[i])**2)
    if abs_error/sum_dose < 0.8:
        bin_edges.append(time[i])
        new_array= np.append(new_array,[[time[i]],[sum_dose],[abs_error/sum_dose]],axis=1)
        sum_dose = 0 
        abs_error = 0 

    if i == len(dose)-1:
        bin_edges.append(time[i])
    
plt.plot(new_array[0,:],new_array[1,:])         
plt.show()       
        
    





    

