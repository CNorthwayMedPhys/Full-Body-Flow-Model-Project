# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:01:37 2025

@author: Cassidy.Northway
"""

import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
#%% Load in the vessel
cutoff = 1.25 * 10 #mm
name = 'right_pulmonary_veins3'
path = dir_path + '\\FittedVesselSegments\\'+ name + '.npy'
vesselArray =np.load(path)
#%% Create an array that is the total distance elapsed between n and n+1 vessel
distanceArray = np.zeros(np.shape(vesselArray)[0])
for i in range(np.shape(vesselArray)[0]-1):
    a = vesselArray[i,0:3]
    b = vesselArray[i+1,0:3]
    distanceArray[i+1] = np.linalg.norm(a-b) + distanceArray[i]
    #linear distance between the two points plus previous distance
#%%Find index of where that new cut-off value (from P1) is
differences = np.abs(distanceArray - cutoff)
closest_index = differences.argmin()

#%%Save the modified segement
trimmedArray = vesselArray[:closest_index+1,:]
np.save(path,trimmedArray)