# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:07:20 2025

@author: Cassidy.Northway
"""

import os
import numpy as np

#%%Files names and manual data 

cd = os.getcwd()
filePath = "VMAT_Files\\4DDoseData\\"
majorDir = os.path.join(cd,filePath)
for (root,dirs,files) in os.walk(majorDir,topdown=True):
  print("Directory path: %s"%root)
  print("Directory Names: %s"%dirs)
  print("Files Names: %s"%files)

MU = [971, 1260, 112, 112, 150, 150 ,214, 1072, 214]
i = 0
VOI = np.zeros([1,1])
sumDose = []  
for file in files:
    if 'Vessel' in file:

        data = np.load(os.path.join(root,file))
        if np.all(VOI==0):
            VOI = data[0,1:]
            data = data[1:,1:]
            sumDose = np.sum(data,axis = 0) * MU[i]
            i += 1
        else:
            if np.array_equal(VOI, data[0,1:]):
                data = data[1:,1:]
                sumDose = sumDose + (np.sum(data,axis = 0) * MU[i])
                i +=1
                
finalDose = 6.6511073E13 * sumDose               


