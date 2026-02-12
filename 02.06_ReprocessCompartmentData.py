# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:07:20 2025

@author: Cassidy.Northway
"""

import os
import numpy as np
import matplotlib.pyplot as plt
#%%Files names and manual data 
cd = os.getcwd()
filename = "CompartmentData\\PA\\"
compartmentnum = [0,1,2,3,4,5,6,6,7,8,9,10,12,15,18,19,20,21,22,23,24,25,26,27]



for num in compartmentnum:
    iFilePath = os.path.join(cd,filename,str(num)+"eventTrace500ms.npy")
    iArray = np.load(iFilePath)
    

    
    binEdgesL = iArray[0,:]
    binCounts = iArray[1,:]
    
    binCenters = binEdgesL + 0.25
    newFirst = binCenters[0] - 0.5
    newEnd = binCenters[-1] + 0.5
    fFirst = np.array([[newFirst],[binCounts[0]]])
    fEnd = np.array([[newEnd],[binCounts[-1]]])
    fMiddle = np.vstack([binCenters,binCounts])
    
    fArray = np.append(fFirst, fMiddle ,axis = 1)
    fArray = np.append(fArray, fEnd, axis =1 )
    
    fPath = os.path.join(cd,filename,str(num)+"eventTrace500ms.npy")
    np.save(fPath, fArray)
    
    
  
    





