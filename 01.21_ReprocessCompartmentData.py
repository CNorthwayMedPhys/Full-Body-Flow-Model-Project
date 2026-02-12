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
newTimeStep = 0.1 #s, assuming og step size is 0.002 s


for num in compartmentnum:
    iFilePath = os.path.join(cd,filename,str(num)+"eventTrace2msRes.npy")
    iArray = np.load(iFilePath)
    
    mergeCount = int(newTimeStep/0.002)
    
    binEdgesL = iArray[0,:]
    binCounts = iArray[1,:]
    
    newCount = []
    newEdges = []
    for i in range(round(len(binCounts)/mergeCount)):
        newCount.append(np.sum(binCounts[i*mergeCount:(i+1)*mergeCount]))
        newEdges.append(binEdgesL[i*mergeCount])
        
       
    remainder = len(binCounts)-(i*mergeCount)
    newCount.append(np.sum(binCounts[-remainder:]))  
    newEdges.append(binEdgesL[-remainder])
    
    newCount = np.array(newCount)
    newEdges = np.array(newEdges)
    
    fArray = np.vstack([newEdges,newCount])
    fFilePath = os.path.join(cd,filename,str(num)+"eventTrace100ms.npy")
    np.save(fFilePath,fArray)
    
    #plt.hist()
  
    





