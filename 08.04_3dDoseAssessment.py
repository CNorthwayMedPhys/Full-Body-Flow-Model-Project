# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:07:20 2025

@author: Cassidy.Northway
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
#%%Files names and manual data 

cd = os.getcwd()
dosefile = os.path.join(cd,"extSSDdose.3ddose")


#%%
class dose:
    
    def __init__(self, dimensions, edges, centers, doseArray, unCertArray):
        self._dimensions = dimensions
        self._edges = edges
        self._centers = centers
        self._doseArray = doseArray
        self._unCertArray = unCertArray
    @property
    def dimensions(self):
        return self._dimensions
    @property
    def doseArray(self):
        return self._doseArray
    @property
    def centers(self):
        return self._centers
    @property
    def edges(self):
        return self._edges  
    @property
    def unCertArray(self):
        
        return self._unCertArray
    
def readDose(dosefile):

    #Parses file line by line
    with open(dosefile) as x:
    
        #Read dimensions
        xdim, ydim, zdim = x.readline().split()
        xdim = int(xdim)
        ydim = int(ydim)
        zdim = int(zdim)
        
        #Read voxel edges, not in one consitent line in the complete version so this looks for 
        #lines going from pos to negative
        xedgesList = []
        yedgesList = []
        zedgesList = []
        #Read voxel edges
        xedgesList = x.readline().split()
        xedgesList = [float(item) for item in xedgesList]
        yedgesList = x.readline().split()
        yedgesList = [float(item) for item in yedgesList]
        zedgesList = x.readline().split()
        zedgesList = [float(item) for item in zedgesList]
                
        
        
        
        #Determine the center of the voxels
        xcenterList = []
        ycenterList = []
        zcenterList = []
        for i in range(xdim):
            edge1 = xedgesList[i]
            edge2 = xedgesList[i+1]
            xcenterList.append((edge1+edge2)/2)
        for i in range(ydim):
            edge1 = yedgesList[i]
            edge2 = yedgesList[i+1]
            ycenterList.append((edge1+edge2)/2)
        for i in range(zdim):
            edge1 = zedgesList[i]
            edge2 = zedgesList[i+1]
            zcenterList.append((edge1+edge2)/2)
        
        #Build dose image
        doseArray = np.zeros((xdim,ydim,zdim))
        doseList = []
        m=0
        for z in range(ydim*zdim):
                tempList = x.readline().strip().split()
                doseList.extend(tempList)
        for k in range(zdim):
            for j in range(ydim):
                    for i in range(xdim):
                            doseArray[i,j,k] = doseList[m]
                            m += 1        
                
                
        unCertArray = np.zeros((xdim,ydim,zdim))
        # unCertList = []
        # m=0
        # for z in range(ydim*zdim):
        #         tempList = x.readline().strip().split()
        #         unCertList.extend(tempList)
        # for k in range(zdim):
        #     for j in range(ydim):
        #             for i in range(xdim):
        #                     unCertArray[i,j,k] = unCertList[m]
        #                     m += 1
        
  #####Normally use this for combined dose   #####   
        # doseList = x.readline().strip().split()
        # m = 0
    
        # for k in range(zdim):
        #     for j in range(ydim):
        #             for i in range(xdim):
        #                     doseArray[i,j,k] = doseList[m]
        #                     m += 1
        # unCertArray = np.zeros((xdim,ydim,zdim))                    
        # unCertList = x.readline().strip().split()
        # m = 0

        # for k in range(zdim):
        #     for j in range(ydim):
        #             for i in range(xdim):
        #                     unCertArray[i,j,k] = unCertList[m]
        #                     m += 1                    

            
        


        doseObject = dose([xdim, ydim, zdim], [xedgesList, yedgesList, zedgesList], [xcenterList, ycenterList, zcenterList], doseArray,unCertArray) 
    return doseObject

#%%Begin file processing 
doseObj= readDose(dosefile)
dose = doseObj.doseArray
#%%
xMidIndex = 127
yMidIndex = 62 
zMidIndex= 441 #based on ct looks good
doseObj = doseObj

xcenters = doseObj.centers[0][127]
ycenters = doseObj.centers[1][62]
zcenters = doseObj.centers[2][441]

doseIso = np.mean(doseObj.doseArray[xMidIndex-1:xMidIndex+2,yMidIndex-1:yMidIndex+2,zMidIndex-1:zMidIndex+2])
unCertIso= np.mean(doseObj.unCertArray[xMidIndex-1:xMidIndex+2,yMidIndex-1:yMidIndex+2,zMidIndex-1:zMidIndex+2])

dosePlane = doseObj.doseArray[:,yMidIndex,zMidIndex]

#%% Calc treatment time for both fields!!! Assuming delivering 2 Gy per frac
k = 9.76E14
deliveryTime = 1 *(1/doseIso) *(1/k)


    

