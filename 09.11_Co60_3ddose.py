# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:07:20 2025

@author: Cassidy.Northway
based on https://github.com/prehensilecode/egsnrcpy/blob/main/egsnrc/_egsphant.c#L164 
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.filedialog as fd
import scipy as sp

root = tk.Tk()
root.lift()
root.attributes('-topmost',True)
root.after_idle(root.attributes,'-topmost',False)

#%%Define classes for handling files
class Dose:
    
    def __init__(self,dimensions, edges, centers, doseArray, errorArray):
        self._dimensions = dimensions
        self._edges = edges
        self._centers = centers
        self._doseArray = doseArray
        self._errorArray = errorArray
    
    @property
    def dimensions(self):
        return self._dimensions
    @property
    def doseArray(self):
        return self._doseArray
    @property
    def edges(self):
        return self._edges
    @property
    def centers(self):
        return self._centers
    @property
    def errorArray(self):
        return self._errorArray
    
    
def readDose():    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = fd.askopenfilenames(parent=root, filetypes =[ ("3DDose Files", '*.3ddose')])    
    file_list = list(file_path)
    doseObj = []
    

    #Parses file line by line
    for i in range(len(file_list)):
        file_path = file_list[i]
        with open(file_path) as x:
       
            #Read dimensions
            xdim, ydim, zdim = x.readline().split()
            xdim = int(xdim)
            ydim = int(ydim)
            zdim = int(zdim)
            
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
            
            m = 0
            #Build dose image
            doseArray = np.zeros((xdim,ydim,zdim))
            doseList = x.readline().strip().split()
            for k in range(zdim):
                for j in range(ydim):
                        for i in range(xdim):
                                doseArray[i,j,k] = doseList[m]
                                m += 1
            unCertArray = np.zeros((xdim,ydim,zdim))                    
            unCertList = x.readline().strip().split()
            m = 0
    
            for k in range(zdim):
                for j in range(ydim):
                        for i in range(xdim):
                                unCertArray[i,j,k] = unCertList[m]
                                m += 1                    
    
        
        doseObject = Dose([xdim, ydim, zdim], [xedgesList, yedgesList, zedgesList], [xcenterList, ycenterList, zcenterList], doseArray, unCertArray) 
        doseObj.append(doseObject)
    return doseObj
                
doseRect= readDose()
    
        
#%%Average the imported dose files

doseSum = np.zeros((doseRect[0].dimensions[0],doseRect[0].dimensions[1],doseRect[0].dimensions[2],0))
for i in range(len(doseRect)):
    doseObj = doseRect[i]
    doseSum = np.append(doseSum,np.expand_dims(doseObj.doseArray,axis=3),axis=3)
    
doseRect = doseRect[0]
doseRect._doseArray = np.mean(doseSum,axis=3) 
doseRect._errorArray =np.std(doseSum,axis=3)


#%% find the location 
location = [0,0,0]

for doseObj in [doseRect]:
    x = np.searchsorted(doseObj.centers[0],location[0])
    y = np.searchsorted(doseObj.centers[1],location[1])
    z = np.searchsorted(doseObj.centers[2],location[2])
    print(doseObj.doseArray[x,y,z])
    print((doseObj.errorArray[x,y,z]/doseObj.doseArray[x,y,z])*100)

doseIso = np.mean(doseObj.doseArray[x-1:x+2,y-1:y+2,z-1:z+2])
plt.plot(doseObj.centers[0][:],doseObj.doseArray[:,y,z])
dosePlane = doseObj.doseArray[:,y,z]
line = np.polynomial.Polynomial.fit(doseObj.centers[0][87:168], doseObj.doseArray[87:168,y,z], deg=1).convert()
b, m = line.coef