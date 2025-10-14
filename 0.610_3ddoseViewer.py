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
#%%

#Define classes for handling files
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
    def centers(self):
        return self._centers
    @property
    def errorArray(self):
        return self._errorArray
    
    
def readDose():    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = fd.askopenfilename(parent=root)    
    doseObjects = []
    

    #Parses file line by line
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
    return doseObject
                
dose= readDose()        
        

#%% Dose we want to locate
x = 0
y = 9.99999999
z = 0

x_ind = dose.centers[0].index(x)
y_ind = dose.centers[1].index(y)
z_ind = dose.centers[2].index(z)

value = dose.doseArray[x_ind,y_ind,z_ind]
error = dose.errorArray[x_ind,y_ind,z_ind]

#%% Look for values in the range around the index point