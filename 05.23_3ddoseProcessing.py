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

    
    
def read2Dose():    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = fd.askopenfilenames(parent=root)    
    doseObjects = []
    for path in file_paths:

        #Parses file line by line
        with open(path) as x:
   
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
            
            #Build dose image
            doseArray = np.zeros((xdim,ydim,zdim))
            for k in range(zdim):
                for j in range(ydim+1):
                    xRow = x.readline().strip()
                    if len(xRow) == xdim:
                        for i in range(xdim):
                            doseArray[i,j,k] = int(xRow[i])

            
            #Build density image
            errorArray = np.zeros((xdim,ydim,zdim))
            for k in range(zdim):
                for j in range(ydim+1):
                    xRow = x.readline().strip().split()
                    xRow = [float(item) for item in xRow]
                    if len(xRow) == xdim:
                        for i in range(xdim):
                           errorArray[i,j,k] = xRow[i]

    
        doseObjects.append(Dose([xdim, ydim, zdim], [xedgesList, yedgesList, zedgesList], [xcenterList, ycenterList, zcenterList], doseArray, errorArray)) 
    return doseObjects
                
[doseAP, dosePA]=read2Dose()        
        
#%%      
#Rotate and Sum egsphant files
AP_dose = doseAP.densityArray
PA_dose = dosePA.densityArray

#Firstly check dims and crop if an empty slice leads to a mismatch

if doseAP.dimensions[0] != dosePA.dimensions[0]:
    if doseAP.dimensions[0] > dosePA.dimensions[0]:
        if (doseAP.doseArray[0,:,:] == 1).all() :
            AP_dose = AP_dose[1:0,:,:]
        elif (doseAP.doseArray[-1,:,:] == 1).all():
            AP_dose = AP_dose[0:-1,:,:]
        else:
            sys.exit('AP x-dim issue')
    else:
        if (dosePA.doseArray[0,:,:] == 1).all() :
            PA_dose = PA_dose[1:0,:,:]
        elif (dosePA.doseArray[-1,:,:] == 1).all():
            PA_dose = PA_dose[0:-1,:,:]
        else:
            sys.exit('PA x-dim issue')           
if doseAP.dimensions[1] != dosePA.dimensions[1]:
    if doseAP.dimensions[1] > dosePA.dimensions[1]:
        if (doseAP.doseArray[:,0,:] == 1).all() :
            AP_dose = AP_dose[:,1:0,:]
        elif (doseAP.doseArray[:,-1,:] == 1).all():
            AP_dose = AP_dose[:,0:-1,:]
        else:
            sys.exit('AP y-dim issue')
    else:
        if (dosePA.doseArray[:,0,:] == 1).all() :
            PA_dose = PA_dose[:,1:0,:]
        elif (dosePA.doseArray[:,-1,:] == 1).all():
            PA_dose = PA_dose[:,0:-1,:]
        else:
            sys.exit('PA y-dim issue')                    
if doseAP.dimensions[2] != dosePA.dimensions[2]:
    if doseAP.dimensions[2] > dosePA.dimensions[2]:
        if (doseAP.doseArray[:,:,0] == 1).all() :
            AP_dose = AP_dose[:,:,1:0]
        elif (doseAP.doseArray[:,:,-1] == 1).all():
            AP_dose = AP_dose[:,:,0:-1]
        else:
            sys.exit('AP z-dim issue')
    else:
        if (dosePA.doseArray[:,:,0] == 1).all() :
            PA_dose = PA_dose[:,:,1:0]
        elif (dosePA.doseArray[:,:,-1] == 1).all():
            PA_dose = PA_dose[:,:,0:-1]
        else:
            sys.exit('PA z-dim issue')       

#Sum the arrays
Summed = AP_dose + sp.ndimage.rotate(PA_dose, 180)     
plt.imshow(Summed[:,:,6], cmap='hot', interpolation='nearest')
plt.show()
