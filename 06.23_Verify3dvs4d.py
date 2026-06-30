# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:30:06 2026

@author: Cassidy.Northway
"""

import os
import numpy as np
import tkinter as tk
import tkinter.filedialog as fd
import scipy as sp

root = tk.Tk()
root.lift()
root.attributes('-topmost',True)
root.after_idle(root.attributes,'-topmost',False)


#%% FCN:Read 3ddose file
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
    file_path = fd.askopenfilename(parent=root, filetypes =[ ("3DDose Files", '*.3ddose')])    
    
    

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
                
#%%Load in the data
#Load in 4ddose for vessels for an iso
cd = os.getcwd()
root = tk.Tk()
root.withdraw()  # Hide the main window
file_path = fd.askopenfilename(parent=root, filetypes =[ ("4DDose Files", '*.npy')])
dose4D = np.load(file_path)
  
#Load in 3ddose file for an iso
dose3D= readDose()
[xdimD,ydimD,zdimD] = dose3D.dimensions

#%%
#Select a VOI index
for  i in range(500,38844):
    voiIndex = int(dose4D[0,i])
    #Find 3ddose at location
    x_indD = int((voiIndex-1) % xdimD)
    y_indD = int(((voiIndex-1) //xdimD) % ydimD)
    z_indD = int(((voiIndex-1)//xdimD) // ydimD)
    
    if  np.sum(dose4D[1:,i]) != 0 and dose3D.doseArray[x_indD,y_indD,z_indD] != 0 :
        print(np.sum(dose4D[1:,i]))
        x = dose4D[1:,i]
        #Find 3ddose at location
        x_indD = int((voiIndex-1) % xdimD)
        y_indD = int(((voiIndex-1) //xdimD) % ydimD)
        z_indD = int(((voiIndex-1)//xdimD) // ydimD)
        
        
        print("dose to VOI: " + str(dose3D.doseArray[x_indD,y_indD,z_indD]))
        
        break
#Sum 4ddose at location

#Compare values