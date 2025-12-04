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
class egsphant:
    
    def __init__(self, nmaterials, materialList, dimensions, edges, centers, materialArray, densityArray, VOIArray):
        self._nmaterials = nmaterials
        self._materialList = materialList
        self._dimensions = dimensions
        self._edges = edges
        self._centers = centers
        self._materialArray = materialArray
        self._densityArray = densityArray
        self._VOIArray = VOIArray
        
    @property
    def dimensions(self):
        return self._dimensions
    @property
    def materialArray(self):
        return self._materialArray
    @property
    def densityArray(self):
        return self._densityArray
    @property
    def VOIArray(self):
        return self._VOIArray
    @VOIArray.setter
    def VOIArray(self,value):
        self._VOIArray = value
    
def readEgsphant():    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)
    file_paths = fd.askopenfilenames(parent=root, title = 'Select egsphant file')    
    for path in file_paths:

        #Parses file line by line
        with open(path) as x:
            
            #Number of materials
            nmaterials = int(x.readline().strip())
            
            #List materials
            materialList = []
            for i in range(nmaterials):
                material = x.readline().strip()
                materialList.append(material)
                
            #Placeholder values
            estepe = x.readline().strip()
                 
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
            
            #Build material image
            materialArray = np.zeros((xdim,ydim,zdim))
            for k in range(zdim):
                for j in range(ydim+1):
                    xRow = x.readline().strip()
                    if len(xRow) == xdim:
                        for i in range(xdim):
                            materialArray[i,j,k] = int(xRow[i])
                    else:
                        m = 0
                        for i in range(len(xRow)):
                            try:
                                xValue = int(xRow[i])
                                if int(xRow[i]) in range(1,nmaterials):
                                    materialArray[m,j,k] = int(xRow[i])
                                    m += 1
                            except:
                                pass
            
            #Build density image
            densityArray = np.zeros((xdim,ydim,zdim))
            for k in range(zdim):
                for j in range(ydim+1):
                    xRow = x.readline().strip().split()
                    xRow = [float(item) for item in xRow]
                    if len(xRow) == xdim:
                        for i in range(xdim):
                            densityArray[i,j,k] = xRow[i]

    
        egsphantObject=egsphant(nmaterials, materialList, [xdim, ydim, zdim], [xedgesList, yedgesList, zedgesList], [xcenterList, ycenterList, zcenterList], materialArray, densityArray, []) 
    return egsphantObject

def addVOI(egsphantObject):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)
    file_paths = fd.askopenfilenames(parent=root, title = 'Select egsvoi file')    
    for path in file_paths:
        #Parses file line by line
        with open(path) as x:
            
            #Number of voxels
            [edepOption, numVoxels] = x.readline().split()
            
            #Produce array of voxels
            voxelIndices = []
            for i in range(int(numVoxels)):
                voxelIndices.append(int(x.readline().strip()))
                
            #Confirm size is the same
            if int(numVoxels) == len(voxelIndices):
                print('All voxels counted')
                
            #Create the empty array
            [xdim,ydim,zdim] = egsphantObj.dimensions
            VOIArray = np.zeros((xdim,ydim,zdim))
            
            
            for lin_ind in voxelIndices:
                
                x_ind = int((lin_ind-1) % xdim)
                y_ind = int(((lin_ind-1) //xdim) % ydim)
                z_ind = int(((lin_ind-1)//xdim) // ydim)
                VOIArray[x_ind,y_ind,z_ind] = 3
            # for k in range(zdim):
            #     for j in range(ydim):
            #          for i in range(xdim):
            #              if tag == 0:
            #                  if ind == voxelIndices[vInd]:
            #                      VOIArray[i,j,k] = 3
            #                      vInd += 1
            #                      if vInd == len(voxelIndices):
            #                          tag = 1
            #              ind += 1
                         
            egsphantObj.VOIArray = VOIArray

#%%
egsphantObj=readEgsphant() 

#addVOI(egsphantObj)  
#%% 
#Summed = egsphantObj.densityArray + egsphantObj.VOIArray 
#index = np.where(Summed > 100)
# plt.figure()         
# plt.imshow(egsphantObj.densityArray[:,:,250])   
#plt.figure()         
#plt.imshow(Summed[:,:,500], cmap='hot')  
# plt.figure()         
# plt.imshow(Summed[:,26,:], cmap='hot')  
     
        
