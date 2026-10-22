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
    
    def __init__(self, nmaterials, materialList, dimensions, edges, centers, materialArray, densityArray):
        self._nmaterials = nmaterials
        self._materialList = materialList
        self._dimensions = dimensions
        self._edges = edges
        self._centers = centers
        self._materialArray = materialArray
        self._densityArray = densityArray
    
    @property
    def dimensions(self):
        return self._dimensions
    @property
    def materialArray(self):
        return self._materialArray
    @property
    def densityArray(self):
        return self._densityArray
    
    
def read2Egsphant():    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = fd.askopenfilenames(parent=root)    
    egsphantObjects = []
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

    
        egsphantObjects.append(egsphant(nmaterials, materialList, [xdim, ydim, zdim], [xedgesList, yedgesList, zedgesList], [xcenterList, ycenterList, zcenterList], materialArray, densityArray)) 
    return egsphantObjects
                
[egsphantAP, egsphantPA]=read2Egsphant()        
        
#%%      
#Rotate and Sum egsphant files
AP_density = egsphantAP.densityArray
PA_density = egsphantPA.densityArray

#Firstly check dims and crop if an empty slice leads to a mismatch

if egsphantAP.dimensions[0] != egsphantPA.dimensions[0]:
    if egsphantAP.dimensions[0] > egsphantPA.dimensions[0]:
        if (egsphantAP.materialArray[0,:,:] == 1).all() :
            AP_density = AP_density[1:0,:,:]
        elif (egsphantAP.materialArray[-1,:,:] == 1).all():
            AP_density = AP_density[0:-1,:,:]
        else:
            sys.exit('AP x-dim issue')
    else:
        if (egsphantPA.materialArray[0,:,:] == 1).all() :
            PA_density = PA_density[1:0,:,:]
        elif (egsphantPA.materialArray[-1,:,:] == 1).all():
            PA_density = PA_density[0:-1,:,:]
        else:
            sys.exit('PA x-dim issue')           
if egsphantAP.dimensions[1] != egsphantPA.dimensions[1]:
    if egsphantAP.dimensions[1] > egsphantPA.dimensions[1]:
        if (egsphantAP.materialArray[:,0,:] == 1).all() :
            AP_density = AP_density[:,1:0,:]
        elif (egsphantAP.materialArray[:,-1,:] == 1).all():
            AP_density = AP_density[:,0:-1,:]
        else:
            sys.exit('AP y-dim issue')
    else:
        if (egsphantPA.materialArray[:,0,:] == 1).all() :
            PA_density = PA_density[:,1:0,:]
        elif (egsphantPA.materialArray[:,-1,:] == 1).all():
            PA_density = PA_density[:,0:-1,:]
        else:
            sys.exit('PA y-dim issue')                    
if egsphantAP.dimensions[2] != egsphantPA.dimensions[2]:
    if egsphantAP.dimensions[2] > egsphantPA.dimensions[2]:
        if (egsphantAP.materialArray[:,:,0] == 1).all() :
            AP_density = AP_density[:,:,1:0]
        elif (egsphantAP.materialArray[:,:,-1] == 1).all():
            AP_density = AP_density[:,:,0:-1]
        else:
            sys.exit('AP z-dim issue')
    else:
        if (egsphantPA.materialArray[:,:,0] == 1).all() :
            PA_density = PA_density[:,:,1:0]
        elif (egsphantPA.materialArray[:,:,-1] == 1).all():
            PA_density = PA_density[:,:,0:-1]
        else:
            sys.exit('PA z-dim issue')       

#Sum the arrays
Summed = AP_density + sp.ndimage.rotate(PA_density, 180)     
plt.imshow(Summed[:,:,6], cmap='hot', interpolation='nearest')
plt.show()
