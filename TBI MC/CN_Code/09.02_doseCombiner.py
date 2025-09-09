# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:07:20 2025

@author: Cassidy.Northway
based on https://github.com/prehensilecode/egsnrcpy/blob/main/egsnrc/_egsphant.c#L164 
and https://discourse.itk.org/t/resample-volume-to-specific-voxel-spacing-simpleitk/3531
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.filedialog as fd
import scipy as sp

#%%

#Define classes for handling files
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
    def centers(self):
        return self._centers
    @property
    def doseArray(self):
        return self._doseArray
    @property
    def edges(self):
        return self._edges  
    @edges.setter
    def edges(self,a):
        self._edges=a
    @property
    def unCertArray(self):
        return self._unCertArray
    
    
def readDose():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    path = fd.askopenfilename(parent=root)    
    
    

    #Parses file line by line
    with open(path) as x:
    
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
        for k in range(zdim):
            for j in range(ydim):
                xRow = x.readline().strip().split()
                if len(xRow) == xdim:
                    for i in range(xdim):
                        doseArray[i,j,k] = xRow[i]

        unCertArray = np.zeros((xdim,ydim,zdim))                    
        for k in range(zdim):
            for j in range(ydim):
                xRow = x.readline().strip().split()
                for i in range(xdim):
                    unCertArray[i,j,k] = xRow[i]

        
        


        doseObject = dose([xdim, ydim, zdim], [xedgesList, yedgesList, zedgesList], [xcenterList, ycenterList, zcenterList], doseArray,unCertArray) 
    return doseObject

def getGrid(doseObj):
    x = doseObj.edges[0]
    y = doseObj.edges[1]
    z = doseObj.edges[2]
    
    xGrid, yGrid, zGrid = np.meshgrid(x,y,z)
    return [xGrid.astype(float), yGrid.astype(float), zGrid.astype(float)]
    

        
#%%
doseObjAP = readDose()
doseObjPA = readDose()      


#First rotate the PA data to an AP orientation (no name change)
PA_dose = sp.ndimage.rotate(doseObjPA.doseArray, 180)
plt.imshow(PA_dose[:,:,57])
plt.show()
plt.imshow(doseObjAP.doseArray[:,:,57])
plt.show()
PA_centers = doseObjPA.centers
newX = PA_centers[0][::-1]
newX = [i * -1 for i in newX]
newY = PA_centers[1][::-1]
newY = [i * -1 for i in newY]


points = (newX, newY, PA_centers[2])
values = PA_dose
interpolating_function = sp.interpolate.RegularGridInterpolator(points, values, method='linear', bounds_error = False, fill_value = 0)

grid_points = np.array(np.meshgrid(doseObjAP.centers[0], doseObjAP.centers[1], doseObjAP.centers[2], indexing='ij')).reshape(3, -1).T
PA_dose_resampled = interpolating_function(grid_points)
PA_dose_resampled = np.reshape(PA_dose_resampled, doseObjAP.dimensions)

PA_dose_resampled =sp.interpolate.interpn(points, values, grid_points,  bounds_error = False, fill_value = 0)
    
# #Rotate and Sum egsphant files
# AP_density = egsphantAP.densityArray
# PA_density = egsphantPA.densityArray

# #Firstly check dims and crop if an empty slice leads to a mismatch

# if egsphantAP.dimensions[0] != egsphantPA.dimensions[0]:
#     if egsphantAP.dimensions[0] > egsphantPA.dimensions[0]:
#         if (egsphantAP.materialArray[0,:,:] == 1).all() :
#             AP_density = AP_density[1:0,:,:]
#         elif (egsphantAP.materialArray[-1,:,:] == 1).all():
#             AP_density = AP_density[0:-1,:,:]
#         else:
#             sys.exit('AP x-dim issue')
#     else:
#         if (egsphantPA.materialArray[0,:,:] == 1).all() :
#             PA_density = PA_density[1:0,:,:]
#         elif (egsphantPA.materialArray[-1,:,:] == 1).all():
#             PA_density = PA_density[0:-1,:,:]
#         else:
#             sys.exit('PA x-dim issue')           
# if egsphantAP.dimensions[1] != egsphantPA.dimensions[1]:
#     if egsphantAP.dimensions[1] > egsphantPA.dimensions[1]:
#         if (egsphantAP.materialArray[:,0,:] == 1).all() :
#             AP_density = AP_density[:,1:0,:]
#         elif (egsphantAP.materialArray[:,-1,:] == 1).all():
#             AP_density = AP_density[:,0:-1,:]
#         else:
#             sys.exit('AP y-dim issue')
#     else:
#         if (egsphantPA.materialArray[:,0,:] == 1).all() :
#             PA_density = PA_density[:,1:0,:]
#         elif (egsphantPA.materialArray[:,-1,:] == 1).all():
#             PA_density = PA_density[:,0:-1,:]
#         else:
#             sys.exit('PA y-dim issue')                    
# if egsphantAP.dimensions[2] != egsphantPA.dimensions[2]:
#     if egsphantAP.dimensions[2] > egsphantPA.dimensions[2]:
#         if (egsphantAP.materialArray[:,:,0] == 1).all() :
#             AP_density = AP_density[:,:,1:0]
#         elif (egsphantAP.materialArray[:,:,-1] == 1).all():
#             AP_density = AP_density[:,:,0:-1]
#         else:
#             sys.exit('AP z-dim issue')
#     else:
#         if (egsphantPA.materialArray[:,:,0] == 1).all() :
#             PA_density = PA_density[:,:,1:0]
#         elif (egsphantPA.materialArray[:,:,-1] == 1).all():
#             PA_density = PA_density[:,:,0:-1]
#         else:
#             sys.exit('PA z-dim issue')       

# #Sum the arrays
# Summed = AP_density + sp.ndimage.rotate(PA_density, 180)     
# plt.imshow(Summed[:,:,6], cmap='hot', interpolation='nearest')
# plt.show()

Summed = doseObjAP.doseArray[:,:,57] + PA_dose[:,:,57]
plt.imshow(Summed, cmap='hot', interpolation='nearest')