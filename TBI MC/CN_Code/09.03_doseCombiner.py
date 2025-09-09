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
fname = "TBI_XCAT_Full_Combined.3ddose"
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
    @property
    def edges(self):
        return self._edges
    
    
def readEgsphant():    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = fd.askopenfilename(parent=root)    

    #Parses file line by line
    with open(file_path) as x:
        
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


    egsphantObject = egsphant(nmaterials, materialList, [xdim, ydim, zdim], [xedgesList, yedgesList, zedgesList], [xcenterList, ycenterList, zcenterList], materialArray, densityArray) 
    return egsphantObject
def writeTrimmedDose(fname, doseArray, uncertArray, xedges, yedges,zedges):
    with open(fname, "w") as fp:

        fp.write(f"{len(xedges) - 1:5d}{len(yedges) - 1:5d}{len(zedges) - 1:5d}\n")

        for i in range(len(xedges)):
            edges= xedges
            fp.write(f"{edges[i]:7.5f} ")

        fp.write("\n")

        for i in range(len(yedges)): 
            edges= yedges
            fp.write(f"{edges[i]:7.5f} ")

        fp.write("\n")

        for i in range(len(zedges)):
            edges= zedges
            fp.write(f"{edges[i]:7.5f} ")

        fp.write("\n")

       

        for k in range(len(zedges)-1):  
            for j in range(len(yedges)-1):
                for i in range(len(xedges)-1):
                     fp.write(f"{doseArray[i, j, k]:9.7E} ")
           

        fp.write("\n")


        n = 0
        # UNCERTS
        for k in range(len(zedges)-1):  
            for j in range(len(yedges)-1):
                for i in range(len(xedges)-1):
                     fp.write(f"{uncertArray[ i,  j, k ]:.8f} ")
                     n += 1 
     
        fp.write("\n")


    return                
       
#%%
egsphantAP=readEgsphant()
doseObjAP = readDose()

egsphantPA = readEgsphant()
doseObjPA = readDose()      

AP_dose = doseObjAP.doseArray
PA_dose = doseObjPA.doseArray

AP_uncert=doseObjAP.unCertArray
PA_uncert=doseObjPA.unCertArray

#Firstly check dims and crop if an empty slice leads to a mismatch
[xedges,yedges,zedges] = egsphantAP.edges

if egsphantAP.dimensions[0] != egsphantPA.dimensions[0]:
    if egsphantAP.dimensions[0] > egsphantPA.dimensions[0]:
        if (egsphantAP.materialArray[0,:,:] == 1).all() :
            AP_dose = AP_dose[1:0,:,:]
            AP_uncert = AP_uncert[1:0,:,:]
            xedges = egsphantAP.edges[0][1:0]
        elif (egsphantAP.materialArray[-1,:,:] == 1).all():
            AP_dose = AP_dose[0:-1,:,:]
            AP_uncert = AP_uncert[0:-1,:,:]
            xedges = egsphantAP.edges[0][0:-1]
        else:
            sys.exit('AP x-dim issue')
    else:
        if (egsphantPA.materialArray[0,:,:] == 1).all() :
            PA_dose = PA_dose[1:0,:,:]
            PA_uncert = PA_uncert[1:0,:,:]
            xedges = egsphantAP.edges[0]
        elif (egsphantPA.materialArray[-1,:,:] == 1).all():
            PA_dose = PA_dose[0:-1,:,:]
            PA_uncert = PA_uncert[0:-1,:,:]
            xedges = egsphantAP.edges[0]
        else:
            sys.exit('PA x-dim issue')           
if egsphantAP.dimensions[1] != egsphantPA.dimensions[1]:
    if egsphantAP.dimensions[1] > egsphantPA.dimensions[1]:
        if (egsphantAP.materialArray[:,0,:] == 1).all() :
            AP_dose = AP_dose[:,1:0,:]
            AP_uncert = AP_uncert[:,1:0,:]
            yedges = egsphantAP.edges[1][1:0]
        elif (egsphantAP.materialArray[:,-1,:] == 1).all():
            AP_dose = AP_dose[:,0:-1,:]
            AP_uncert = AP_uncert[:,0:-1,:]
            yedges = egsphantAP.edges[1][0:-1]
        else:
            sys.exit('AP y-dim issue')
    else:
        if (egsphantPA.materialArray[:,0,:] == 1).all() :
            PA_dose = PA_dose[:,1:0,:]
            PA_uncert = PA_uncert[:,1:0,:]
            yedges = egsphantAP.edges[1]
        elif (egsphantPA.materialArray[:,-1,:] == 1).all():
            PA_dose = PA_dose[:,0:-1,:]
            PA_uncert = PA_uncert[:,0:-1,:]
            yedges = egsphantAP.edges[1]
        else:
            sys.exit('PA y-dim issue')                    
if egsphantAP.dimensions[2] != egsphantPA.dimensions[2]:
    if egsphantAP.dimensions[2] > egsphantPA.dimensions[2]:
        if (egsphantAP.materialArray[:,:,0] == 1).all() :
            AP_dose = AP_dose[:,:,1:0]
            AP_uncert = AP_uncert[:,:,1:0]
            zedges = egsphantAP.edges[2][1:0]
        elif (egsphantAP.materialArray[:,:,-1] == 1).all():
            AP_dose = AP_dose[:,:,0:-1]
            AP_uncert = AP_uncert[:,:,0:-1]
            zedges = egsphantAP.edges[2][0:-1]
        else:
            sys.exit('AP z-dim issue')
    else:
        if (egsphantPA.materialArray[:,:,0] == 1).all() :
            PA_dose = PA_dose[:,:,1:0]
            PA_uncert = PA_uncert[:,:,1:0]
            zedges = egsphantAP.edges[2]
        elif (egsphantPA.materialArray[:,:,-1] == 1).all():
            PA_dose = PA_dose[:,:,0:-1]
            PA_uncert = PA_uncert[:,:,0:-1]
            zedges = egsphantAP.edges[2]
        else:
            sys.exit('PA z-dim issue')       

#Sum the arrays
Summed_dose = AP_dose + sp.ndimage.rotate(PA_dose, 180)     
Summed_uncert = np.sqrt(AP_uncert**2 + sp.ndimage.rotate(PA_uncert, 180)**2 )  

#Write to new 3DDose File
writeTrimmedDose(fname, Summed_dose, Summed_uncert, xedges, yedges,zedges)
