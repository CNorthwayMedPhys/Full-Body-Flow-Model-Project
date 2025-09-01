# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 10:21:48 2025
3D Dose Trimmer base don work by Levi Burns
@author: cbnor
"""


import numpy as np
import tkinter as tk
import tkinter.filedialog as fd

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
    @property
    def edges(self):
        return self._edges

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
    def edges(self):
        return self._edges  
    @property
    def unCertArray(self):
        return self._unCertArray
#%% Define read and write functions    
def readEgsphant():    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    path = fd.askopenfilename(parent=root, title = 'Select Egsphant file')    
    
    

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
        
        #Read voxel edges, not in one consitent line in the complete version so this looks for 
        #lines going from pos to negative
        xedgesList = []
        yedgesList = []
        zedgesList = []
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
        
        #Reset position 
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

def readDose():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    path = fd.askopenfilename(parent=root, title = 'Select 3ddose file')    
    
    

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
        doseList = x.readline().strip().split()
        m = 0

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

        
        


        doseObject = dose([xdim, ydim, zdim], [xedgesList, yedgesList, zedgesList], [xcenterList, ycenterList, zcenterList], doseArray,unCertArray) 
    return doseObject

def writeTrimmedDose(fname, doseObject, egsphantObject):
    with open(fname, "w") as fp:
        print(f"\n Opened file {fname}")

        fp.write(f"{egsphantObject.dimensions[0]:5d}{egsphantObject.dimensions[1]:5d}{egsphantObject.dimensions[2]:5d}\n")

        for i in range(egsphantObject.dimensions[0]+1):
            edges= egsphantObject.edges[0]
            fp.write(f"{edges[i]:7.5f} ")

        fp.write("\n")

        for i in range(egsphantObject.dimensions[1]+1): 
            edges= egsphantObject.edges[1]
            fp.write(f"{edges[i]:7.5f} ")

        fp.write("\n")

        for i in range(egsphantObject.dimensions[2]+1):
            edges= egsphantObject.edges[2]
            fp.write(f"{edges[i]:7.5f} ")

        fp.write("\n")

        zdiff = doseObject.dimensions[2] - egsphantObject.dimensions[2]
        n= 0
        m = 0
        for k in range(zdiff, egsphantObject.dimensions[2]+ zdiff-1):  
            for j in range(21, egsphantObject.dimensions[1] + 20): #21 was based on the number of added filter slices
                for i in range(egsphantObject.dimensions[0]):
                     fp.write(f"{doseObject.doseArray[i, j, k]:9.7E} ")
           

        fp.write("\n")



        # UNCERTS
        for k in range(zdiff, egsphantObject.dimensions[2]+ zdiff-1):  
            for j in range(21, egsphantObject.dimensions[1]+20): 
                for i in range(egsphantObject.dimensions[0]):
                     fp.write(f"{doseObject.unCertArray[ i,  j, k ]:.8f} ")
           
        fp.write("\n")


    return 
#%%
fname = 'TBI_XCAT_Full_AP_trimmed.3ddose'
doseObject = readDose()
egsphantObject = readEgsphant()
writeTrimmedDose(fname, doseObject, egsphantObject)


    