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

egsphantfile = os.path.join(cd,'XCAT_AP.egsphant')
dosefile = os.path.join(cd,"TBI_XCAT_2E11.3ddose")



#%% FCN and class: Read egsphant and voi
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
    def edges(self):
        return self._edges
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

def readEgsphant(egsphantfile):    


    #Parses file line by line
    with open(egsphantfile) as x:
        
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
        # doseList = []
        # m=0
        # for z in range(ydim*zdim):
        #         tempList = x.readline().strip().split()
        #         doseList.extend(tempList)
        # for k in range(zdim):
        #     for j in range(ydim):
        #             for i in range(xdim):
        #                     doseArray[i,j,k] = doseList[m]
        #                     m += 1        
                
                
        # unCertArray = np.zeros((xdim,ydim,zdim))
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
        
  ####Normally use this for combined dose   #####   
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

#%%Begin file processing 
egsphant = readEgsphant(egsphantfile)
doseObj = readDose(dosefile)



#%%
#Firstly check dims and crop if an empty slice leads to a mismatch
[xedges,yedges,zedges] = doseObj.edges

xMidIndex = 128
yMidIndex = 50 
zMidIndex= 439 #based on ct looks good


doseIso = np.mean(doseObj.doseArray[xMidIndex-1:xMidIndex+2,yMidIndex-1:yMidIndex+2,zMidIndex-1:zMidIndex+2])
unCertIso= np.mean(doseObj.unCertArray[xMidIndex-1:xMidIndex+2,yMidIndex-1:yMidIndex+2,zMidIndex-1:zMidIndex+2])

dosePlane = doseObj.doseArray[:,yMidIndex,zMidIndex]



#%% 
Summed = egsphant.materialArray
Summed[:,yMidIndex,zMidIndex] = 8
Summed[xMidIndex, yMidIndex,:] = 8

plt.imshow(Summed[:,:,zMidIndex], cmap='hot')  
plt.scatter(x=yMidIndex, y=xMidIndex, c='b', s=10)
plt.show()


                   
#%% Do linear least squares for data        

x =  np.array(doseObj.centers[0][69:183])
y =  dosePlane[69:183]       

A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]

plt.plot(x, y, 'b.', label='Data points')
plt.plot(x, m*x + c, 'r', label='Best fit line')
plt.show()    
    

