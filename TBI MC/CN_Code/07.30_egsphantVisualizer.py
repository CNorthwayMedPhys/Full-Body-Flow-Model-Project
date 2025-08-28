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
    @property
    def edges(self):
        return self._edges
    
def readEgsphant():    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    path = fd.askopenfilename(parent=root)    
    
    

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
        tag = 0
        while tag == 0:
            xedges = x.readline().split()
            xedges = [float(item) for item in xedges]
            xedgesList = np.append(xedgesList,xedges)
            if min(xedges) > 0:
                tag = 1
        while tag == 1:
            xedges = x.readline().split()
            xedges = [float(item) for item in xedges]
            if max(xedges) < 0:
                yedgesList = np.append(yedgesList,xedges)
                tag = 2
            else:
                xedgesList = np.append(xedgesList,xedges)
        while tag == 2:
            yedges = x.readline().split()
            yedges = [float(item) for item in yedges]
            yedgesList = np.append(yedgesList,yedges)
            if min(yedges) > 0:
                tag = 3    
        while tag == 3:
            yedges = x.readline().split()
            yedges = [float(item) for item in yedges]
            if max(yedges) < 0:
                zedgesList = np.append(zedgesList,yedges)
                tag = 4
            else:
                yedgesList = np.append(yedgesList,yedges)
        while tag == 4:
            zedges = x.readline().split()
            zedges = [float(item) for item in zedges]
            if len(zedges) > 1:
                zedgesList = np.append(zedgesList,zedges)
            else:
                tag = 9
                currentposition = x.tell()
                
        
        
        
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
        x.seek(currentposition-xdim)
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

#%%                
def plotMaterials(egsphant,sliceDim,sliceLocation):
    if sliceDim == 'z':
        array = egsphant.materialArray[:,:,sliceLocation]
        [xedges, yedges, zedges] = egsphant.edges
        plottingArray = np.zeros([5,egsphant.dimensions[1]*egsphant.dimensions[0]])
        k= 0
        for j in range(egsphant.dimensions[1]):
            for i in range(egsphant.dimensions[0]):
                plottingArray[0,k] = array[i,j]
                plottingArray[1,k] = xedges[i]
                plottingArray[2,k] = xedges[i + 1]
                plottingArray[3,k] = yedges[j]
                plottingArray[4,k] = yedges[j + 1]
                k += 1
    
        
        body = plottingArray[:, np.where(plottingArray[0,:] == 3)]
        plastic = plottingArray[:, np.where(plottingArray[0,:] == 8)]
        plastic = np.append(plastic,plottingArray[:, np.where(plottingArray[0,:] == 6)], axis = 2)
        lead = plottingArray[:, np.where(plottingArray[0,:] == 9)]
        
    
        plt.scatter(body[1:3,:], body[3:5], c= "blue")
        plt.scatter(plastic[1:3,:], plastic[3:5], c= "red")
        plt.scatter(lead[1:3,:], lead[3:5], c= "green")
     
                
#%%
def measureDist(egsphant, sliceDim, sliceLocation, lineLocation):
    if sliceDim == 'z':
        array = egsphant.materialArray[:,:,sliceLocation]
        [xedges, yedges, zedges] = egsphant.edges
        plottingArray = np.zeros([5,egsphant.dimensions[1]*egsphant.dimensions[0]])
        k= 0
        for j in range(egsphant.dimensions[1]):
            for i in range(egsphant.dimensions[0]):
                plottingArray[0,k] = array[i,j]
                plottingArray[1,k] = xedges[i]
                plottingArray[2,k] = xedges[i + 1]
                plottingArray[3,k] = yedges[j]
                plottingArray[4,k] = yedges[j + 1]
                k += 1
    
        
        body = np.squeeze(plottingArray[:, np.where(plottingArray[0,:] == 3)])
        plastic = plottingArray[:, np.where(plottingArray[0,:] == 8)]
        plastic = np.squeeze(np.append(plastic,plottingArray[:, np.where(plottingArray[0,:] == 6)], axis = 2))
        lead = np.squeeze(plottingArray[:, np.where(plottingArray[0,:] == 9)])
        
        body_line = np.squeeze(body[:,np.where(body[1,:] == lineLocation)])
        plastic_line = np.squeeze(plastic[:,np.where(plastic[1,:] == lineLocation)])
        lead_line = np.squeeze(lead[:,np.where(lead[1,:] == lineLocation)])
        
        body_ant = np.min(body_line[4,:])
        plastic_tray_ant = plastic_line[4,-1]
        plastic_filter_post = np.max(plastic_line[4,:])
        lead_ant = lead_line[4]

  
    
        plt.figure
        plt.scatter(body_line[1,:],body_line[3,:])
        plt.scatter(plastic_line[1,:],plastic_line[3,:])
        
        print("body ant: " + str(body_ant))
        print('tray ant: ' + str(plastic_tray_ant))
        print('filter post: ' + str(plastic_filter_post))
        print('lead ant: ' + str(lead_ant))
        
            
        
    
    
        
        
#%%      
egsphant = readEgsphant()

#%%
plotMaterials(egsphant,'z',204)
measureDist(egsphant, 'z', 204, -4.7100000)

