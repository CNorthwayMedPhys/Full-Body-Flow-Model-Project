# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:23:11 2026

@author: Cassidy.Northway
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# %% Event Rate plot


cd = os.getcwd()
compartment_AP = "CompartmentData\\AP\\4eventTrace.npy"
eventTrace = np.load(os.path.join(cd,compartment_AP))

x= eventTrace[0,:90000]
y = eventTrace[1,:90000]

##########
from scipy.signal import savgol_filter
yhat = savgol_filter(y, 2500, 3) 
ind = 4372
fig=plt.figure()
ax=fig.subplots()
ax.plot(x,yhat, color='k',linewidth=0.7)
ax.get_xaxis().set_ticks([x[ind]])
ax.get_yaxis().set_ticks([yhat[ind]])
ax.tick_params(direction="in")
xlabels = [item.get_text() for item in ax.get_xticklabels()]
xlabels[0] = "t"
ax.set_xticklabels(xlabels)
ax.tick_params(axis='x', colors='red')

ylabels = [item.get_text() for item in ax.get_yticklabels()]
ylabels[0] = "ER(t)"
ax.set_yticklabels(ylabels, rotation=90)
ax.tick_params(axis='y', colors='red')

plt.xlabel('Time (s)')
plt.ylabel('Dose Rate (1/s)')
ax.xaxis.set_label_coords(.5, -0.03)
ax.yaxis.set_label_coords(-0.03, 0.45)
ax.axvline(x = x[ind], ymin = 0, ymax = 0.93, color = 'r',linestyle = "--")
ax.axhline(y = yhat[ind], xmin = 0, xmax =0.09, color ='r', linestyle = "--")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()


#%% DVH Plot

DVH_AP = "CompartmentData\\AP\\4DVH.npy"
DVH = np.load(os.path.join(cd,DVH_AP))
x = DVH[0,:]/100 
y = DVH[1,:]
ind = 400
fig=plt.figure()
ax=fig.subplots()
ax.tick_params(direction="in")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.plot(x,y,color='k')

plt.ylabel('Dose (Gy)')
ax.axhline(y = y[ind], xmin = 0, xmax= 0.38, color = 'red', linestyle = '--')
ax.axvline(x = x[ind], ymin = 0, ymax = 0.39, color = 'r',linestyle = "--")

plt.xlabel('Random Numbers in [0,1]')
plt.locator_params(axis='x', nbins=3)
ax.get_xaxis().set_ticks([0,x[ind],1])
xlabels = [item.get_text() for item in ax.get_xticklabels()]
xlabels[0] = 0
xlabels[1] = ""
xlabels[2] = 1
ax.set_xticklabels(xlabels)
plt.gca().get_xticklabels()[1].set_color("red")

ax.get_yaxis().set_ticks([y[ind]])
ylabels = [item.get_text() for item in ax.get_yticklabels()]
ylabels[0] = "D"
ax.set_yticklabels(ylabels)
ax.tick_params(axis='y', colors='red')

plt.show()


#%%Plot egs voi


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

    file_paths = os.path.join(cd,"XCAT_AP.egsphant")


    #Parses file line by line
    with open(file_paths ) as x:
        
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

def addVOI(egsphantObject,item):
    file_paths = os.path.join(cd,item)    
    
    #Parses file line by line
    with open(file_paths) as x:
        
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


egsphantObj=readEgsphant() 


#%%
#Go down row by row
phant = egsphantObj.materialArray
phant2d = np.zeros([264,697])
for i in range(264):
    for j in range(697):
        if any(phant[i,:,j] > 1):
            phant2d[i,j] = 3

voinames = [3,5,8,10] #darker grey
for items in voinames:     
    addVOI(egsphantObj,"CompartmentData//AP//UnStackedEgsvoi//" +str(items)+".egsvoi") 
    voi = egsphantObj.VOIArray
    for i in range(264):
        for j in range(697):
            if any(voi[i,:,j] > 0):
                phant2d[i,j] = 4 

voinames = [2,6,9,26,27] 
for items in voinames:     
    addVOI(egsphantObj,"CompartmentData//AP//UnStackedEgsvoi//" +str(items)+".egsvoi") 
    voi = egsphantObj.VOIArray
    for i in range(264):
        for j in range(697):
            if any(voi[i,:,j] > 0):
                phant2d[i,j] = 5 
                
addVOI(egsphantObj,"XCAT_AP_unstacked.egsvoi")              

voi = egsphantObj.VOIArray
for i in range(264):
    for j in range(697):
        if any(voi[i,:,j] > 0):
            phant2d[i,j] = 10
            
voinames = [0,1] 
for items in voinames:     
    addVOI(egsphantObj,"CompartmentData//AP//UnStackedEgsvoi//" +str(items)+".egsvoi") 
    voi = egsphantObj.VOIArray
    for i in range(264):
        for j in range(697):
            if any(voi[i,:,j] > 0):
                phant2d[i,j] = 7.5 


viridis = cm.get_cmap('binary', 256)
newcolors = viridis(np.linspace(0, 1, 256))
yellow = np.array([255/256, 255/256, 0/256, 1])
red = np.array([255/256, 0/256, 0/256, 1])
newcolors[125:130, :] = yellow
newcolors[190:194,:] =red
newcmp = ListedColormap(newcolors)

plt.figure()
plt.imshow(phant2d, cmap = newcmp)
plt.axis('off')
# plt.figure()         
# plt.imshow(egsphantObj.densityArray[:,:,250])   
#plt.figure()         
#plt.imshow(Summed[:,:,515], cmap='hot')  
# plt.figure()         
# plt.imshow(Summed[:,26,:], cmap='hot')  