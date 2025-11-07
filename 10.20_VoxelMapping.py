"""
Oct 20 2025
Cassidy Northway

Intention: Create array for every vessel were we have |voxel index | xmin | xmax| 
    and write that to a excel sheet, the xmin and xmax belong to [0,L] 
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#HARD CODED THE VOXEL SIZE AS 0.5,0.5,0.5 cm
#HARD CODED the VOXEL ARRAY ORIGIN -32.21, -15.37, -110.5 cm
#%%Define classes for egsphant handling files and the fcn for loading them in
class egsphant:
    
    def __init__(self, nmaterials, materialList, dimensions, edges, centers, materialArray, densityArray, voxelIndex):
        self._nmaterials = nmaterials
        self._materialList = materialList
        self._dimensions = dimensions
        self._edges = edges
        self._centers = centers
        self._materialArray = materialArray
        self._densityArray = densityArray
        self._voxelIndex = voxelIndex
    
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
    def voxelIndex(self):
        return self._voxelIndex
    
def readEgsphant(path):    
    

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

        #Create voxel index
        voxelIndex = np.zeros((xdim,ydim,zdim))
        index = 1
        for k in range(zdim):
             for j in range(ydim):
                  for i in range(xdim):
                      voxelIndex[i,j,k] = index
                      index += 1         
        egsphantObjects = egsphant(nmaterials, materialList, [xdim, ydim, zdim], [xedgesList, yedgesList, zedgesList], [xcenterList, ycenterList, zcenterList], materialArray, densityArray, voxelIndex) 
    return egsphantObjects
                

#%%Load in the flow tracker data
dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path + "\\FlowTracker.xlsx"
df = pd.read_excel(path)

#%%Load in the relevant egsphant
path = dir_path + "\\TBI_XCAT_Full_AP.egsphant"
egsphant = readEgsphant(path)

#%% Iterate for all vessel numbers
#for vesselNum in range(28,126+1)
vesselNum = 28
#%% Load in the length array from the excel document
path = dir_path + '\\BVSimulationFiles\\'+ str(vesselNum) + '.xlsx'
flowdf = pd.read_excel(path, header = None)
flowdata = flowdf.to_numpy()
length_array = flowdata[0,1:] * 1000 # m*1000 = mm

#%% Load in the radius and central axis data
vesselName = df.loc[vesselNum, 'NAME']
path = dir_path + '\\FittedVesselSegments\\'+ vesselName + '.npy' 
vesselArray = np.load(path) #[x,y,z,r] mm

#%% Create an array that is the total distance elapsed between n and n+1 vessel
distanceArray = np.zeros(np.shape(vesselArray)[0])
for i in range(np.shape(vesselArray)[0]-1):
    a = vesselArray[i,0:3]
    b = vesselArray[i+1,0:3]
    distanceArray[i+1] = np.linalg.norm(a-b) + distanceArray[i] #mm
distanceArray = np.array([distanceArray]).T 
  
#Append the distance array to the vesselArray  [x,y,z,r,d] mm
vesselArray = np.append(vesselArray,distanceArray, axis = 1)    
#%% Create an index array which pairs  [0,L] and [position_i, position_f] by 
#binning distance data 
# bin_edges = np.histogram_bin_edges(vesselArray[:,4], bins = len(length_array)-1) 
# inds = np.digitize(vesselArray[:,4], bin_edges)

#%% Map positions in orginal vessels location to finalized egsphant coords
transition_map = [(1.6*10)-8.0, (-1.1*10)+5.7, (51.5*10) - 553.7]
vesselArray[:,0:3] = vesselArray[:,0:3] + transition_map

#%% Determine which positions are in which voxel
#HARD CODED THE VOXEL SIZE AS 0.5,0.5,0.5 cm
#HARD CODED the VOXEL ARRAY ORIGIN -32.21, -15.37, -110.5 cm
voxelArray = np.zeros(np.shape(vesselArray)[0])
origin = np.array ([-32.21,-15.37,-110.5]) * 10 #mm
size = np. array ([0.5,0.5,0.5]) * 10 # mm
for i in range(np.shape(vesselArray)[0]):
    position = vesselArray[i,0:3]
    x = int(np.floor((position[0]-origin[0])/size[0])) 
    y = int(np.floor((position[1]-origin[1])/size[1]))
    z = int(np.floor((position[2]-origin[2])/size[2]))
    voxelArray[i] = egsphant.voxelIndex[x,y,z]
    
voxelArray = np.array([voxelArray]).T
vesselArray = np.append(vesselArray ,voxelArray, axis = 1  )    


#%%To check if translation of array actually aligns everything properly or not.
#Create the empty array
[xdim,ydim,zdim] = egsphant.dimensions
VOIArray = np.zeros((xdim,ydim,zdim))

#Start filling array
ind = 1
vInd = 0
tag = 0

for lin_ind in voxelArray:
    lin_ind = int(lin_ind)
    x_ind = int((lin_ind-1) % xdim)
    y_ind = int(((lin_ind-1) //xdim) % ydim)
    z_ind = int(((lin_ind-1)//xdim) // ydim)
    VOIArray[x_ind,y_ind,z_ind] = 10

Summed = egsphant.materialArray + VOIArray  
index = np.where(Summed > 100)             
plt.imshow(Summed[:,:,250], cmap='hot')  
# #%%Find the transition points from voxel to voxel
# transition_index = [0]
# for i in range(np.shape(vesselArray)[0]-1):
#     a = vesselArray[i,5]
#     b = vesselArray[i+1,5]
#     if a-b != 0:
#         transition_index.append(i+1)

#     dist_voxel_match = np.vstack([vesselArray[transition_index,4], vesselArray[transition_index,5]]).T    
         
# #%%Write to an excel document
# np.save(dir_path + '\\VOIMappingArrays\\' + str(vesselNum) + '.npy')

