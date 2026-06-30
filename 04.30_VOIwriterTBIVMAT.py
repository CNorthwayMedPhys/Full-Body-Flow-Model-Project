# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:35:21 2026

@author: Cassidy.Northway

Take the TBI VMAT egsphant file produced by RA with the couch and generate VOI files
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
import scipy as sp
import pydicom 
import mahotas
from skimage.transform import resize
from skimage import img_as_bool


#%% File names and user inputs

egsphantName = 'VCCN_XCAT.egsphant'
structName = 'RS.CN_XCAT.FullAP_cHU.dcm'

roiNum = 14
LocationID = '4'
filename = "VMAT_Files//VOI_Files"
#%%Define classes for handling files
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


        egsphantObject=egsphant(nmaterials, materialList, [xdim, ydim, zdim], [xedgesList, yedgesList, zedgesList], [xcenterList, ycenterList, zcenterList], materialArray, densityArray, []) 
    return egsphantObject


#%%
def interpolate_binary(image1, image2, t):
    """
    Interpolates between two binary images using Signed Distance Functions.
    t: Weight factor (0.0 returns image1, 1.0 returns image2)
    
    # Resize using order=0 (equivalent to nearest-neighbor)
    high_res_mask = resize(mask, (1024, 1024), order=0, anti_aliasing=False)

    # Convert back to actual boolean/binary if needed
    high_res_mask = img_as_bool(high_res_mask)
    
    
    """
    
    # Resize using order=0 (equivalent to nearest-neighbor)
    high_res_image1 = resize(image1, (image1.shape[0] * 4,image1.shape[1] * 4), order=0, anti_aliasing=False)
    high_res_image2 = resize(image1, (image2.shape[0] * 4,image2.shape[1] * 4), order=0, anti_aliasing=False)

    # Convert back to actual boolean/binary if needed
    image1 = img_as_bool(high_res_image1)
    image2 = img_as_bool(high_res_image2)
    
    # 1. Compute Distance Transforms
    # Distances inside are positive, distances outside are negative
    dist1 = sp.ndimage.distance_transform_edt(image1) - sp.ndimage.distance_transform_edt(1 - image1)
    dist2 = sp.ndimage.distance_transform_edt(image2) - sp.ndimage.distance_transform_edt(1 - image2)

    # 2. Linearly interpolate the distance fields
    interp_dist = (1 - t) * dist1 + t * dist2

    # # 3. Threshold at zero to get the new binary image
    interp_image = (interp_dist >= 0).astype(np.uint8)
    
    #Try dilation prior to downsizing
    dil_image = sp.ndimage.binary_dilation(interp_image)
    
    #Resize down again and make binary
    resized_image = resize(dil_image, (image1.shape[0]/4,image1.shape[1]/4), preserve_range=True, order=0, anti_aliasing=False)
    return img_as_bool(resized_image)
#%%Load in files

cd = os.getcwd()

egsphantFile = os.path.join(cd,egsphantName)
egsphantObj = readEgsphant(egsphantFile)
print("Phantom Read")

structFile = os.path.join(cd,structName)
ds = pydicom.dcmread(structFile)

roiNum = roiNum - 1
stSet = ds.StructureSetROISequence[roiNum]
print("Opening "+ stSet.ROIName)

stSet = ds.ROIContourSequence[roiNum]
contourSet = stSet.ContourSequence

#%% Access the desired ROI contours and create a list
contourArray = []

for contour in contourSet:
    data = contour.ContourData
    sliceArray = np.zeros((int(len(data)/3)+1,3))
    j = 0
    k = 0
    for i in range(len(data)):
        if k == 0:
            sliceArray[j,0] = data[i]/10
            k = 1
        elif k == 1:
            sliceArray[j,1] = data[i]/10
            k = 2
        else:
            sliceArray[j,2] = data[i]/10
            k = 0
            j += 1
    sliceArray[-1,:] = sliceArray[0,:]       
    contourArray.append(sliceArray)
    

    
#%% Create empty array to write VOI "true values into", same size egsphant

VOIArray = np.zeros_like(egsphantObj.densityArray)

[xq,yq] = np.meshgrid(egsphantObj._centers[0] , egsphantObj._centers[1])

zIndex=[]   

stackValue = 63
for z in range(len(contourArray)):
    sliceArray = contourArray[z]
    shape = xq.shape
    poly = path.Path(sliceArray[:,0:2])
    points = np.vstack((xq.ravel(), yq.ravel())).T
    inside = poly.contains_points(points)
    insideArray = inside.reshape(shape)
    insideArray = insideArray.astype(int)
    zj = np.where(egsphantObj._edges[2] == sliceArray[0,2])[0][0]
    zIndex.append(zj)
    insideArray = np.flip(np.flip(np.rot90(insideArray),axis=0),axis=1)
    insideArray = np.hstack((np.zeros((263,stackValue)),insideArray[:,:-stackValue]))
    VOIArray[:,:,zj] = VOIArray [:,:,zj] + insideArray    

#Assure VOI Array is binary mask
VOIArray[VOIArray > 1] = 1    


#%%NEED TO INTERP  Z
zIndex = np.unique(zIndex)
zMin = zIndex[0]
zMax = zIndex[-1]

for zValue in range(len(VOIArray[0,0,:])):
    if zValue > zMin and zValue < zMax:
        if zValue not in zIndex:
            # Run interpolation
           
            image1 = VOIArray[:,:,zValue-1]
            image2 = VOIArray[:,:,zValue+1]
            image3 = interpolate_binary(image1,image2, 0.5)
            VOIArray[:,:,zValue] = image3
            
            # plt.imshow(image1)
            # plt.show()
            # plt.imshow(image3)
            # plt.show()
            # plt.imshow(image2)
            # plt.show()



#%% Show images
index = 566

Summed = egsphantObj.densityArray + VOIArray*8
plt.figure()         
plt.imshow(Summed[:,:,index], cmap='hot')  

plt.figure()         
plt.imshow(egsphantObj.densityArray[:,:,index], cmap='hot')

plt.figure()         
plt.imshow(VOIArray[:,:,index], cmap='hot')


#%%Convert to egsvoi voxel number
egsvoi = []
for k in range(VOIArray.shape[2]):
    for j in range(VOIArray.shape[1]):
        for i in range(VOIArray.shape[0]):
            if VOIArray[i,j,k] == 1:
                index = (i) + (j*VOIArray.shape[0]) + (k *VOIArray.shape[0] *VOIArray.shape[1]) + 1
                egsvoi.append(index)
#%% Now write the file
filename = os.path.join(cd,"VMAT_Files//VOI_Files",str(LocationID)+".egsvoi")
mode = 1

with open(filename, 'wt') as fid:
        fid.write(f"{mode}, {len(egsvoi)}\n")
        for val in egsvoi:
            fid.write(f"{val}\n")


    

   
            
            
    

    

