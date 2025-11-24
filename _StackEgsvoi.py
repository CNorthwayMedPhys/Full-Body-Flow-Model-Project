# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:07:24 2025

@author: Cassidy.Northway
"""
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import os
#%%

#Enter the orginal number of dimensions
iXDim =  264 
iYDim =  108 
iZDim =  697

#Enter the stacked number of dimensions

fXDim = 264 
fYDim = 129
fZDim = 697

#Enter the file name you want to save this under
filename_unstacked = '\\XCAT_AP_unstacked.egsvoi'
filename_stacked = '\\XCAT_AP_stacked.egsvoi'  
mapping_array = 'APVOIUnstackToStackMapHiRes.npy'
#%%

#Load in the egsvoi file
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = dir_path + filename_unstacked


with open(file_path) as x:

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
    VOIArray = np.zeros((iXDim,iYDim,iZDim))
    
    #Start filling array
    for lin_ind in voxelIndices:           
        x_ind = int((lin_ind-1) % iXDim)
        y_ind = int(((lin_ind-1) //iXDim) % iYDim)
        z_ind = int(((lin_ind-1)//iXDim) // iYDim)
        VOIArray[x_ind,y_ind,z_ind] = int(lin_ind)
#VOIArray contains the all the indice numbers written to the appropriate location within the matrix                 
#%%

# We know that the we have z-extensions padding the front and y-padding on the "top"
yPadding = fYDim - iYDim

#First we need to know the indices taken up by the padding.
yPaddingIndices = yPadding * fXDim

egsvoi=[]
#Replace the VOI Array values with the appropriate 
for k in range(iZDim):
    for j in range(iYDim):
        for i in range(iXDim):
            if VOIArray[i,j,k] != 0:
                value = VOIArray[i,j,k]
                value = int(value + ((k+1)*yPaddingIndices))
                egsvoi.append(value)

#%% Write egsvoi
numVOI = np.count_nonzero(VOIArray)
with open(dir_path + filename_stacked, 'w') as fid:
    fid.write(f'{1}, {numVOI} \n')
    for value in egsvoi:
        fid.write(f'{value}\n')
fid.close()
print('Writing egsvoi... Done.')

#%%
pairedArray = np.array(np.vstack((voxelIndices, egsvoi)))
dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path + '\\StackToUnstackMaps\\'+ mapping_array
np.save(path, pairedArray)


