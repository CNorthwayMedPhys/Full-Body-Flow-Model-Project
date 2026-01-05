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
files_unstacked_location = '\\CompartmentData\\AP\\UnStackedEgsvoi'
files_stacked_location = '\\CompartmentData\\AP\\StackedEgsvoi\\'  

dir_path = os.path.dirname(os.path.realpath(__file__))
file_search_location = dir_path + files_unstacked_location

file_locations = []
for root, dirs, files in os.walk(file_search_location):
    for file in files:
        if file.endswith(".egsvoi"):
            full_path = os.path.join(root, file)
            file_locations.append(full_path)

#%%

#Load in the egsvoi files
for file_path in file_locations:
    compartmentName = file_path.split('\\')[-1].split('.')[0]

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
    with open(dir_path + files_stacked_location +compartmentName +'.egsvoi',  'w') as fid:
        fid.write(f'{1}, {numVOI} \n')
        for value in egsvoi:
            fid.write(f'{value}\n')
    fid.close()
    print('Writing egsvoi... Done.')



