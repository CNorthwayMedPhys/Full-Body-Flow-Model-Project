# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:07:24 2025

@author: Cassidy.Northway
"""
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
#%%

#Enter the orginal number of dimensions
iXDim = 132
iYDim = 53
iZDim = 349

#Enter the stacked number of dimensions

fXDim = 132
fYDim = 74
fZDim = 489

#Enter the file name you want to save this under
filename = 'XCAT_PA_stacked.egsvoi'  
#%%

#Load in the egsvoi file
root = tk.Tk()
root.withdraw()  # Hide the main window
root.attributes("-topmost", True)
file_path = fd.askopenfilename(parent=root, title = 'Select egsvoi file')


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
    ind = 1
    vInd = 0
    tag = 0
    for k in range(iZDim):
        for j in range(iYDim):
             for i in range(iXDim):
                 if tag == 0:
                     if ind == voxelIndices[vInd]:
                         VOIArray[i,j,k] = voxelIndices[vInd]
                         vInd += 1
                         if vInd == len(voxelIndices):
                             tag = 1
                 ind += 1
#VOIArray contains the all the indice numbers written to the appropriate location within the matrix                 
#%%

# We know that the we have z-extensions padding the front and y-padding on the "top"
zPadding = fZDim - iZDim
yPadding = fYDim - iYDim

#First we need to know the indices taken up by the padding.
zPaddedIndices = zPadding * fXDim * fYDim
yPaddingIndices = yPadding * fXDim

egsvoi=[]
#Replace the VOI Array values with the appropriate 
for k in range(iZDim):
    for j in range(iYDim):
        for i in range(iXDim):
            if VOIArray[i,j,k] != 0:
                value = VOIArray[i,j,k]
                value = int(value + zPaddedIndices + (j*yPaddingIndices))
                egsvoi.append(value)

#%% 
numVOI = np.count_nonzero(VOIArray)
with open(filename, 'w') as fid:
    fid.write(f'{1}, {numVOI} \n')
    for value in egsvoi:
        fid.write(f'{value}\n')

print('Writing egsvoi... Done.')           


