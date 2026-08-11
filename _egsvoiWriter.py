# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:06:17 2026

@author: Cassidy.Northway
"""

import os
import numpy as np

#%% Load in Maps for AP

#Load in the relevant voxel indices
cd = os.getcwd()
mapping_filename = "\\Remote Git\\ExtSSD_Files\\VOIMappingArray\\AP\\"
voxelMaps = np.zeros((0,2))
for i in range(28,127):
    voxelMapSub =np.load(os.path.join(cd,mapping_filename+str(i)+".npy"))
    voxelMaps = np.append(voxelMaps,voxelMapSub,axis =0)
    
#%%Find unique values and sort
   
mapVoxels = np.unique(voxelMaps[:,1])  
mapVoxels = [int(item) for item in mapVoxels]
mapVoxels = np.sort(mapVoxels)


#%%Write unstacked
filename_unstacked = '\\ExtSSD_Files\\VOI_Files\\AP\\stacked_VOI\\vessels.egsvoi'
numVOI = np.count_nonzero(mapVoxels)
with open(cd + filename_unstacked, 'w') as fid:
    fid.write(f'{1}, {numVOI} \n')
    for value in mapVoxels:
        fid.write(f'{value}\n')
fid.close()
print('Writing egsvoi... Done.')



# #%%Write stacked
# filename_stacked = '\\XCAT_PA_stacked_new.egsvoi'  
# mapping_array = 'PAVOIUnstackToStackMapHiRes_new.npy'

# #Enter the orginal number of dimensions
# iXDim =  263 
# iYDim =  107 
# iZDim =  697

# #Create the empty array
# VOIArray = np.zeros((iXDim,iYDim,iZDim))

# #Start filling array
# for lin_ind in mapVoxels:           
#     x_ind = int((lin_ind-1) % iXDim)
#     y_ind = int(((lin_ind-1) //iXDim) % iYDim)
#     z_ind = int(((lin_ind-1)//iXDim) // iYDim)
#     VOIArray[x_ind,y_ind,z_ind] = int(lin_ind)

# #Enter the stacked number of dimensions

# fXDim = 263 
# fYDim = 128
# fZDim = 697

# # We know that the we have z-extensions padding the front and y-padding on the "top"
# yPadding = fYDim - iYDim

# #First we need to know the indices taken up by the padding.
# yPaddingIndices = yPadding * fXDim

# egsvoi=[]
# #Replace the VOI Array values with the appropriate 
# for k in range(iZDim):
#     for j in range(iYDim):
#         for i in range(iXDim):
#             if VOIArray[i,j,k] != 0:
#                 value = VOIArray[i,j,k]
#                 value = int(value + ((k+1)*yPaddingIndices))
#                 egsvoi.append(value)
                
# with open(cd + filename_stacked, 'w') as fid:
#     fid.write(f'{1}, {numVOI} \n')
#     for value in egsvoi:
#         fid.write(f'{value}\n')
# fid.close()
# print('Writing egsvoi... Done.')

# #%%
# pairedArray = np.array(np.vstack((mapVoxels, egsvoi)))
# dir_path = os.path.dirname(os.path.realpath(__file__))
# path = dir_path + '\\StackToUnstackMaps\\'+ mapping_array
# np.save(path, pairedArray)
                