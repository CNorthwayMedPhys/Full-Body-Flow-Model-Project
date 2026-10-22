# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 12:53:10 2025

@author: Cassidy.Northway

"""
import os
import sys
import numpy as np


#%% Check out the new dose data
cd = os.path.dirname(os.path.abspath(__file__))
dose_filename_AP = "4DDoseData\\PA\\XCAT_PAFullDoseData500msRes.npy"

dosedata = np.load(os.path.join(cd,dose_filename_AP))
print(np.max(dosedata[0,:]))
print(np.min(dosedata[0,:]))
print(np.mean(dosedata[0,:]))

#%%
#Dose file locations 
cd = os.getcwd()
dose_filename_AP = "4DDoseData\\PA\\XCAT_PA"
doseData = np.zeros((3,0)) 
for i in range(1,81):
    doseDataSub = np.load(os.path.join(cd,dose_filename_AP+"_w"+str(i)+".npy"))   
    doseData = np.append(doseData,doseDataSub,axis = 1)                         


#Get Voxel Mapping Arrays
mapping_filename_AP = "VOIMappingArrays\\Hi-Res\\PA\\"
voxelMap = np.zeros((0,2))
for i in range(28,81):
    voxelMapSub =np.load(os.path.join(cd,mapping_filename_AP+str(i)+".npy"))
    voxelMap = np.append(voxelMap,voxelMapSub,axis =0)

#Get unique Voxel indices
doseVoxels = np.unique(doseData[1,:])
mapVoxels = np.unique(voxelMap[:,1])

#Find matching voxels should = len(mapVoxels)
setDose = set(doseVoxels)
setMap = set(mapVoxels)

commonSet = setDose.intersection(setMap)

print( "length of voxel map is "+ str(len(mapVoxels)))
print(" Where as the number of matches is " + str(len(commonSet)))