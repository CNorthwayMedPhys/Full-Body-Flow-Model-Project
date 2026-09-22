# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:21:07 2025

@author: Cassidy.Northway
"""

import os
import pydicom as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def DVHSampler(DVHArray):
    volume = random.random()*100 #convert to percentages
    DVHArray = np.fliplr(DVHArray)
    index = np.searchsorted(DVHArray[0,:],volume)
    value = np.interp(volume, [DVHArray[0,index-1], DVHArray[0,index]],[DVHArray[1,index-1],DVHArray[1,index]])
    return value #Gy
#%% Load in the RT file
cd = os.getcwd()
filenameRD = "ExtSSD_Files\\RD.CN_XCAT.PA.dcm"
rd_file = pd.dcmread(os.path.join(cd,filenameRD))
filenameRS = "ExtSSD_Files\\RS.CN_XCAT.CT_1.dcm"
rs_file = pd.dcmread(os.path.join(cd,filenameRS))

write_file = "ExtSSD_Files\\CompartmentData\\PA\\"
#%%Create ROI seq
ROIseq = rs_file.StructureSetROISequence
numROI = len(ROIseq)

ROIMap = np.zeros((2,numROI),dtype = object)
for i in range(numROI):
    ROI = ROIseq[i]
    ROInumber = ROI.ROINumber
    ROIname = ROI.ROIName
    ROIMap[0,i] = int(ROInumber)
    ROIMap[1,i] = ROIname

#%% Pull out the DVHS from the file
DVHSeq = rd_file.DVHSequence
numDVH = len(DVHSeq)

for i in range(numDVH):
    DVH = DVHSeq[i]
    ROINumber = int(DVH.DVHReferencedROISequence[0].ReferencedROINumber)
    ROIName = ROIMap[1,np.where(ROIMap[0,:]==ROINumber)][0][0]
    DVHData = DVH.DVHData
    listDVH = list(DVHData)
    DVHData = np.array(listDVH, dtype=float)
    doseArray = DVHData[::2]
    volumeArray = DVHData[1::2] 
    Dmax = DVH.DVHMaximumDose
    Dmin = DVH.DVHMinimumDose
    
    doseArray[0] = 0
    for i in range(1,len(doseArray)):
        step = doseArray[i]
        doseArray[i] = doseArray[i-1]+step
        
    DVHArray = np.concat(([volumeArray],[doseArray])) 
    plt.plot(doseArray,volumeArray)
    plt.ylim((0,100))
    np.save(os.path.join(cd,write_file + ROIName + "DVH.npy"),DVHArray)
    
    #DVHSampler(DVHArray)

#%% ScratchPad



