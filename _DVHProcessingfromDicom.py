# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:21:07 2025

@author: Cassidy.Northway
"""

import os
import pydicom as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Load in the RT file
cd = os.getcwd()
filenameRD = "XCAT_RPRSRD_files\\RD.CN_XCAT.TBI_Full_PA.dcm"
rd_file = pd.dcmread(os.path.join(cd,filenameRD))
filenameRS = "XCAT_RPRSRD_files\\RS.CN_XCAT.Full_PA.dcm"
rs_file = pd.dcmread(os.path.join(cd,filenameRS))

write_file = "CompartmentData\\PA\\"
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
    
    doseArray[0] = Dmin
    for i in range(1,len(doseArray)):
        step = doseArray[i]
        doseArray[i] = doseArray[i-1]+step
        
    DVHArray = np.concat(([volumeArray],[doseArray])) 
    
    np.save(os.path.join(cd,write_file + ROIName + "DVH.npy"),DVHArray)
    

#%% ScratchPad



