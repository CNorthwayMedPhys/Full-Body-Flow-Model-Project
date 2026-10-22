# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 12:53:10 2025

@author: Cassidy.Northway

"""
import os
import sys
import numpy as np
import pandas as pd
import pydicom as pyd
import seaborn as sns
import matplotlib.pyplot as plt

#%% Check out the new dose data
cd = os.getcwd()
dose_folder = "ExtSSD_Files\\4DDoseData\\"
mapping_folder = "ExtSSD_Files\\VOIMappingArray\\"

VOIValues = []

iso = 'AP'   
numbers  = range(28,127)
for number in numbers:
    VOIMappingArrayPath = os.path.join(cd,mapping_folder,iso,str(number)+'.npy')
    VOIMappingArray = np.load(VOIMappingArrayPath)
    VOIValues = np.append(VOIValues, VOIMappingArray[:,1])
    
VOIValues = np.unique(VOIValues)

doseData = np.load(os.path.join(cd,dose_folder,'ExtendedAP.npy'))
doseDataVOI = doseData[0,1:]

shared_count = len(set(VOIValues).intersection(set(doseDataVOI)))
print(shared_count)

#What about the Sweeping Co-60 maps, should be the same
mapping_folder_sweeping = 'VOIMappingArrays\Hi-Res\AP'

VOIValues = []

iso = 'AP'   
numbers  = range(28,127)
for number in numbers:
    VOIMappingArrayPath = os.path.join(cd,mapping_folder_sweeping,str(number)+'.npy')
    VOIMappingArray = np.load(VOIMappingArrayPath)
    VOIValues = np.append(VOIValues, VOIMappingArray[:,1])
    
VOIValues = np.unique(VOIValues)


shared_count = len(set(VOIValues).intersection(set(doseDataVOI)))
print(shared_count)