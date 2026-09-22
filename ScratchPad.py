# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:57:20 2026

@author: Cassidy.Northway
"""
import numpy as np
import pandas as pd
import os

#%%
#Dose file locations 
cd = os.getcwd()

#File locations 
dose_folder= "extSSD_Files\\4DDoseData\\"
APDoseData = np.load(os.path.join(cd,dose_folder,"ExtendedAP.npy")) 
PADoseData = np.load(os.path.join(cd,dose_folder,"ExtendedPA.npy")) 

APDoseData = APDoseData[1:,1:].flatten()
PADoseData = PADoseData[1:,1:].flatten()

APDoseData = np.sort(APDoseData , descending=True)
PADoseData = np.sort(PADoseData , descending=True)