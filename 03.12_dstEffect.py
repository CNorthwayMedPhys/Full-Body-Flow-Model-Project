# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:46:17 2026

@author: Cassidy.Northway
"""
import numpy as np
import os

def DoseDataSampler (doseArray,time,voi,dst):
    try:
        timeRange = int(dst/0.002)
        voiIndex = np.where(doseArray[0,:] == int(voi))[0][0]
        timeIndex = np.where(np.isclose(doseArray[1:,0],time))[0][0] + 1
        dose = np.sum(doseArray[timeIndex:timeIndex+timeRange,voiIndex])
        
    except:
        dose = 0
        print("this shouldnt happen")

    return dose 

#%%

cd = os.getcwd()
dose_filename_AP = "2ms_SingleSweep.npy"
doseArray = np.load(os.path.join(cd,dose_filename_AP))
doseArray[1:,1:] = doseArray[1:,1:] * 9.76E14 #Gy/min

voi = 620200
dsts = [0.002,0.1,0.2,0.3,0.4,0.5]
for dst in dsts:
    dose = 0
    for time_step in np.arange(0, (0.45*60),dst):
        dose_step = DoseDataSampler(doseArray,time_step,voi,dst) 
        absdoseStep = dose_step 
        dose = dose + absdoseStep
    dose = dose * 0.45
    print(str(dst)+ " s time step results in summed dose of "+ str(dose))    
 
print("Summed dose is " + str(np.sum(doseArray[1:,26])* 0.45 ))    