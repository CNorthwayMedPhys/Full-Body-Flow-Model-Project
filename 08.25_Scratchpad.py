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
folders =["VMAT_Files", "allVMAT_Files"]
fig, ax = plt.subplots()
# compData = np.load(os.path.join(cd,"BVResults_VMAT_static_CompOnly.npy"))
# compData = compData[compData!=0]
for folder in folders:
    dose_folder = folder+"\\4DDoseData\\"
    treatmentplan = folder+"\\TreatmentPlan.txt"
    
    #%%
    def readTreatmentPlan (file):
        with open(file) as x:
            array = x.readlines()
        array = array[1:]
        iso_list = []
        time_list = []
        MU_list = []
        gap_array = []
        for i in range(len(array)):
            item = array[i]
            item = item.split(',')
            iso_list.append(item[0])
            time_list.append(float(item[1]))
            MU_list.append(float(item[2]))
            gap_array.append(float(item[3]))
        d = {'iso': iso_list, 'time': time_list, 'MU': MU_list, 'Gap':gap_array}
        df = pd.DataFrame(d)
        return df
    #%%
    dfTreatmentPlan = readTreatmentPlan(treatmentplan) 
    
    for iso_num in range(dfTreatmentPlan.shape[0]):
        iso = dfTreatmentPlan.iso[iso_num]
        MU = dfTreatmentPlan.MU[iso_num]
        DoseData = np.load(os.path.join(cd,dose_folder,iso+"_Vessel.npy"))
        DoseData = DoseData[1:,1:]
        DoseData = np.sum(DoseData,axis = 0) * MU * 6.6511073E13
        #fig, ax = plt.subplots()
        #sns.ecdfplot(data=DoseData, complementary = True)
        plt.title(iso)
        if iso_num ==0:
            summedDoseData = DoseData
        else:
            summedDoseData = summedDoseData + DoseData
        print(iso)
        

    sns.ecdfplot(data=summedDoseData/2, complementary = True, label = "Vessel Sum "+folder)
    print('summed dose mean is  '+str(np.mean(summedDoseData)))
    #%% Load in the RD file
    cd = os.getcwd()
    filenameRD = folder+"\\TPSExport\\RD.CN_XCAT.VOI.dcm"
    rd_file = pyd.dcmread(os.path.join(cd,filenameRD))
    DVHSeq = rd_file.DVHSequence
    numDVH = len(DVHSeq)
    
    for i in range(numDVH):
        DVH = DVHSeq[i]
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
    
    ax.plot(doseArray/100,volumeArray/volumeArray[0], linestyle = ':', label ='TPS VOI '+folder)
    plt.legend() 