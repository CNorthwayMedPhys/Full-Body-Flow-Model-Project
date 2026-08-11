# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:35:59 2026

@author: Cassidy.Northway

!!!Works!!!
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cd = os.getcwd()
       
#%% Load in one sheet at a time
fieldnames = ['Abdomen', 'HeadL', 'HeadR', 'Chest', 'Pelvis', 'KneeAnt', 'KneePost', 'LFoot', 'RFoot']

compartments = ['Lung_R','L_kidney','R_DeepFemoral','L_AnteriorTib','L_Arm Region',
                'L_InterIliac', 'R_PostTib', 'Spleen', 'L_heart', 'Lung_L', 'Liver',
                'Brain', 'R_Heart', 'R_AnteriorTib', 'S_int', 'R_InterIliac', 'R_Arm Region',
                'R_Kidney', 'L_DeepFemoral', 'Pancreas', 'L_int', 'Stomach', 'L_PostTib']
compartmentnumbers = [27,9,23,20,12,18,25,7,1,26,10,2,0,24,3,22,15,6,19,4,8,5,21]
for fieldname in fieldnames:
    filepath = os.path.join(cd,'VMAT_Files\\DVH_Data',fieldname+'.csv')
    df = pd.read_csv(filepath)
    for i in range(len(compartmentnumbers)): #Going through each compartment
        compartment = compartments[i]
        compartmentnumber = compartmentnumbers[i]
        columns = [col for col in df.columns if compartment in col]
        doseData = df[columns[0]].to_numpy()
        volumeData = df[columns[1]].to_numpy()
        finalData = np.vstack((volumeData,doseData))
    
        filename = os.path.join(cd,'VMAT_Files\\CompartmentData\\',fieldname,str(compartmentnumber)+'DVH.npy')
        
        np.save(filename,finalData)
        
        