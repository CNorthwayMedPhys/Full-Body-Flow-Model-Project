# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:22:45 2026

@author: Cassidy.Northway
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#%%Load in all sheets

cd = os.getcwd()
excelfile = os.path.join(cd,'VMAT_Files\\FieldTime Calcs.xlsx' )
all_sheets = pd.read_excel(excelfile,sheet_name=None)


#%% Load in treatment plan
num_fields = 11

treatmentPlan = os.path.join(cd,"VMAT_Files\\TreatmentPlan.txt")
fieldsArray = np.zeros([num_fields,2],dtype=object)
with open(treatmentPlan, 'r') as x:
    for i in range(num_fields):
        line = x.readline().split(',')
        fieldsArray[i,0] = line[0]
        fieldsArray[i,1] = float(line[1].strip())

#%% Process files
for iso in ["Abdo","Chest","Pelvis"]:


    keys = [key for key in all_sheets if iso in key] 
    MU = []
    time = []
    MU_per_field = [] 
    i = 0
    for key in keys:
        sheet = all_sheets[key]
        MUseries = sheet.loc[:,'Meterset Weight']
        MU_per_fieldseries = sheet.loc[:,'Meterset Weight']
        MU_field = sheet.loc[0,"Total MU"]
        MU_per_field = np.append(MU_per_field,MU_per_fieldseries.to_numpy()*MU_field)
        MU= np.append(MU,MUseries.to_numpy() + i)
        timeseries = sheet.loc[:,'Time (s)']
        time = np.append(time, timeseries.to_numpy())
        i += 1.0001
        
    #Renormalize MU scale 
    MU = (MU - MU.min())/(MU.max()-MU.min())
    
    #Cumulatively sum time
    time[np.isnan(time)] = 0
    time = np.cumsum(time)
    
    final_file = np.vstack((MU.T,MU_per_field.T,time.T))
    writename = os.path.join(cd,'VMAT_Files\\MUvsTimeArray',iso+".npy" )
    np.save(writename,final_file)


