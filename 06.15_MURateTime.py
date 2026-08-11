# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:22:45 2026

@author: Cassidy.Northway
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#%% https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

#%%Load in all sheets

cd = os.getcwd()
excelfile = os.path.join(cd,'VMAT_Files\\FieldTime Calcs.xlsx' )
all_sheets = pd.read_excel(excelfile,sheet_name=None)


#%% Load in treatment plan
num_fields = 11

treatmentPlan = os.path.join(cd,"VMAT_Files\\TreatmentPlan_4Feet.txt")
fieldsArray = np.zeros([num_fields,2],dtype=object)
with open(treatmentPlan, 'r') as x:
    for i in range(num_fields+1):
            line = x.readline().split(',')
            if i != 0:
                fieldsArray[i-1,0] = line[0]
                fieldsArray[i-1,1] = float(line[1].strip())

#%% Process VMAT files
for iso in ["Abdo","Chest","Pelvis"]:


    keys = [key for key in all_sheets if iso in key] 
    MU = []
    time = []
    MU_per_field = [] 
    i = 0
    for key in keys:
        sheet = all_sheets[key]
        MUseries = sheet.loc[:,'Meterset Weight']
        MU_per_fieldseries = sheet.loc[:,'MU']
        MU_per_field = np.append(MU_per_field,MU_per_fieldseries.to_numpy())
        MU= np.append(MU,MUseries.to_numpy() + i)
        timeseries = sheet.loc[:,'Time (s)']
        time = np.append(time, timeseries.to_numpy())
        i += 1.0001
        
    #Renormalize MU scale 
    MU = (MU - MU.min())/(MU.max()-MU.min())
    
    
    #Cumulatively sum time
    time[np.isnan(time)] = 0
    time = np.cumsum(time)
    
    MU_per_field[0] = 0
    MU_per_field[np.isnan(MU_per_field)] = 0
    MU_per_field = np.cumsum(MU_per_field)

    
    final_file = np.vstack((MU.T,MU_per_field.T,time.T))
    writename = os.path.join(cd,'VMAT_Files\\MUvsTimeArray',iso+".npy" )
    np.save(writename,final_file)

#%% Process IMRT files
for iso in ["HeadL","HeadR","RFoot","LFoot","KneeAnt","KneePost"]:


    keys = [key for key in all_sheets if iso in key] 
    MU = []
    time = []
    MU_per_field = [] 
    i = 0
    for key in keys:
        sheet = all_sheets[key]
        MUseries = sheet.loc[:,'Meterset Weight']
        MU_fieldseries = sheet.loc[:,"MU"]
        MU_per_field = np.append(MU_per_field,MU_fieldseries.to_numpy())
        MU= np.append(MU,MUseries.to_numpy() + i)
        timeseries = sheet.loc[:,'Time (s)']
        time = np.append(time, timeseries.to_numpy())
        i += 1.0001
        
    #Renormalize MU scale 
    MU = (MU - MU.min())/(MU.max()-MU.min())
    
    #Interpolate NaN values
    time[0] = 0
    nans, x =nan_helper(time)
    time[nans] =np.interp(x(nans),x(~nans),time[~nans]) 

    MU_per_field[0] = 0
    nans, x = nan_helper(MU_per_field)
    MU_per_field[nans] = np.interp(x(nans),x(~nans),MU_per_field[~nans])
   
    
    if "Foot" in iso:
        final_file = np.vstack((MU.T,MU_per_field.T*2,time.T))
    else:
        final_file = np.vstack((MU.T,MU_per_field.T,time.T))
    writename = os.path.join(cd,'VMAT_Files\\MUvsTimeArray',iso+".npy" )
    np.save(writename,final_file)
