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


#%% Fcn: read treatment plan
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

#%% Load in compartment folder
cd = os.getcwd()
folders =["VMAT_Files", "allVMAT_Files"]
iso ="Chest"
#fig, ax = plt.subplots()
# compData = np.load(os.path.join(cd,"BVResults_VMAT_static_CompOnly.npy"))
# compData = compData[compData!=0]

for folder in folders:
    print('for '+ folder)
    dose_folder = folder+"\\4DDoseData\\"
    treatmentplan = folder+"\\TreatmentPlan.txt"
    compartment_folder = folder+ "\\CompartmentData\\"
    MUData = np.load(os.path.join(cd,dose_folder,iso+"_MUsteps.npy"))
    
    print(MUData[:,-5:-1])
    print(np.sum(MUData[1,:]))
#     # Count zeros per column: array([2, 1, 2])
#     DoseData = DoseData[1:,1:]
#     zero_counts = (DoseData  == 0).sum(axis=0)

# # Find the column index with the fewest zeros
#     zero_counts = (DoseData  == 0).sum(axis=0)
#     col_index = np.argmin(zero_counts)
#     #print("Zeros per column:", zero_counts)
#     #print("Column with fewest zeros index:", col_index)
#     print("Fewest zeros is ", zero_counts[col_index] )
    
#     print("Summed value of column is ", sum(DoseData[col_index]))
    
    
    
