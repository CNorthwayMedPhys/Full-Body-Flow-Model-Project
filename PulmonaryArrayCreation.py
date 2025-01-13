# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:48:02 2024

@author: Cassidy.Northway
"""


#%%Import
import pandas as pd
import numpy as np
import scipy
import tkinter as tk
import tkinter.filedialog as fd

#%%Select all the relevant fitted vessel files
tk.Tk().withdraw()
filez = fd.askopenfilenames()

#Define the data frame
df = pd.DataFrame(columns=['Name', 'Length (mm)', 'Radius Values (mm)']) 

for file in filez:
    file_name = file
    name = file.split('/')[-1]
    name = name.split('.')[0]
    
    #Load the file data
    array = np.load(file_name)

    #Extract geometeric info
    center_array = array[:,0:3 ]
    radius_values = array[:,3 ]

    #Determine the vessels length
    tot_dist = 0
    for i in range(0,np.shape(center_array)[0]-1):
        dist = np.linalg.norm(center_array[i,:] - center_array[i+1,:])
        tot_dist = tot_dist + dist    

    #Determine radius values
    index_rounding=np.round(len(radius_values)*0.05).astype(int)+1
    Ru = np.mean(radius_values[0:index_rounding])
    Rd = np.mean(radius_values[-index_rounding:-1])
    radius_array = [Ru, Rd]   
    
    new_row = {'Name' : name, 'Length (mm)': tot_dist, 'Radius Values (mm)': [Ru,Rd]}
    df.loc[len(df)] = new_row  

#Manually add the dias_pul_art using data from 2014_Qureshi
new_row = {'Name' : 'dias_pul_art', 'Length (mm)': 45.0, 'Radius Values (mm)': [13.5,13]}
df.loc[len(df)] = new_row      

#%% Save the data frame    
df.to_pickle('Pulmonary.pkl')       
        