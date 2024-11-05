# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:41:37 2024

@author: cbnor
"""

import mat73
import os
import pandas as pd
import numpy as np

#%%Useful fcns
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#%%Load in .mat file
mat_dir = os.path.join(os.getcwd(),'NektarFiles\\ExportFolder\\pwdb_1\\processed_data\\history_files_data.mat')
mat = mat73.loadmat(mat_dir)

#%% Move through the dictionaries to extract relevant info
mat_dic = mat['data']
mat_dic = mat_dic['sim_55art_elas']

#%%Load in the Excel File containing the mapping data 
excel_pd = pd.read_excel('NektarMap.xlsx')

#%% Now we want to map the Nektar data in a way that aligns with our XCAT vessels
combo_index = []
new_df = pd.DataFrame()


for i in range(0,len(mat_dic)):
    combo_tag = 0
    crop_tag = 0
    
    i_dict = mat_dic[i]
    domain_no = i_dict['domain_no']-1
    temp_df = excel_pd[excel_pd['Nektar ID Number'] == domain_no]
    
    if np.shape(temp_df)[0] == 0:
        #special case when we are combining to Nektar data into 1 XCAT Array
        index = excel_pd.index[excel_pd['Nektar ID Number'].str.contains(str(int(domain_no))).fillna(False)].tolist()
        combo_index.append(index[0])
        continue

    #Skip non-mapped vessels
    if np.isnan(temp_df.iloc[0]['segments']):
        continue
        
    if not pd.isna(temp_df.iloc[0]['Crop']):
        crop_tag = 1 
        
    nektar_length = temp_df.iloc[0]['Length(cm)'] #cm
    XCAT_length = temp_df.iloc[0]['length (cm)'] #cm

    if not crop_tag:
        num_steps = len(i_dict['distances'])
        new_dist = np.linspace(0, XCAT_length, num_steps) #cm
    else:
        new_dist = i_dict['distances']*100 #cm
    Q_data = i_dict['Q']*1000000 
    Q_data =  Q_data.tolist()
    i_dic_update = {'XCAT_ID': temp_df.iloc[0]['XCAT ID'], 'Distances (cm)': new_dist, 'Q (cm3/s)': Q_data , 'dt (s)': 0.002, 'Crop tag': crop_tag}    
    new_df = pd.concat([new_df, pd.DataFrame([i_dic_update])], ignore_index=True)
  
#%% Combine the extended vessels

for i in combo_index:
    temp_df = excel_pd.iloc[i]
    
    if not pd.isna(temp_df['Crop']):
        crop_tag = 1 
    nekar_dos = temp_df['Nektar ID Number']
    nekar_dos = nekar_dos.replace(' ', '')
    [a,b] = nekar_dos.split(',')
    
    a_dict = list(filter(lambda mat_dic: mat_dic['domain_no'] == int(a)+1, mat_dic))[0]
    b_dict = list(filter(lambda mat_dic: mat_dic['domain_no'] == int(b)+1, mat_dic))[0]
    
    a_end = a_dict['distances'][-1]
    b_distances = b_dict['distances'] + a_end
    
    ab_distances = np.append(a_dict['distances'],b_distances) * 100 #cm
    
    nektar_length = ab_distances[-1] #cm
    XCAT_length = temp_df['length (cm)'] #cm
    num_steps = len(ab_distances)
    new_dist = np.linspace(0, XCAT_length, num_steps) #cm    
        
    Q_data = np.append(a_dict['Q'] , b_dict['Q'])
    Q_data =  Q_data.tolist() * 1000000 
    i_dic_update = {'XCAT_ID': temp_df['XCAT ID'], 'Distances (cm)': new_dist, 'Q (cm3/s)': Q_data , 'dt (s)': 0.002, 'Crop tag': crop_tag}    
    new_df = pd.concat([new_df, pd.DataFrame([i_dic_update])], ignore_index=True)
      
#%% Save and write
new_df.to_pickle('FlowDataFrame.pkl')    