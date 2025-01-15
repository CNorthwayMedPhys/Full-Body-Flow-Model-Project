# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:48:02 2024

@author: Cassidy.Northway
"""
###Goal: ID where we have overlapping vessel bifurcations, does it match what I've been seeing in my code

#%%Import

#Import 
import pandas as pd
import numpy as np
import scipy
import open3d as o3d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#%% Display three point clouds

#%%Display two point clouds

#%% Import and parse Excel Sheet data


#Import excel sheet
try:
    arteries_sheet = pd.read_excel('C:\\Users\\Cassidy.Northway\\Remote Git\\FlowTracker.xlsx', sheet_name = 0)
    #veins_sheet = pd.read_excel('C:\\Users\\Cassidy.Northway\\GitRemoteRepo\\FlowTracker.xlsx', sheet_name = 1)
except:
    arteries_sheet = pd.read_excel('C:\\Users\\cbnor\\Documents\\Full Body Flow Model Project\\FlowTracker.xlsx', sheet_name = 0)
    #veins_sheet = pd.read_excel('C:\\Users\\cbnor\\Documents\\Full Body Flow Model Project\\FlowTracker.xlsx', sheet_name = 1)
    
sheet=arteries_sheet

#Define the data frame
df = pd.DataFrame(columns=['Name', 'lam', 'Radius Values', 'End Condition']) 

#Define percentile used to estimate Rd and Ru
percent = 0.05

#Define the input artery location 
index_0 = 0

#For every vessel in the sheet
for index in range(0,sheet.shape[0]):
    
    
    name = sheet.at[index,'Anatomy Name']
    file_name = sheet.at[index, 'Filename']
    end_point = sheet.at[index, 'End Point']
    branches = sheet.at[index,'Out Flow']
    
    if name == 'art_carotid':
       new_row = {'Name' : 'art_carotid_0', 'lam': 58, 'Radius Values': [2.9,2.8], 'End Condition': 'ST' }
       df.loc[len(df)] = new_row 
    else:
        #Do the branches terminate in vessel(s) or a ST?
        if pd.isna(end_point):
            final_cnd ='ST'
        else:
            final_cnd = end_point.split(',')
            final_cnd = [s.strip() for s in final_cnd]
            
        #Does the vessel branch along it's length?
        if pd.isna(branches):
            seg_tag = False
        else:
            seg_tag = True
            branches = branches.split(',')
            branches = [s.strip() for s in branches]
            
            #Remove branches with branching from the end 
            if final_cnd != 'ST':
                for vessels in final_cnd:
                    branches.remove(vessels)
                    
        #If there are no branches add the vessel to the data frame
        if seg_tag == False:
            seg_name = name + '_0'
            main_file = file_name + '_fitted_data.npy'
            
            #Load the file data
            try:
                array = np.load('C:\\Users\\Cassidy.Northway\\Remote Git\\FittedVesselsFiles\\' + main_file)
            except:
                array = np.load('C:\\Users\\cbnor\\Documents\\Full Body Flow Model Project\\FittedVesselsFiles\\' + main_file)
            
            #Extract geometeric info
            center_array = array[:,0:3 ]
            radius_values = array[:,3 ]
            
            #Determine radius values
            index_rounding= np.round(len(radius_values)*percent).astype(int)+1
            Ru = np.mean(radius_values[0:index_rounding])
            Rd = np.mean(radius_values[-index_rounding:-1])
            radius_array = [Ru, Rd]
            
            #Determine the vessels length
            tot_dist = 0
            for i in range(0,np.shape(center_array)[0]-1):
                dist = np.linalg.norm(center_array[i,:] - center_array[i+1,:])
                tot_dist = tot_dist + dist    
            lam_value = tot_dist / Ru 
            
            #Add to the dataframe
            new_row = {'Name' : seg_name, 'lam': lam_value, 'Radius Values': radius_array, 'End Condition': final_cnd }
            df.loc[len(df)] = new_row
            
        #If the vessel does branch we need to subdivide it into segements via an additional data frame
        if seg_tag == True:
            seg_df = pd.DataFrame(columns=['Branch Name','Index of Split','Dist'])
            sub_index = 0
            
            #Get the names of all the files and import data
            main_file = file_name + '_fitted_data.npy'
            try:
                main_array = np.load('C:\\Users\\Cassidy.Northway\\Remote Git\\FittedVesselsFiles\\' + main_file)
            except:
                main_array = np.load('C:\\Users\\cbnor\\Documents\\Full Body Flow Model Project\\FittedVesselsFiles\\' + main_file)
            
            
            branch_files = []
            
            for i in range(0,len(branches)):
                if branches[i] == 'art_carotid':
                    seg_df.loc[len(seg_df)] = {'Branch Name': branches[i] , 'Index of Split': 1, 'Dist': 0.1}  
                else:   
                    branch_row = sheet[sheet['Anatomy Name'].str.match(branches[i])].index.values[0]
                    branch_file = sheet.at[sheet.index[branch_row],'Filename']
                    branch_file = branch_file + '_fitted_data.npy'
            
                    try:
                       branch_array = np.load('C:\\Users\\Cassidy.Northway\\Remote Git\\FittedVesselsFiles\\' + branch_file)
                    except:
                       branch_array = np.load('C:\\Users\\cbnor\\Documents\\Full Body Flow Model Project\\FittedVesselsFiles\\' + branch_file)
                   
                #Find the nearest points
                    dist_array = scipy.spatial.distance.cdist(main_array[:,0:3],branch_array[:,0:3])
                    dist_array_a = dist_array[:,0]
                    dist_array_b = dist_array[:,-1]
                    if np.min(dist_array_b) < np.min(dist_array_a):
                        branch_array = np.flipud(branch_array)
                        np.save('C:\\Users\\Cassidy.Northway\\Remote Git\\FittedVesselsFiles\\' + branch_file, branch_array)
                        
                    index_split = np.where (np.min(dist_array) == dist_array)[0]
                    seg_df.loc[len(seg_df)] = {'Branch Name': branches[i] , 'Index of Split': index_split[0], 'Dist': np.min(dist_array)}
                    
            #######I want to include something here to ID the distance b/w the branch and the main vessel
                # if np.min(dist_array) >= 3:
                #     print('main = ' + file_name)
                #     print('branch = ' + branches[i])
                       
            ###########In addition if index are very close together I want to know that ####################
                
                #Sometimes segments have identical index values
                match_index = seg_df.duplicated(subset = 'Index of Split', keep = False)
                sub_df = seg_df[match_index]
            
                
                # for i in range(0,len(branches)):
                #     prim_index = seg_df.at[i,'Index of Split']
                #     all_index = seg_df['Index of Split'].to_numpy()
                    # diff = np.abs(all_index - prim_index)
                    
                    #if (diff < 6).any() & (diff > 1).any():
                    
                    
            if not sub_df.empty:
                #print('main = ' + file_name)
                #print(sub_df)
                dist_col = sub_df['Dist'].idxmax()
                intial_index = sub_df.at[dist_col,'Index of Split']
                seg_df.at[dist_col,'Index of Split'] = (intial_index + 1)
                #print(seg_df)
            #Sometime segment index values are equal to zero
            seg_df = seg_df.replace(0,1)
                
                
                
               #We now have the number of off branching vessels and where they branch so now we need to now save the segements and off branches and sort segment frame by distance along vessel
    
            seg_df = seg_df.sort_values(by ='Index of Split')
            seg_df = seg_df.reset_index(drop=True)
            intial_index = 0
            
            
            for i in range(0,len(seg_df)+1):
                if i != len(seg_df):
                    sub_name =  name + '_' + str(i)
                    final_index = seg_df.at [ i , 'Index of Split']
                    center_array = main_array[intial_index:final_index+1,0:3 ]
                    radius_values = main_array[intial_index:final_index,3 ]
                    end_cnd = [name + '_' + str(i+1), seg_df.at[i, 'Branch Name' ]+'_0' ]        
                    intial_index = final_index
                    
                    
                    #Determine radius values
                    index_rounding=     np.round(len(radius_values)*percent).astype(int)+1
                    Ru = np.mean(radius_values[0:index_rounding])
                    Rd = np.mean(radius_values[-index_rounding:-1])
                    radius_array = [Ru, Rd]
                    
                    #Sometimes the radius values are very minimal then we get nan values, catch and correct here.
                    if np.isnan(radius_array).any():
                        if np.isnan(Ru):
                            Ru = radius_values[0]
                        else:
                            Rd = radius_values[-1]
                        radius_array = [Ru,Rd]    
                    
                    #Determine the vessels length
                    tot_dist = 0
                    for i in range(0,np.shape(center_array)[0]-1):
                        dist = np.linalg.norm(center_array[i,:] - center_array[i+1,:])
                        tot_dist = tot_dist + dist    
                    lam_value = tot_dist / Ru 
                    
                    #Add to the dataframe
                    new_row = {'Name' : sub_name, 'lam': lam_value, 'Radius Values': radius_array, 'End Condition': end_cnd }
                    df.loc[len(df)] = new_row
                    
                else:
                    sub_name =  name + '_' + str(i)
                    final_index = -1
                    center_array = main_array[intial_index:final_index,0:3 ]
                    radius_values = main_array[intial_index:final_index,3 ]
                    
                    if radius_values.size == 0:
                        center_array = main_array[-3:final_index,0:3 ] 
                        radius_values = main_array[-3:final_index,3 ]
                    
                    if final_cnd != 'ST':
                        end_cnd = list()
                        for j in range(0,len(final_cnd)):
                            end_cnd.append(final_cnd[j] +'_0')
                    else:
                        end_cnd = final_cnd
    
                   
                    #Determine radius values
                    index_rounding=     np.round(len(radius_values)*percent).astype(int)+1
                    Ru = np.mean(radius_values[0:index_rounding])
                    Rd = np.mean(radius_values[-index_rounding:-1])
                    radius_array = [Ru, Rd]
                    
                    if np.isnan(radius_array).any():
                        Ru = radius_values[0]
                        Rd = radius_values[-1]
                        radius_array = [Ru,Rd]
                     
                    #Determine the vessels length
                    tot_dist = 0
                    for i in range(0,np.shape(center_array)[0]-1):
                        dist = np.linalg.norm(center_array[i,:] - center_array[i+1,:])
                        tot_dist = tot_dist + dist    
                    lam_value = tot_dist / Ru 
                     
                    #Add to the dataframe
                    new_row = {'Name' : sub_name, 'lam': lam_value, 'Radius Values': radius_array, 'End Condition': end_cnd }
                    df.loc[len(df)] = new_row
                    


#%% Combine end to start vessels                  
                
removal_indices=[]
for i in range (0,len(df)):
    end_condition = df.at[i,'End Condition']
    if end_condition != 'ST' and len(end_condition)==1:
        index = df[df['Name'] == end_condition[0]].index[0]
        name = df.at[i,'Name']
        Ru = df.at[i,'Radius Values'][0]
        Rd = df.at[index, 'Radius Values'][1]
        radius_values = [Ru, Rd]
        lam_value = ((df.at[i,'Radius Values'][0]*df.at[i,'lam'])+(df.at[index, 'Radius Values'][0]*df.at[index, 'Radius Values'][0]))/Ru
        new_end_condition = df.at[index,'End Condition']
        new_row = {'Name' : name, 'lam': lam_value, 'Radius Values': radius_values, 'End Condition': new_end_condition }
        df.loc[len(df)] = new_row
        removal_indices.append(i)
        removal_indices.append(index)

df_updated = df.drop(df.index[removal_indices])
df_updated = df_updated.reset_index(drop=True)
             
            
#%% Sort everything descending from the input vessel
df_ordered = pd.DataFrame(columns=['Name', 'lam', 'Radius Values', 'End Condition']) 

df_ordered.loc[0]=df_updated.loc[index_0]
next_layer = df_ordered.at[0,'End Condition']
next_layer_1 = []
while len(next_layer)>0:
    for daughter in next_layer:
        index = df_updated[df_updated['Name']==daughter].index.values[0]
        df_ordered.loc[len(df_ordered)] = df_updated.loc[index]
        end_condition = df_updated.at[index, 'End Condition']
        
        if end_condition !='ST':
            next_layer_1 = np.append(next_layer_1, end_condition)
    next_layer = next_layer_1
    next_layer_1 = []
 
#%% Why are there fewer after the ordering...
df_leftovers = df_updated
for i in range(0,len(df_ordered)):
    name = df_ordered.at[i,'Name']
    df_leftovers = df_leftovers.drop(df_leftovers[df_leftovers['Name'] == name].index.values[0]) 
    
 
    
 
    
#%% Rename everything by the index        
for i in range (0,len(df_ordered)):
    end_condition = df_ordered.at[i,'End Condition']
    if end_condition != 'ST':
        replacement = []
        if isinstance(end_condition[0], str): 
            for j in range(0,len(end_condition)):
                condition = end_condition[j]
                try: 
                    index = df_ordered[df_ordered['Name']==condition].index.values[0]
                except:
                    print(condition)
                    
                replacement.append(index)
            df_ordered.at[i,'End Condition'] = replacement 

#%% Round every thing to two decimal places
for i in range (0, len(df_ordered)):
    Ru_i,Rd_i = df_ordered.at[i, 'Radius Values' ]
    lam_i = df_ordered.at[i, 'lam']
    dist = lam_i*Ru_i 
    Ru_f = np.round(Ru_i,decimals=2)
    Rd_f = np.round(Rd_i,decimals=2)
    lam_f = np.round(dist,decimals=2)/Ru_f
    if Rd_f>Ru_f:
        Rd_f=Ru_f
    if dist < 0.1:
        dist = 0.1
        lam_f = 0.1/Ru_f
    df_ordered.at[i, 'Radius Values' ] = [Ru_f,Rd_f]   
    df_ordered.at[i, 'lam'] = lam_f   
    
#I want to ID the smallest Rd values
# for i in range(0,len(df_ordered)):
#     Ru_i,Rd_i = df_ordered.at[i, 'Radius Values' ]
#     if Rd_i < 0.2:
#         df_ordered.at[i, 'Radius Values' ] = [Ru_i,0.2]
#         print(str(i))            
#%% Harded coded solutions for right subclavian artery Ru

index = df_ordered[ df_ordered['Name']== 'right_subclavian_artery_0'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = 35 
Ru_f = 7 #4.4 #(mm) from 2000_Olufsen
Rd_f = 7
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f   

index = df_ordered[ df_ordered['Name']== 'right_subclavian_artery_1'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = 32 #og length (67) - sc_0 lenth
Ru_f = 4.4 #4.4 #(mm) from 2000_Olufsen
Rd_f = 3 #b/w suggested 4.3 and my value 2.6
lam_f = dist/Ru_f #og length (67) - sc_0 lenth

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f  

#%%% Hard code in values
index = df_ordered[ df_ordered['Name']== 'aorta_0'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = 12.5 #(mm) from 2000_Olufsen
Rd_f = 11.4
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f   

index = df_ordered[ df_ordered['Name']== 'aorta_1'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = 11.4 #(mm) from 2000_Olufsen
Rd_f = 11.1
lam_f = 1.59

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f 

index = df_ordered[ df_ordered['Name']== 'aorta_2'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = 11.1 #(mm) from 2000_Olufsen
Rd_f = 10.9
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f 

index = df_ordered[ df_ordered['Name']== 'arteries_lleg25_0'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = 1.5 #(mm) measured manually in mesh lab
#Rd_f = 11.4
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_i]   
df_ordered.at[index, 'lam'] = lam_f   

index = df_ordered[ df_ordered['Name']== 'left_external_iliac_artery_0'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = 3.02 #(mm) matching Right
Rd_f = 2.5 #median b/w 2.3 and 2.8
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f 

index = df_ordered[ df_ordered['Name']== 'arteries_rleg4_0'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = 0.63 #(mm) match left leg lleg4
Rd_f = 0.63 
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f 

index = df_ordered[ df_ordered['Name']== 'arteries_rleg3_0'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = 0.66 #(mm) match left leg lleg1
Rd_f = 0.66 
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f 

index = df_ordered[ df_ordered['Name']== 'lkidney_arteries18_0'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = 0.875 #(mm) match left leg lleg1
Rd_f = Rd_i 
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f

index = df_ordered[ df_ordered['Name']== 'arteries_rarm8_0'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = 0.6 #(mm) guess
Rd_f = Rd_i 
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f

index = df_ordered[ df_ordered['Name']== 'arteries_rarm3_1'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = 0.7 #(mm) guess
Rd_f = Rd_i 
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f

index = df_ordered[ df_ordered['Name']== 'arteries_lleg28_0'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = 0.38 #(mm) equal to inital value of prior vessel Rd
Rd_f = 0.38 
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f


index = df_ordered[ df_ordered['Name']== 'right_subclavian_artery_2'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = 2.5 #(mm) guess
Rd_f = Rd_i
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f

index = df_ordered[ df_ordered['Name']== 'right_subclavian_artery_1'].index.values[0]   
Ru_i,Rd_i = df_ordered.at[index, 'Radius Values' ]
lam_i =  df_ordered.at[index, 'lam']
dist = lam_i*Ru_i 
Ru_f = Ru_i #(mm) guess
Rd_f =2.7
lam_f = dist/Ru_f

df_ordered.at[index, 'Radius Values' ] = [Ru_f,Rd_f]   
df_ordered.at[index, 'lam'] = lam_f

#%%Manually increase l<0.1

# BROKEN for i in range(0,len(df_ordered)):
#     lam_i = df_ordered.at[i,'lam']
#     Ru_i = df_ordered.at[i,'lam']
#     if lam_i*Ru_i <= 0.01:
#         L= 0.05
#         lam_f = L/Ru_i
#         print(lam_i,lam_f,lam_i*Ru_i)
#         df_ordered.at[i,'lam'] = lam_f
#         print(i)


 
    
 
df_ordered.to_pickle('SysArteries.pkl')       
        
                
    
        
    
    


        
    

def display_two_pointclouds(array_1,array_2,windowname):
    
    p1_pcd = o3d.geometry.PointCloud()
    p1_pcd.points = o3d.utility.Vector3dVector(array_1)
    p1_pcd.paint_uniform_color([1, 0.706, 0])


    p2_pcd = o3d.geometry.PointCloud()
    p2_pcd.points = o3d.utility.Vector3dVector(array_2)
    p2_pcd.paint_uniform_color([0, 0.706, 1])

    
    concate_pc = np.concatenate((array_1, array_2),axis = 0)
    p1_color = np.asarray(p1_pcd.colors)
    p2_color = np.asarray(p2_pcd.colors)

    p4_color = np.concatenate((p1_color,p2_color), axis=0)

    p4_pcd = o3d.geometry.PointCloud()
    p4_pcd.points = o3d.utility.Vector3dVector(concate_pc)
    p4_pcd.colors = o3d.utility.Vector3dVector(p4_color)
    o3d.visualization.draw_geometries([p4_pcd],window_name = windowname)

#Display three point cloud with different colours in one o3d window
def display_three_pointclouds(array_1,array_2,array_3,windowname):
    p1_pcd = o3d.geometry.PointCloud()
    p1_pcd.points = o3d.utility.Vector3dVector(array_1)
    p1_pcd.paint_uniform_color([0, 0, 0])


    p2_pcd = o3d.geometry.PointCloud()
    p2_pcd.points = o3d.utility.Vector3dVector(array_2)
    p2_pcd.paint_uniform_color([38/255, 191/255, 214/255])

    p3_pcd = o3d.geometry.PointCloud()
    p3_pcd.points = o3d.utility.Vector3dVector(array_3)
    p3_pcd.paint_uniform_color([140/225, 76/255, 164/225]) 
    
    
    concate_pc = np.concatenate((array_1, array_2,array_3),axis = 0)
    p1_color = np.asarray(p1_pcd.colors)
    p2_color = np.asarray(p2_pcd.colors)
    p3_color = np.asarray(p3_pcd.colors)
    p4_color = np.concatenate((p1_color,p2_color,p3_color), axis=0)

    p4_pcd = o3d.geometry.PointCloud()
    p4_pcd.points = o3d.utility.Vector3dVector(concate_pc)
    p4_pcd.colors = o3d.utility.Vector3dVector(p4_color)
    o3d.visualization.draw_geometries([p4_pcd],window_name = windowname)