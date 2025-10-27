# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:48:02 2024

@author: Cassidy.Northway
"""
###Goal: ID where we have overlapping vessel bifurcations, does it match what I've been seeing in my code

#%%Import

#Import
import os 
import pandas as pd
import numpy as np
import scipy


#%% Import and parse Excel Sheet data

#Import excel sheet
dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path + '\\FlowTracker.xlsx'
#arteries_sheet = pd.read_excel(path, sheet_name = 1)
veins_sheet = pd.read_excel(path, sheet_name = 4)
   
sheet=veins_sheet

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
    name = sheet.at[index,'Filename']
    end_point = sheet.at[index, 'End Point']
    branches = sheet.at[index,'Out Flow']
    
    if name == 'art_carotid':
       skip = 1
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
            seg_name = file_name + '_0'
            #Load the file data
            path = dir_path + '\\FittedVesselsFiles\\' + main_file
            array = np.load(path)
            
            
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
            
            ###MODIFICATION FOR WRITING SEGMENTED BRANCH
            
            #Write the segment names file
            main_file = seg_name + '.npy'
            write_path = dir_path + '\\FittedVesselSegments\\' + main_file
            write_array = array
            np.save(write_path,array)
            ########################################
        #If the vessel does branch we need to subdivide it into segements via an additional data frame
        if seg_tag == True:
            seg_df = pd.DataFrame(columns=['Branch Name','Index of Split','Dist'])
            sub_index = 0
            
            #Get the names of all the files and import data
            main_file = file_name + '_fitted_data.npy'
            main_array = np.load(dir_path + '\\FittedVesselsFiles\\' + main_file)
            branch_files = []
            
            for i in range(0,len(branches)):
                if branches[i] == 'art_carotid':
                    seg_df.loc[len(seg_df)] = {'Branch Name': branches[i] , 'Index of Split': 1, 'Dist': 0.1}  
                else:   
                    branch_row = sheet[sheet['Filename'].str.match(branches[i])].index.values[0]
                    branch_file = sheet.at[sheet.index[branch_row],'Filename']
                    branch_file = branch_file + '_fitted_data.npy'
            
                    
                    branch_array = np.load(dir_path + '\\FittedVesselsFiles\\' + branch_file)
                    
                   
                #Find the nearest points
                    dist_array = scipy.spatial.distance.cdist(main_array[:,0:3],branch_array[:,0:3])
                    dist_array_a = dist_array[:,0]
                    dist_array_b = dist_array[:,-1]
                    if np.min(dist_array_b) < np.min(dist_array_a):
                        branch_array = np.flipud(branch_array)
                        np.save(dir_path + '\\FittedVesselsFiles\\' + branch_file, branch_array)
                        
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
                    sub_name =  file_name + '_' + str(i)
                    final_index = seg_df.at [ i , 'Index of Split']
                    center_array = main_array[intial_index:final_index+1,0:3 ]
                    radius_values = main_array[intial_index:final_index+1,3 ]
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
                    
                    ###MODIFICATION FOR WRITING SEGMENTED BRANCH
                    write_array = np.vstack([center_array.T,radius_values]).T
                    #Write the segment names file
                    
                    main_file = sub_name + '.npy'
                    write_path = dir_path + '\\FittedVesselSegments\\' + main_file
                    np.save(write_path,write_array)
                    
                    ####MODIFY TO WRITE HERE
                else:
                    sub_name =  file_name + '_' + str(i)
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
                    
                    ###MODIFICATION FOR WRITING SEGMENTED BRANCH
                    write_array = np.vstack([center_array.T,radius_values]).T
                   
                    #Write the segment names file
  
                    main_file = sub_name + '.npy'
                    write_path = dir_path + '\\FittedVesselSegments\\' + main_file
                    np.save(write_path,write_array)
                    
                    ####MODIFY TO WRITE HERE
                    


#%% Combine end to start vessels                  


#Just going to hard code solution here for sys arteries.

removal_indices=[]
for i in range (0,len(df)):
    end_condition = df.at[i,'End Condition']
    if end_condition != 'ST' and len(end_condition)==1:
        if end_condition[0] == 'left_arm_artery_0':
            vessel = 'arteries_larm2_0'
        elif end_condition[0] == 'right_arm_artery_0':
            vessel = 'arteries_rarm10_0'
        else:
            vessel = end_condition[0]
        index = df[df['Name'] == vessel].index[0]
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
        

        vessel_array_1 = np.load(dir_path + '\\FittedVesselSegments\\' + name +'.npy')
        vessel_array_2 = np.load(dir_path + '\\FittedVesselSegments\\' + vessel + '.npy' )
        combined_array= np.append(vessel_array_1,vessel_array_2, axis=0)
        np.save(dir_path + '\\FittedVesselSegments\\' + name +'.npy', combined_array)
        print( vessel)
df_updated = df.drop(df.index[removal_indices])
df_updated = df_updated.reset_index(drop=True)
             
            

 

 
    
 
    

                
    
        
    
    


        
