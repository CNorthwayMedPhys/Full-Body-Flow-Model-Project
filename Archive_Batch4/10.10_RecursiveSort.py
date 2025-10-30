# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 12:53:10 2025

@author: Cassidy.Northway

"""
import os
import sys
import numpy as np

#%% File names and manual data
cd = os.getcwd()
filename = "MCDAO_pt9_plan02"
num_files  = 3

sorted_flags = np.zeros(num_files-1)

#%% FCN: Check for duplicate location, time events
def has_duplicates(arr):
    seen = {}
    dup_keys = []
    for i in range(np.shape(arr)[1]):
        key = str(arr[:,i])
        if key in seen:
            index = seen.get(key)
            index.append(i)
            seen[key] = index
            dup_keys.append(key)
        else:    
            seen[key] = [i]
    newdict = {k: seen[k] for k in dup_keys}        
    return newdict
#%% Recursively pass through all files
while any(flags!= 1 for flags in sorted_flags ):
    
    #Load in two files at a time
    for i in range(1,num_files):
        file_1 = os.path.join(cd,"4DDoseFiles",filename +"_w" + str(i) +'.npy')
        file_2 = os.path.join(cd,"4DDoseFiles",filename +"_w" + str(i+1) +'.npy')
        
        array_1 = np.load(file_1)
        array_2 = np.load(file_2)
        
        
        #Combine the two files
        array = np.append(array_1,array_2,axis=1)
        
        #Sum shared events
        dup_dict = has_duplicates(array[1:3,:])
        rem_ind = []
        
        if dup_dict != {}:
            for key, value in dup_dict.items():
                sumValues = []
                for ind in value:
                    sumValues.append(array[0,ind])
                    location = array[1,ind]
                    time = array [2,ind]
                    rem_ind.append(ind)
                array = np.append(array,np.vstack([sum(sumValues), location, time]), axis = 1)
                
        #Remove summed values
        array = np.delete(array, rem_ind, axis = 1)
        
        #Sort based on time
        ind = np.argsort(array[2,:], kind='stable')
        sorted_array = array[:,ind]
        
        #Check to see if sorting leads to changes
        if (sorted_array[2,:] == array[2,:]).all():
            sorted_flags[i-1] = 1
            sys.stdout.write ('Flagged')
        array = sorted_array
        
        #Split the array in half
        midpoint = np.shape(array)[1] // 2
        
        new_array_1 = array[:,:midpoint]
        new_array_2 =  array[:,midpoint:]
        
        np.save(os.path.join(cd,"4DDoseFiles",filename +"_w" + str(i) +'.npy'),new_array_1)
        np.save(os.path.join(cd,"4DDoseFiles",filename +"_w" + str(i+1) +'.npy'),new_array_2)

     
        
                            