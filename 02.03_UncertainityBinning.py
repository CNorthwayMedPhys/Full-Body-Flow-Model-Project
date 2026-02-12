# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:07:20 2025

@author: Cassidy.Northway
"""

import os
import numpy as np
import matplotlib.pyplot as plt
#%%Files names 
cd = os.getcwd()
filename = "timeStepQuery\XCAT_AP_2ms15E9.npy"
filepath = os.path.join(cd,filename)

data = np.load(filepath)
new_data = np.empty([4,0])

error_filter =0.9
#%%Find the unique VOI values
unq_VOI = np.unique(data[1,:])

for VOI_value in unq_VOI:
    single_VOI_data = data[:,(data[1,:] == VOI_value)]
    
    #Make sure sorted in time
    single_VOI_data = single_VOI_data[:,single_VOI_data[2, :].argsort()]

    #Prep for filter
    time = single_VOI_data[2,:]
    dose = single_VOI_data[0,:]
    rel_error = single_VOI_data[3,:]
    

    sum_dose = 0
    abs_error = 0
    new_array = np.array([[0],[VOI_value],[0],[0]])
    for i in range(len(dose)):
        sum_dose = sum_dose + dose[i]
        abs_error = np.sqrt(abs_error**2 + (dose[i]*rel_error[i])**2)
        if abs_error/sum_dose < error_filter:

            new_array= np.append(new_array,[[time[i]],[VOI_value],[sum_dose],[abs_error/sum_dose]],axis=1)
            sum_dose = 0 
            abs_error = 0 

        if i == len(dose)-1:
            new_array= np.append(new_array,[[time[i]],[VOI_value],[0],[0]],axis=1)
    new_data = np.append(new_data, new_array,axis=1)

#np.save( 'XCAT_AP_15E9_relbin.npy',new_data)
    
#%%compare data
old_data = np.load(os.path.join(cd,"timeStepQuery\XCAT_AP_500ms15E9.npy"))

#%%
VOI = new_data[1,5]

old_VOI_data = old_data[:,(old_data[1,:] == VOI)]
old_VOI_data = old_VOI_data[:,np.argsort(old_VOI_data[2,:])]

new_VOI_data = new_data[:,(new_data[1,:] == VOI)]

#Dose Rate?
oldDoseRate=[]
for i in range(np.shape(old_VOI_data)[1]-1):
    doseRate = old_VOI_data[0,i]/(old_VOI_data[2,i+1]-old_VOI_data[2,i])
    oldDoseRate = np.append(oldDoseRate,doseRate)

newDoseRate=[]
for i in range(np.shape(new_VOI_data)[1]-1):
    doseRate = new_VOI_data[2,i]/(new_VOI_data[0,i+1]-new_VOI_data[0,i])
    newDoseRate = np.append(newDoseRate,doseRate)    

plt.plot(new_VOI_data[0,:],new_VOI_data[2,:],linestyle='None',marker='o' , color = 'r')   
plt.plot(old_VOI_data[2,:],old_VOI_data[0,:],linestyle='None' ,marker='o')  
plt.show() 

plt.plot(new_VOI_data[0,:-1],newDoseRate[:],linestyle='None',marker='o' , color = 'r')   
plt.plot(old_VOI_data[2,:-1],oldDoseRate[:],linestyle='None' ,marker='o')  
plt.show() 