# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:25:12 2026

@author: Cassidy.Northway
"""

import os
import matplotlib.pyplot as plt
import numpy as np

#%%File names and manual inputs
cd = os.getcwd()
stepsizes = ["2ms"]

#%%Load data into arrays MANUAL

datapath= os.path.join(cd,"TimeStepQuery","XCAT_AP_500ms15E9.npy")
data = np.load(datapath)
data_1 = data[3,:]


datapath = os.path.join(cd,"TimeStepQuery","XCAT_AP_IS_100msSweep2IS.npy")
data100 = np.load(datapath)
data_2 = data100[3,:]

datapath= os.path.join(cd,"TimeStepQuery","XCAT_AP_SI_250msSweep5.npy")
data250 = np.load(datapath)
data_3 = data250[3,:]

datapath= os.path.join(cd,"TimeStepQuery","XCAT_AP_IS_500msSweep2IS.npy")
data500 = np.load(datapath)
data_4 = data500[3,:]

#%% Plot

# #Uncert box and whisker plot
# fig=plt.figure()
# ax=fig.subplots()
# bp = ax.boxplot([data_1,data_2,data_3,data_4],showfliers=True)

# plt.show()

# #Plot dose vs uncert
# plt.scatter(data100[3,:],data100[0,:])
# plt.show()

# plt.scatter(data250[3,:],data250[0,:])
# plt.show()

# plt.scatter(data500[3,:],data500[0,:])
# plt.show()

#%%
dataAP = np.load(os.path.join(cd,"4DDoseData\\Individual Sweeps\\AP\\XCAT_AP_250msSweep9.npy"))
dataAP[0,:] = dataAP[0,:] * 9.76E14  * 0.250

doseIndex = np.where(dataAP[0,:] > 2)
voxels = dataAP[1,doseIndex]
# plt.scatter(dataVoxel[3,:],dataVoxel[0,:])
# plt.xlabel("uncert")
# plt.show()

# print(sum(dataVoxel[0,:]))
# print(np.mean(dataVoxel[0,:]))
# plt.scatter(dataVoxel[2,:],dataVoxel[0,:])
# plt.xlabel("time")
# plt.show()
