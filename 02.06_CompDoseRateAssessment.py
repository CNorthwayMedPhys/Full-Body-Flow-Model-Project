# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 11:05:54 2026

@author: Cassidy.Northway
"""

import os
import numpy as np
import matplotlib.pyplot as plt

#%%Load in files
cd = os.getcwd()
filename = "CompartmentData\\AP\\4eventTrace"
path100 =os.path.join(cd,filename +"100ms.npy")
path500 =os.path.join(cd,filename+"500ms.npy")

data500 = np.load(path500)
data100 = np.load(path100)

#%% Split into values and edges

value100 = data100[1,:]
value500 = data500[1,:]

edges100 = data100[0,:]
edges500 = data500[0,:]

#%% Lets do two bar graphs
plt.bar(edges500, value500,  align='edge', color = 'r')
plt.bar(edges100, value100,  alpha = 0.5, align='edge')

plt.show()
 