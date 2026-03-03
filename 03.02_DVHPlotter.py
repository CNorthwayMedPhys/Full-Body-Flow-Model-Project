# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:14:27 2026

@author: Cassidy.Northway

Take dose data from BV simulations and output a DVH of the dose to blood
https://matplotlib.org/stable/gallery/statistics/histogram_cumulative.html
"""

import numpy as np
import matplotlib.pyplot as plt
import os


cd = os.getcwd()
#%% INPUTS

#BV data file name
doseFile = "1FracCo60.npy"

binSize = 0.05 #Gy
plotTitle = "DVH of Circulating Blood After One Fraction"

#%% Load and Process Data

dosePath = os.path.join(cd,doseFile)
doseData = np.load(dosePath)

#Define bin sequence  of size 0.01 Gy
numBins = int(np.ceil(np.max(doseData)/binSize))
binSeq = np.linspace(0,np.ceil(np.max(doseData)),num = numBins, endpoint=True)

#Bin data into histograms
counts,binOut= np.histogram(doseData,bins= binSeq, density = True)

#Plot the Differential DVH
plt.stairs(counts,binOut)
plt.title("Differential DVH")
plt.xlabel("Dose (Gy)")
plt.ylabel("Percentage of Total BVs (%)")

#Bin data into histograms
counts,binOut= np.histogram(doseData,bins= binSeq, density = True)

#Plot the Differential DVH
plt.stairs(counts,binOut)
plt.title("Differential DVH")
plt.xlabel("Dose (Gy)")
plt.ylabel("Percentage of Total BVs (%)")
plt.figure()

#Plot the Cumulativ DVH
plt.hist (doseData, bins = binSeq, density = True, cumulative= -1, histtype= "step")
