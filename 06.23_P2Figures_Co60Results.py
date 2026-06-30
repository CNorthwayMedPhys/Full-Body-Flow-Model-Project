# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:48:29 2026

@author: Cassidy.Northway
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
from scipy.stats import norm
import pydicom
import matplotlib.lines as lines

#%% Load in data

cd = os.getcwd()

Frac1 = np.load(os.path.join(cd,"SingleFracCo60.npy"))
Frac2 = np.load(os.path.join(cd,"SingleFracCo60_2.npy"))
Frac3 = np.load(os.path.join(cd,"SingleFracCo60_3.npy"))
Frac4 = np.load(os.path.join(cd,"SingleFracCo60_4.npy"))
Frac5 = np.load(os.path.join(cd,"SingleFracCo60_5.npy"))
Frac6 = np.load(os.path.join(cd,"SingleFracCo60_6.npy"))
FracStatic = np.load(os.path.join(cd,"SingleFracCo60_Static.npy"))


AllFrac = Frac1+Frac2+Frac3+Frac4+Frac5+Frac6
FracStatic = FracStatic[:,0]+FracStatic[:,1]

#%% Calculate Mean and Std
print("Single Fraction mean and std are " + f"{np.mean(Frac1):.2f}" + " +/- " + f"{np.std(Frac1):.2f}")
print("Six Fraction mean and std are " + f"{np.mean(AllFrac):.2f}" + " +/- " + f"{np.std(AllFrac):.2f}")
print("Static Fraction mean and std are " + f"{np.mean(FracStatic):.2f}" + " +/- " + f"{np.std(FracStatic):.2f}")

#%% Calculate D95 -> Dose to at least 95% 
print("Single Fraction D95 is " + f"{np.percentile(Frac1,95):.2f}")
print("Six Fraction D95 is " + f"{np.percentile(AllFrac,95):.2f}")
print("Static Fraction D95 is " + f"{np.percentile(FracStatic,95):.2f}")

#%% Dose as Percent of Rx
print("Single Fraction Mean % of Rx " + f"{np.mean(Frac1)/2 * 100:.2f}" +"%")
print("Six Fraction Mean % of Rx " + f"{np.mean(AllFrac)/12 * 100:.2f}" +"%")
print("Static Fraction Mean % of Rx " + f"{np.mean(FracStatic)/2 * 100:.2f}" +"%")





#%% Plot the DVHs
fig, ax = plt.subplots()

sns.ecdfplot(data=(AllFrac/12), complementary = True, linestyle = '-', label = "BV - All Fractions (Rx = 12 Gy)")
sns.ecdfplot(data=(Frac1/2), complementary = True, linestyle = '--', label = "BV - One Fractions (Rx = 2 Gy)")
sns.ecdfplot(data=(FracStatic/2), complementary = True,linestyle = ':',  label = "BV - One Fractions - Static (Rx = 2 Gy)")

ax.set_title ("Dose Volume Histograms")

ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.set_ylim(0, 1.001)
ax.set_ylabel("Fractional Volume")

ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.set_xlim(0,1.3)
ax.set_xlabel("Fractional Dose")

#%%Load in the DVH data from Eclipse 
stNames = ['Body (Rx = 2 Gy)','Torso (Rx = 2 Gy)' ]
dvhlocation= os.path.join(cd,'RD.CN_XCAT.TBI_2Field_DVH.dcm')
rd_file =  pydicom.dcmread(dvhlocation)
DVHSeq = rd_file.DVHSequence
numDVH = len(DVHSeq)
for j in range(numDVH):
    DVH = DVHSeq[j]
    DVHData = DVH.DVHData
    listDVH = list(DVHData)
    DVHData = np.array(listDVH, dtype=float)
    doseArray = DVHData[::2]
    volumeArray = DVHData[1::2] 
    Dmax = DVH.DVHMaximumDose
    Dmin = DVH.DVHMinimumDose
    
    doseArray[0] = Dmin
    for i in range(1,len(doseArray)):
        step = doseArray[i]
        doseArray[i] = doseArray[i-1]+step  
    doseArray = doseArray/2 #Relative to Rx    
    DVHArray = np.concat(([volumeArray],[doseArray])) 
    lstyle = '-'
    if j == 1:
        lstyle = '-.'
    ax.plot(doseArray,volumeArray/100, label = stNames[j], linestyle = lstyle)

ax.legend()

#%% Plot the gaussian fits

fig, ax = plt.subplots()
# Fit the Gamma distribution to the data
mu, std = norm.fit(AllFrac)

# Create a range of x-values for the plot
x = np.linspace(np.min(AllFrac), np.max(AllFrac), 1000)
pdf_fitted = norm.pdf(x, mu, std)

# Plot histogram and fitted curve
pdf_fitted = norm.pdf(x, mu, std)
counts, edges = np.histogram(AllFrac, bins=30, density = True)
bin_centers = 0.5 * (edges[:-1] + edges[1:])

plt.scatter(bin_centers, counts, marker = "_", color = 'k')
plt.plot(x, pdf_fitted, 'k-', label='Fitted Norm PDF')
#plt.axvline(mu, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mu:.2f} Gy')
plt.title('Dose to Blood Volumes after All Fractions')
plt.xlabel('Dose to Blood Volume (Gy)')
plt.ylabel('Density')
plt.show()

#%% Histogram for one fraction
fig, ax = plt.subplots()
# Fit the Gamma distribution to the data
mu, std = norm.fit(Frac1)

# Create a range of x-values for the plot
x = np.linspace(np.min(FracStatic), np.max(Frac1), 1000)
pdf_fitted = norm.pdf(x, mu, std)

#Generate the histogram data
counts, edges = np.histogram(Frac1, bins=30, density = True)
bin_centers = 0.5 * (edges[:-1] + edges[1:])

# Plot histogram and fitted curve
plt.scatter(bin_centers, counts, marker = "_", color = 'k', label = "Moving Blood ")

plt.plot(x, pdf_fitted, 'k-')
#plt.axvline(mu, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mu:.2f} Gy')
plt.title('Dose to Blood Volumes after One Fractions')
plt.xlabel('Dose to Blood Volume (Gy)')
plt.ylabel('Density')

#Now for static repeat the above
mu, std = norm.fit(FracStatic)
x = np.linspace(np.min(FracStatic), np.max(FracStatic), 1000)
pdf_fitted = norm.pdf(x, mu, std)
counts, edges = np.histogram(FracStatic, bins=30, density = True)
bin_centers = 0.5 * (edges[:-1] + edges[1:])

plt.scatter(bin_centers, counts, marker = "_", color = 'grey', label = "Static Blood")

plt.plot(x, pdf_fitted, '-', color = 'grey')


plt.ylim((0,1.5))
plt.legend()
plt.show()


#%% Calculate the HI 

def calculate_HI(data):
    x = np.sort(data)                           # X-axis values sorted
    y = np.arange(1, len(data) + 1) / len(data) # Y-axis proportions (0 to 1)
    y = np.flip(y)
    i2 = np.where(y==0.02)[0]
    i98 = np.where(y==0.98)[0]
    i50 = np.where(y == 0.5)[0]
    HI = (x[i2]-x[i98])/x[i50]
    return HI[0]

print("Single Fraction HI is " + f"{calculate_HI(Frac1):.2f}")
print("Six Fraction HI is " + f"{calculate_HI(AllFrac):.2f}")
print("Static Fraction HI is " + f"{calculate_HI(FracStatic):.2f}")