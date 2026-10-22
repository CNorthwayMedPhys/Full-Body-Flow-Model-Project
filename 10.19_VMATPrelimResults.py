# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:48:29 2026

@author: Cassidy.Northway
"""

import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import seaborn as sns
from scipy.stats import norm

rng = np.random.default_rng()
#%% FCN: Calculate the HI 

def calculate_HI(data):
    x = np.sort(data)                           # X-axis values sorted
    y = np.arange(1, len(data) + 1) / len(data) # Y-axis proportions (0 to 1)
    y = np.flip(y)
    i2 = np.where(y==0.02)[0]
    i98 = np.where(y==0.98)[0]
    i50 = np.where(y == 0.5)[0]
    HI = (x[i2]-x[i98])/x[i50]
    return HI[0]

#%% Load in data
cd = os.getcwd()

#Load in Co60 sweeping 
Frac1 = np.load(os.path.join(cd,"DoseResults\\BVResults_VMAT_0.npy"))
Frac2 = np.load(os.path.join(cd,"DoseResults\\BVResults_VMAT_1.npy"))
Frac3 = np.load(os.path.join(cd,"DoseResults\\BVResults_VMAT_2.npy"))
Frac4 = np.load(os.path.join(cd,"DoseResults\\BVResults_VMAT_3.npy"))
Frac5 = np.load(os.path.join(cd,"DoseResults\\BVResults_VMAT_4.npy"))
Frac6 = np.load(os.path.join(cd,"DoseResults\\BVResults_VMAT_5.npy"))
for x in [Frac1,Frac2,Frac3,Frac4,Frac5,Frac6]:
    rng.shuffle(x)
    
#Load in Co60 sweepinh static 
Frac1S = np.load(os.path.join(cd,"DoseResults\\BVResults_VMAT_static.npy"))
Frac2S = np.load(os.path.join(cd,"DoseResults\\BVResults_VMAT_static_1.npy"))
Frac3S = np.load(os.path.join(cd,"DoseResults\\BVResults_VMAT_static_2.npy"))
Frac4S = np.load(os.path.join(cd,"DoseResults\\BVResults_VMAT_static_3.npy"))
Frac5S = np.load(os.path.join(cd,"DoseResults\\BVResults_VMAT_static_4.npy"))
Frac6S = np.load(os.path.join(cd,"DoseResults\\BVResults_VMAT_static_5.npy"))
for x in [Frac1S,Frac2S,Frac3S,Frac4S,Frac5S,Frac6S]:
    rng.shuffle(x)

#Sum all fracs to create total treatment
SumFrac = Frac1+Frac2+Frac3+Frac4+Frac5+Frac6
SumFracS = Frac1S+Frac2S+Frac3S+Frac4S+Frac5S+Frac6S

#Load in Ext SSD

allFrac1 = np.load(os.path.join(cd,"DoseResults\\BVResults_allVMAT0.npy"))
allFrac2 = np.load(os.path.join(cd,"DoseResults\\BVResults_allVMAT0.npy"))
allFrac3 = np.load(os.path.join(cd,"DoseResults\\BVResults_allVMAT0.npy"))
allFrac4 = np.load(os.path.join(cd,"DoseResults\\BVResults_allVMAT0.npy"))
allFrac5 = np.load(os.path.join(cd,"DoseResults\\BVResults_allVMAT0.npy"))
allFrac6 = np.load(os.path.join(cd,"DoseResults\\BVResults_allVMAT0.npy"))

for x in [allFrac1,allFrac2,allFrac3,allFrac4,allFrac5,allFrac6]:
    rng.shuffle(x)
    
#Load instatic 
allFrac1S = np.load(os.path.join(cd,"DoseResults\\BVResults_allVMAT_Static1.npy"))
allFrac2S = np.load(os.path.join(cd,"DoseResults\\BVResults_allVMAT_Static2.npy"))
allFrac3S = np.load(os.path.join(cd,"DoseResults\\BVResults_allVMAT_Static3.npy"))
allFrac4S = np.load(os.path.join(cd,"DoseResults\\BVResults_allVMAT_Static4.npy"))
allFrac5S = np.load(os.path.join(cd,"DoseResults\\BVResults_allVMAT_Static5.npy"))
allFrac6S = np.load(os.path.join(cd,"DoseResults\\BVResults_allVMAT_Static6.npy"))

for x in [allFrac1S,allFrac2S,allFrac3S,allFrac4S,allFrac5S,allFrac6S]:
    rng.shuffle(x)
    
allSumFrac = allFrac1 +allFrac2+allFrac3+allFrac4+allFrac5+allFrac6
allSumFracS = allFrac1S+allFrac2S+allFrac3S+allFrac4S + allFrac5S+allFrac6S

#%% Calculate Mean and Std
print("For VMAT:")
print("Single Fraction mean and std are " + f"{np.mean(Frac1):.2f}" + " +/- " + f"{np.std(Frac1):.2f}")
print("Six Fraction mean and std are " + f"{np.mean(SumFrac):.2f}" + " +/- " + f"{np.std(SumFrac):.2f}")
print("Static Fraction mean and std are " + f"{np.mean(Frac1S):.2f}" + " +/- " + f"{np.std(Frac1S):.2f}")
print("Sum Static Fraction mean and std are " + f"{np.mean(SumFracS):.2f}" + " +/- " + f"{np.std(SumFracS):.2f}")

print("\n")

print("For all VMAT:")
print("Single Fraction mean and std are " + f"{np.mean(allFrac1):.2f}" + " +/- " + f"{np.std(allFrac1):.2f}")
print("Six Fraction mean and std are " + f"{np.mean(allSumFrac):.2f}" + " +/- " + f"{np.std(allSumFrac):.2f}")
print("Static Fraction mean and std are " + f"{np.mean(allFrac1S):.2f}" + " +/- " + f"{np.std(allFrac1S):.2f}")
print("Sum Static Fraction mean and std are " + f"{np.mean(allSumFracS):.2f}" + " +/- " + f"{np.std(allSumFracS):.2f}")

print('\n')
#%% Calculate D95 -> Dose to at least 95% 
print("For VMAT:")
print("Single Fraction D95 is " + f"{np.percentile(Frac1,95):.2f}")
print("Six Fraction D95 is " + f"{np.percentile(SumFrac,95):.2f}")
print("Static Fraction D95 is " + f"{np.percentile(Frac1S,95):.2f}")
print("Six mStatic Fraction D95 is " + f"{np.percentile(SumFracS,95):.2f}")
print("\n")

print("For allVMAT:")
print("Single Fraction D95 is " + f"{np.percentile(allFrac1,95):.2f}")
print("Six Fraction D95 is " + f"{np.percentile(allSumFrac,95):.2f}")
print("Static Fraction D95 is " + f"{np.percentile(allFrac1S,95):.2f}")
print("Six mStatic Fraction D95 is " + f"{np.percentile(allSumFracS,95):.2f}")
print("\n")
#%% Dose as Percent of Rx
print("For VMAT:")
print("Single Fraction Mean % of Rx " + f"{np.mean(Frac1)/2 * 100:.2f}" +"%")
print("Six Fraction Mean % of Rx " + f"{np.mean(SumFrac)/12 * 100:.2f}" +"%")
print("Static Fraction Mean % of Rx " + f"{np.mean(Frac1S)/2 * 100:.2f}" +"%")
print("Static Six Fraction Mean % of Rx " + f"{np.mean(SumFracS)/12 * 100:.2f}" +"%")

print('\n')

print("For allVMAT:")
print("Single Fraction Mean % of Rx " + f"{np.mean(allFrac1)/2 * 100:.2f}" +"%")
print("Six Fraction Mean % of Rx " + f"{np.mean(allSumFrac)/12 * 100:.2f}" +"%")
print("Static Fraction Mean % of Rx " + f"{np.mean(allFrac1S)/2 * 100:.2f}" +"%")
print("Static Six  Fraction Mean % of Rx " + f"{np.mean(allSumFracS)/12 * 100:.2f}" +"%")

print('\n')
#%% Compute HI 
print("For VMAT:")
print("Single Fraction HI is " + f"{calculate_HI(Frac1):.2f}")
print("Six Fraction HI is " + f"{calculate_HI(SumFrac):.2f}")
print("Static Fraction HI is " + f"{calculate_HI(Frac1S):.2f}")
print("Six Static Fraction HI is " + f"{calculate_HI(SumFracS):.2f}")

print('\n')

print("For allVMAT:")
print("Single Fraction HI is " + f"{calculate_HI(allFrac1):.2f}")
print("Six Fraction HI is " + f"{calculate_HI(allSumFrac):.2f}")
print("Static Fraction HI is " + f"{calculate_HI(allFrac1S):.2f}")
print("Six Static Fraction HI is " + f"{calculate_HI(allSumFracS):.2f}")

print('\n')

#%% Plot the DVHs - 1Frac
fig, ax = plt.subplots()

sns.ecdfplot(data=(allFrac1), complementary = True, color = 'grey', label = "allVMAT - Circulating BVs")
sns.ecdfplot(data=(allFrac1S), complementary = True,linestyle = '--', color = 'grey', label = "allVMAT- Static BVs")

# #Load in the DVH for Ext SSD    
# stNames = [x,x,'Ext SSD - Body',x]#['Body','VOI','Torso','idk' ]
# dose_file = os.path.join(cd,'ExtSSD_Files\\SummedDose.RD\\RD.CN_XCAT.AP.dcm')
# rd_file = pydicom.dcmread(dose_file)
# DVHSeq = rd_file.DVHSequence
# numDVH = len(DVHSeq)
# for j in range(2,3):
#     DVH = DVHSeq[j]
#     DVHData = DVH.DVHData
#     listDVH = list(DVHData)
#     DVHData = np.array(listDVH, dtype=float)
#     doseArray = DVHData[::2]
#     volumeArray = DVHData[1::2] 
#     doseArray[-1] = DVH.DVHMaximumDose

#     for i in reversed(range(1,len(doseArray)-1)):
#         step = doseArray[i-1] 
#         doseArray[i] = doseArray[i+1]-step  

#     DVHArray = np.concat(([volumeArray],[doseArray])) 
#     ax.plot(doseArray,volumeArray/100, label = stNames[j],color = 'grey', linestyle = ":" )
#     ax.set_xlim(0.01,3)
#     ax.set_xlabel("Dose (Gy)")   

sns.ecdfplot(data=(Frac1), complementary = True,color = 'black',  label = "VMAT - Circulating BVs")
sns.ecdfplot(data=(Frac1S), complementary = True,linestyle = '--', color = 'black', label = "VMAT - Static BVs")
#Load in the DVH data from Eclipse Sweeping
# stNames = ['Sweept Beam - Body','Torso (Rx = 2 Gy)' ]
# dose_file = os.path.join(cd,'SweepingCo60//RD.CN_XCAT.TBI_2Field_DVH.dcm')
# rd_file = pydicom.dcmread(dose_file)

# DVHSeq = rd_file.DVHSequence
# numDVH = len(DVHSeq)
# for j in range(0,1):
#     DVH = DVHSeq[j]
#     DVHData = DVH.DVHData
#     listDVH = list(DVHData)
#     DVHData = np.array(listDVH, dtype=float)
#     doseArray = DVHData[::2]
#     volumeArray = DVHData[1::2] 
#     doseArray[-1] = DVH.DVHMaximumDose
#     for i in reversed(range(1,len(doseArray)-1)):
#         step = doseArray[i-1] 
#         doseArray[i] = doseArray[i+1]-step  

#     DVHArray = np.concat(([volumeArray],[doseArray])) 
#     ax.plot(doseArray,volumeArray/100, label = stNames[j], color = 'black', linestyle = ':')
#     ax.set_xlim((0,3))
#     ax.set_xlabel("Dose (Gy)")
#     ax.set_ylabel('Relative Volume')
    
    
plt.title('DVH - Single Fraction')   
ax.legend()


#%% Plot the 6 frac DVHs
fig, ax = plt.subplots()

sns.ecdfplot(data=(allSumFrac), complementary = True, color = 'grey', label = "allVMAT - Circulating BVs")
sns.ecdfplot(data=(allSumFrac), complementary = True,linestyle = '--', color = 'grey', label = "allVMAT - Static BVs")

#Load in the DVH for Ext SSD    
# stNames = [x,x,'Ext SSD - Body',x]#['Body','VOI','Torso','idk' ]
# dose_file = os.path.join(cd,'ExtSSD_Files\\SummedDose.RD\\RD.CN_XCAT.AP.dcm')
# rd_file = pydicom.dcmread(dose_file)
# DVHSeq = rd_file.DVHSequence
# numDVH = len(DVHSeq)
# for j in range(2,3):
#     DVH = DVHSeq[j]
#     DVHData = DVH.DVHData
#     listDVH = list(DVHData)
#     DVHData = np.array(listDVH, dtype=float)
#     doseArray = DVHData[::2]
#     volumeArray = DVHData[1::2] 
#     doseArray[-1] = DVH.DVHMaximumDose

#     for i in reversed(range(1,len(doseArray)-1)):
#         step = doseArray[i-1] 
#         doseArray[i] = doseArray[i+1]-step  

#     DVHArray = np.concat(([volumeArray],[doseArray])) 
#     ax.plot(doseArray*6,volumeArray/100, label = stNames[j],color = 'grey', linestyle = ":" )
#     ax.set_xlim(0.01,18)
#     ax.set_xlabel("Dose (Gy)")   

sns.ecdfplot(data=(SumFrac), complementary = True,color = 'black',  label = "VMAT - Circulating BVs")
sns.ecdfplot(data=(SumFracS), complementary = True,linestyle = '--', color = 'black', label = "VMAT - Static BVs")
#Load in the DVH data from Eclipse Sweeping
# stNames = ['Sweept Beam - Body','Torso (Rx = 2 Gy)' ]
# dose_file = os.path.join(cd,'SweepingCo60//RD.CN_XCAT.TBI_2Field_DVH.dcm')
# rd_file = pydicom.dcmread(dose_file)

# DVHSeq = rd_file.DVHSequence
# numDVH = len(DVHSeq)
# for j in range(0,1):
#     DVH = DVHSeq[j]
#     DVHData = DVH.DVHData
#     listDVH = list(DVHData)
#     DVHData = np.array(listDVH, dtype=float)
#     doseArray = DVHData[::2]
#     volumeArray = DVHData[1::2] 
#     doseArray[-1] = DVH.DVHMaximumDose
#     for i in reversed(range(1,len(doseArray)-1)):
#         step = doseArray[i-1] 
#         doseArray[i] = doseArray[i+1]-step  

#     DVHArray = np.concat(([volumeArray],[doseArray])) 
#     ax.plot(doseArray*6,volumeArray/100, label = stNames[j], color = 'black', linestyle = ':')
#     ax.set_xlim(0.01,18)
#     ax.set_xlabel("Dose (Gy)")
#     ax.set_ylabel('Relative Volume')
    
    
plt.title('DVH - Six Fractions')   
ax.legend()
#%% Plot the gaussian fits - 1 Frac

fig, ax = plt.subplots()

# Fit the Gamma distribution to the data
mu, std = norm.fit(allFrac1)
# Create a range of x-values for the plot
x = np.linspace(np.min(allFrac1), np.max(allFrac1), 1000)
pdf_fitted = norm.pdf(x, mu, std)
# Plot histogram and fitted curve
pdf_fitted = norm.pdf(x, mu, std)
counts, edges = np.histogram(allFrac1, bins=30, density = True)
bin_centers = 0.5 * (edges[:-1] + edges[1:])
plt.scatter(bin_centers, counts, marker = "_", color = 'grey')
plt.plot(x, pdf_fitted, color = 'grey', label = 'allVMAT - Circulating')


mu, std = norm.fit(allFrac1S)
x = np.linspace(np.min(allFrac1S), np.max(allFrac1S), 1000)
pdf_fitted = norm.pdf(x, mu, std)
# counts, edges = np.histogram(Frac1S, bins=30, density = True)
# bin_centers = 0.5 * (edges[:-1] + edges[1:])

# plt.scatter(bin_centers, counts, marker = "_", color = 'grey')

plt.plot(x, pdf_fitted, '--', color = 'grey',  label = "allVMAT - Static")


# Fit the Gamma distribution to the data
mu, std = norm.fit(Frac1)

# Create a range of x-values for the plot
x = np.linspace(np.min(Frac1), np.max(Frac1), 1000)
pdf_fitted = norm.pdf(x, mu, std)

# Plot histogram and fitted curve
pdf_fitted = norm.pdf(x, mu, std)
counts, edges = np.histogram(Frac1, bins=30, density = True)
bin_centers = 0.5 * (edges[:-1] + edges[1:])

plt.scatter(bin_centers, counts, marker = "_", color = 'k')
plt.plot(x, pdf_fitted, 'k-', label= "VMAT- Circulating")
#plt.axvline(mu, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mu:.2f} Gy')

mu, std = norm.fit(Frac1S)
x = np.linspace(np.min(Frac1S), np.max(Frac1S), 1000)
pdf_fitted = norm.pdf(x, mu, std)
# counts, edges = np.histogram(Frac1S, bins=30, density = True)
# bin_centers = 0.5 * (edges[:-1] + edges[1:])

# plt.scatter(bin_centers, counts, marker = "_", color = 'grey', label = "Static Blood")

plt.plot(x, pdf_fitted, '--', color = 'k', label = 'VMAT - Static')

plt.xlim((1,3))

plt.ylim((0.01,3.5))
plt.title('Dose to Blood Volumes after One Fractions')
plt.xlabel('Dose to Blood Volume (Gy)')
plt.ylabel('Density')
plt.legend(fontsize="small")
plt.show()

#%% Histogram for all fractions
fig, ax = plt.subplots()

# Fit the Gamma distribution to the data
mu, std = norm.fit(allSumFrac)
# Create a range of x-values for the plot
x = np.linspace(np.min(allSumFrac), np.max(allSumFrac), 1000)
pdf_fitted = norm.pdf(x, mu, std)
# Plot histogram and fitted curve
pdf_fitted = norm.pdf(x, mu, std)
counts, edges = np.histogram(allSumFrac, bins=30, density = True)
bin_centers = 0.5 * (edges[:-1] + edges[1:])
plt.scatter(bin_centers, counts, marker = "_", color = 'grey')
plt.plot(x, pdf_fitted, color = 'grey', label = 'allVMAT - Circulating')


mu, std = norm.fit(allSumFracS)
x = np.linspace(np.min(allSumFracS), np.max(allSumFracS), 1000)
pdf_fitted = norm.pdf(x, mu, std)
# counts, edges = np.histogram(Frac1S, bins=30, density = True)
# bin_centers = 0.5 * (edges[:-1] + edges[1:])

# plt.scatter(bin_centers, counts, marker = "_", color = 'grey')

plt.plot(x, pdf_fitted, '--', color = 'grey',  label = "allVMAT - Static")


# Fit the Gamma distribution to the data
mu, std = norm.fit(SumFrac)

# Create a range of x-values for the plot
x = np.linspace(np.min(SumFrac), np.max(SumFrac), 1000)
pdf_fitted = norm.pdf(x, mu, std)

# Plot histogram and fitted curve
pdf_fitted = norm.pdf(x, mu, std)
counts, edges = np.histogram(SumFrac, bins=30, density = True)
bin_centers = 0.5 * (edges[:-1] + edges[1:])

plt.scatter(bin_centers, counts, marker = "_", color = 'k')
plt.plot(x, pdf_fitted, 'k-', label= "VMAT - Circulating")
#plt.axvline(mu, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mu:.2f} Gy')

mu, std = norm.fit(SumFracS)
x = np.linspace(np.min(SumFracS), np.max(SumFracS), 1000)
pdf_fitted = norm.pdf(x, mu, std)
# counts, edges = np.histogram(Frac1S, bins=30, density = True)
# bin_centers = 0.5 * (edges[:-1] + edges[1:])

# plt.scatter(bin_centers, counts, marker = "_", color = 'grey', label = "Static Blood")

plt.plot(x, pdf_fitted, '--', color = 'k', label = 'VMAT - Static')

plt.xlim((9.5,15.5))
plt.ylim((0.01,1.5))
plt.title('Dose to Blood Volumes after Six Fractions')
plt.xlabel('Dose to Blood Volume (Gy)')
plt.ylabel('Density')
plt.legend(fontsize="small")
plt.show()

