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

SweepingRx = 2.36
StaticRx = 2
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
Frac1 = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_1.npy"))*100/SweepingRx
Frac2 = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_2.npy"))*100/SweepingRx
Frac3 = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_3.npy"))*100/SweepingRx
Frac4 = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_4.npy"))*100/SweepingRx
Frac5 = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_5.npy"))*100/SweepingRx
Frac6 = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_6.npy"))*100/SweepingRx
for x in [Frac1,Frac2,Frac3,Frac4,Frac5,Frac6]:
    rng.shuffle(x)
    
#Load in Co60 sweepinh static 
Frac1S = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_Static1.npy"))*100/SweepingRx
Frac2S = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_Static2.npy"))*100/SweepingRx
Frac3S = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_Static3.npy"))*100/SweepingRx
Frac4S = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_Static4.npy"))*100/SweepingRx
Frac5S = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_Static5.npy"))*100/SweepingRx
Frac6S = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_Static6.npy"))*100/SweepingRx
for x in [Frac1S,Frac2S,Frac3S,Frac4S,Frac5S,Frac6S]:
    rng.shuffle(x)

#Sum all fracs to create total treatment
SumFrac = (Frac1+Frac2+Frac3+Frac4+Frac5+Frac6)/6
SumFracS = (Frac1S+Frac2S+Frac3S+Frac4S+Frac5S+Frac6S)/6

#Load in Ext SSD

ExtFrac1 = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results1.npy"))*100/StaticRx
ExtFrac2 = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results2.npy"))*100/StaticRx
ExtFrac3 = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results3.npy"))*100/StaticRx
ExtFrac4 = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results4.npy"))*100/StaticRx
ExtFrac5 = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results5.npy"))*100/StaticRx
ExtFrac6 = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results6.npy"))*100/StaticRx

for x in [ExtFrac1,ExtFrac2,ExtFrac3,ExtFrac4,ExtFrac5,ExtFrac6]:
    rng.shuffle(x)
    
#Load instatic 
ExtFrac1S = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results_Static1.npy"))*100/StaticRx
ExtFrac2S = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results_Static2.npy"))*100/StaticRx
ExtFrac3S = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results_Static3.npy"))*100/StaticRx
ExtFrac4S = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results_Static4.npy"))*100/StaticRx
ExtFrac5S = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results_Static5.npy"))*100/StaticRx
ExtFrac6S = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results_Static6.npy"))*100/StaticRx

for x in [ExtFrac1S,ExtFrac2S,ExtFrac3S,ExtFrac4S,ExtFrac5S,ExtFrac6S]:
    rng.shuffle(x)
    
ExtSumFrac = (ExtFrac1 +ExtFrac2+ExtFrac3+ExtFrac4+ExtFrac5+ExtFrac6)/6
ExtSumFracS = (ExtFrac1S+ExtFrac2S+ExtFrac3S+ExtFrac4S + ExtFrac5S+ExtFrac6S)/6

#%% Calculate Mean and Std
print('Percentage of Dose to BV compared to Dose to Rx point')
print("For Sweeping:")
print("Single Fraction mean and std are " + f"{np.mean(Frac1):.2f}" + " +/- " + f"{np.std(Frac1):.2f}")
print("Six Fraction mean and std are " + f"{np.mean(SumFrac):.2f}" + " +/- " + f"{np.std(SumFrac):.2f}")
print("Static Fraction mean and std are " + f"{np.mean(Frac1S):.2f}" + " +/- " + f"{np.std(Frac1S):.2f}")
print("Sum Static Fraction mean and std are " + f"{np.mean(SumFracS):.2f}" + " +/- " + f"{np.std(SumFracS):.2f}")

print("\n")

print("For Ext SSD:")
print("Single Fraction mean and std are " + f"{np.mean(ExtFrac1):.2f}" + " +/- " + f"{np.std(ExtFrac1):.2f}")
print("Six Fraction mean and std are " + f"{np.mean(ExtSumFrac):.2f}" + " +/- " + f"{np.std(ExtSumFrac):.2f}")
print("Static Fraction mean and std are " + f"{np.mean(ExtFrac1S):.2f}" + " +/- " + f"{np.std(ExtFrac1S):.2f}")
print("Sum Static Fraction mean and std are " + f"{np.mean(ExtSumFracS):.2f}" + " +/- " + f"{np.std(ExtSumFracS):.2f}")

print('\n')
#%% Calculate D95 -> Dose to at least 95% 
print("For Sweeping:")
print("Single Fraction D95 is " + f"{np.percentile(Frac1,95):.2f}")
print("Six Fraction D95 is " + f"{np.percentile(SumFrac,95):.2f}")
print("Static Fraction D95 is " + f"{np.percentile(Frac1S,95):.2f}")
print("Six mStatic Fraction D95 is " + f"{np.percentile(SumFracS,95):.2f}")
print("\n")

print("For Ext SSD:")
print("Single Fraction D95 is " + f"{np.percentile(ExtFrac1,95):.2f}")
print("Six Fraction D95 is " + f"{np.percentile(ExtSumFrac,95):.2f}")
print("Static Fraction D95 is " + f"{np.percentile(ExtFrac1S,95):.2f}")
print("Six mStatic Fraction D95 is " + f"{np.percentile(ExtSumFracS,95):.2f}")
print("\n")
#%% Dose as Percent of Rx
# print("For Sweeping:")
# print("Single Fraction Mean % of Rx " + f"{np.mean(Frac1)/2 * 100:.2f}" +"%")
# print("Six Fraction Mean % of Rx " + f"{np.mean(SumFrac)/12 * 100:.2f}" +"%")
# print("Static Fraction Mean % of Rx " + f"{np.mean(Frac1S)/2 * 100:.2f}" +"%")
# print("Static Fraction Mean % of Rx " + f"{np.mean(SumFracS)/2 * 100:.2f}" +"%")

# print('\n')

# print("For Ext SSD:")
# print("Single Fraction Mean % of Rx " + f"{np.mean(ExtFrac1)/2 * 100:.2f}" +"%")
# print("Six Fraction Mean % of Rx " + f"{np.mean(ExtSumFrac)/12 * 100:.2f}" +"%")
# print("Static Fraction Mean % of Rx " + f"{np.mean(ExtFrac1S)/2 * 100:.2f}" +"%")
# print("Static Fraction Mean % of Rx " + f"{np.mean(ExtSumFracS)/2 * 100:.2f}" +"%")

# print('\n')

#%% Plot the DVHs - 1Frac
fig, ax = plt.subplots()

sns.ecdfplot(data=(ExtFrac1/100), complementary = True, color = 'grey', label = "Stationary - Circulating BVs")
sns.ecdfplot(data=(ExtFrac1S/100), complementary = True,linestyle = '--', color = 'grey', label = "Stationary - Static BVs")

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

sns.ecdfplot(data=(Frac1/100), complementary = True,color = 'black',  label = "Swept - Circulating BVs")
sns.ecdfplot(data=(Frac1S/100), complementary = True,linestyle = '--', color = 'black', label = "Swepth - Static BVs")
# #Load in the DVH data from Eclipse Sweeping
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
ax.set_xlim((0,1.50))
ax.set_xlabel("Relative Dose")
ax.set_ylabel('Relative Volume')
    
    
plt.title('DVH - Single Fraction')   
ax.legend()


#%% Plot the 6 frac DVHs
fig, ax = plt.subplots()

sns.ecdfplot(data=(ExtSumFrac/100), complementary = True, color = 'grey', label = "Stationary - Circulating BVs")
sns.ecdfplot(data=(ExtSumFracS/100), complementary = True,linestyle = '--', color = 'grey', label = "Stationary - Static BVs")

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
#     ax.plot(doseArray*6,volumeArray/100, label = stNames[j],color = 'grey', linestyle = ":" )

ax.set_xlabel("Dose (Gy)")   

sns.ecdfplot(data=(SumFrac/100), complementary = True,color = 'black',  label = "Swept - Circulating BVs")
sns.ecdfplot(data=(SumFracS/100), complementary = True,linestyle = '--', color = 'black', label = "Swept - Static BVs")
# #Load in the DVH data from Eclipse Sweeping
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
# ax.plot(doseArray*6,volumeArray/100, label = stNames[j], color = 'black', linestyle = ':')
ax.set_xlim(0.01,1.30)
ax.set_xlabel("Relative Dose")
ax.set_ylabel('Relative Volume')
    
    
plt.title('DVH - Six Fractions')   
ax.legend()
#%% Plot the gaussian fits - 1 Frac

fig, ax = plt.subplots()

# Fit the Gamma distribution to the data
mu, std = norm.fit(ExtFrac1)
# Create a range of x-values for the plot
x = np.linspace(np.min(ExtFrac1), np.max(ExtFrac1), 1000)
pdf_fitted = norm.pdf(x, mu, std)
# Plot histogram and fitted curve
pdf_fitted = norm.pdf(x, mu, std)
counts, edges = np.histogram(ExtFrac1, bins=30, density = True)
bin_centers = 0.5 * (edges[:-1] + edges[1:])
plt.scatter(bin_centers, counts, marker = "_", color = 'grey')
plt.plot(x, pdf_fitted, color = 'grey', label = 'Stationary - Circulating')


mu, std = norm.fit(ExtFrac1S)
x = np.linspace(np.min(ExtFrac1S), np.max(ExtFrac1S), 1000)
pdf_fitted = norm.pdf(x, mu, std)
# counts, edges = np.histogram(Frac1S, bins=30, density = True)
# bin_centers = 0.5 * (edges[:-1] + edges[1:])

# plt.scatter(bin_centers, counts, marker = "_", color = 'grey')

plt.plot(x, pdf_fitted, '--', color = 'grey',  label = "Stationary - Static")


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
plt.plot(x, pdf_fitted, 'k-', label= "Swept - Circulating")
#plt.axvline(mu, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mu:.2f} Gy')

mu, std = norm.fit(Frac1S)
x = np.linspace(np.min(Frac1S), np.max(Frac1S), 1000)
pdf_fitted = norm.pdf(x, mu, std)
# counts, edges = np.histogram(Frac1S, bins=30, density = True)
# bin_centers = 0.5 * (edges[:-1] + edges[1:])

# plt.scatter(bin_centers, counts, marker = "_", color = 'grey', label = "Static Blood")

plt.plot(x, pdf_fitted, '--', color = 'k', label = 'Swept - Static')

plt.xlim((50,150))

#plt.ylim((0.01,3.5))
plt.title('Relative Dose to Blood Volumes after One Fractions')
plt.xlabel('Relative Dose to Blood Volume (%)')
plt.ylabel('Density')
plt.legend(fontsize="small")
plt.show()

#%% Histogram for all fractions
fig, ax = plt.subplots()

# Fit the Gamma distribution to the data
mu, std = norm.fit(ExtSumFrac)
# Create a range of x-values for the plot
x = np.linspace(np.min(ExtSumFrac), np.max(ExtSumFrac), 1000)
pdf_fitted = norm.pdf(x, mu, std)
# Plot histogram and fitted curve
pdf_fitted = norm.pdf(x, mu, std)
counts, edges = np.histogram(ExtSumFrac, bins=30, density = True)
bin_centers = 0.5 * (edges[:-1] + edges[1:])
plt.scatter(bin_centers, counts, marker = "_", color = 'grey')
plt.plot(x, pdf_fitted, color = 'grey', label = 'Stationary - Circulating')


mu, std = norm.fit(ExtSumFracS)
x = np.linspace(np.min(ExtSumFracS), np.max(ExtSumFracS), 1000)
pdf_fitted = norm.pdf(x, mu, std)
# counts, edges = np.histogram(Frac1S, bins=30, density = True)
# bin_centers = 0.5 * (edges[:-1] + edges[1:])

# plt.scatter(bin_centers, counts, marker = "_", color = 'grey')

plt.plot(x, pdf_fitted, '--', color = 'grey',  label = "Stationary - Static")


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
plt.plot(x, pdf_fitted, 'k-', label= "Swept - Circulating")
#plt.axvline(mu, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mu:.2f} Gy')

mu, std = norm.fit(SumFracS)
x = np.linspace(np.min(SumFracS), np.max(SumFracS), 1000)
pdf_fitted = norm.pdf(x, mu, std)
# counts, edges = np.histogram(Frac1S, bins=30, density = True)
# bin_centers = 0.5 * (edges[:-1] + edges[1:])

# plt.scatter(bin_centers, counts, marker = "_", color = 'grey', label = "Static Blood")

plt.plot(x, pdf_fitted, '--', color = 'k', label = 'Swept - Static')

#plt.xlim((9.5,15.5))
#plt.ylim((0.01,1.5))
plt.title('Relative Dose to Blood Volumes after Six Fractions')
plt.xlabel('Relative Dose to Blood Volume (%)')
plt.ylabel('Density')
plt.legend(fontsize="small")
plt.show()



#%% Compute HI 
#Load in Co60 sweeping 
Frac1 = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_1.npy"))
Frac2 = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_2.npy"))
Frac3 = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_3.npy"))
Frac4 = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_4.npy"))
Frac5 = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_5.npy"))
Frac6 = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_6.npy"))
for x in [Frac1,Frac2,Frac3,Frac4,Frac5,Frac6]:
    rng.shuffle(x)
    
#Load in Co60 sweepinh static 
Frac1S = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_Static1.npy"))
Frac2S = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_Static2.npy"))
Frac3S = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_Static3.npy"))
Frac4S = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_Static4.npy"))
Frac5S = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_Static5.npy"))
Frac6S = np.load(os.path.join(cd,"DoseResults\\Co60Sweeping_Static6.npy"))
for x in [Frac1S,Frac2S,Frac3S,Frac4S,Frac5S,Frac6S]:
    rng.shuffle(x)

#Sum all fracs to create total treatment
SumFrac = (Frac1+Frac2+Frac3+Frac4+Frac5+Frac6)
SumFracS = (Frac1S+Frac2S+Frac3S+Frac4S+Frac5S+Frac6S)

#Load in Ext SSD

ExtFrac1 = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results1.npy"))
ExtFrac2 = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results2.npy"))
ExtFrac3 = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results3.npy"))
ExtFrac4 = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results4.npy"))
ExtFrac5 = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results5.npy"))
ExtFrac6 = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results6.npy"))

for x in [ExtFrac1,ExtFrac2,ExtFrac3,ExtFrac4,ExtFrac5,ExtFrac6]:
    rng.shuffle(x)
    
#Load instatic 
ExtFrac1S = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results_Static1.npy"))
ExtFrac2S = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results_Static2.npy"))
ExtFrac3S = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results_Static3.npy"))
ExtFrac4S = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results_Static4.npy"))
ExtFrac5S = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results_Static5.npy"))
ExtFrac6S = np.load(os.path.join(cd,"DoseResults\\ExtSSD_Results_Static6.npy"))

for x in [ExtFrac1S,ExtFrac2S,ExtFrac3S,ExtFrac4S,ExtFrac5S,ExtFrac6S]:
    rng.shuffle(x)
    
ExtSumFrac = (ExtFrac1 +ExtFrac2+ExtFrac3+ExtFrac4+ExtFrac5+ExtFrac6)
ExtSumFracS = (ExtFrac1S+ExtFrac2S+ExtFrac3S+ExtFrac4S + ExtFrac5S+ExtFrac6S)
print("For Sweeping:")
print("Single Fraction HI is " + f"{calculate_HI(Frac1):.2f}")
print("Six Fraction HI is " + f"{calculate_HI(SumFrac):.2f}")
print("Static Fraction HI is " + f"{calculate_HI(Frac1S):.2f}")
print("Six Static Fraction HI is " + f"{calculate_HI(SumFracS):.2f}")

print('\n')

print("For Ext SSD:")
print("Single Fraction HI is " + f"{calculate_HI(ExtFrac1):.2f}")
print("Six Fraction HI is " + f"{calculate_HI(ExtSumFrac):.2f}")
print("Static Fraction HI is " + f"{calculate_HI(ExtFrac1S):.2f}")
print("Six Static Fraction HI is " + f"{calculate_HI(ExtSumFracS):.2f}")

print('\n')


