# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:22:32 2026

@author: Cassidy.Northway
"""
import os
import numpy as np
import matplotlib.pyplot as plt

#%%Load in the relevant voxel indices
cd = os.getcwd()

mapping_filenameAP = "VOIMappingArrays\\Hi-Res\\AP\\"
mapping_filenamePA = "VOIMappingArrays\\Hi-Res\\PA\\"

voxelMaps = np.zeros((0,2))
for i in range(28,127):
    voxelMapAP =np.load(os.path.join(cd,mapping_filenameAP+str(i)+".npy"))
    voxelMapPA =np.load(os.path.join(cd,mapping_filenamePA+str(i)+".npy"))
    try:
        pairedMap = np.vstack((voxelMapAP[:,1],voxelMapPA[:,1]))
    except:
        if len(voxelMapPA) > len(voxelMapAP):
            diff = len(voxelMapPA) - len(voxelMapAP)
            pairedMap = np.vstack((voxelMapAP[:,1],voxelMapPA[:-diff,1]))
        else:
            diff = len(voxelMapAP) - len(voxelMapPA)
            pairedMap = np.vstack((voxelMapAP[:-diff,1],voxelMapPA[:,1]))
            
            
    voxelMaps = np.append(voxelMaps,np.transpose(pairedMap),axis =0)


#%%
def DoseDataSampler (doseArray,time,voi):
    timeRange = int(dst/0.002)
    voiIndex = np.where(doseArray[0,:] == int(voi))[0][0]
    timeIndex = np.where(np.isclose(doseArray[1:,0],time))[0][0] + 1
    dose = np.sum(doseArray[timeIndex:timeIndex+timeRange,voiIndex])
    return dose     
#%%
doseArray = np.append(voxelMaps, np.zeros((7135,1)), axis  = 1 )  

dose_filename_AP = "4DDoseData\\Individual Sweeps\\AP\\Sweep"
dose_filename_PA = "4DDoseData\\Individual Sweeps\\PA\\Sweep"


# for i in range(1,16):
#     doseAPArray = np.load(os.path.join(cd,dose_filename_AP+str(i)+".npy"))
#     dosePAArray = np.load(os.path.join(cd,dose_filename_PA+str(i)+".npy"))
    
#     for j in range(7135):
#         VOIAP = doseArray[j,0]
#         VOIPA = doseArray[j,1]
        
#         voiIndexAP = np.where(doseAPArray[0,:] == int(VOIAP))[0][0]
#         voiIndexPA = np.where(dosePAArray[0,:] == int(VOIPA))[0][0]
        
            
#         doseAP = np.sum(doseAPArray[1:,voiIndexAP]) * 9.76E14 * 0.45
#         dosePA = np.sum(dosePAArray[1:,voiIndexPA]) * 9.76E14 * 0.45
        
#         doseArray[j,2] = doseArray[j,2] + doseAP + dosePA
        
#%%

doseAPArray = np.load(os.path.join(cd,dose_filename_AP+str(1)+".npy"))
dosePAArray = np.load(os.path.join(cd,dose_filename_PA+str(1)+".npy"))

VOIAP = doseArray[0,0]
VOIPA = doseArray[0,1]

voiIndexAP = np.where(doseAPArray[0,:] == int(VOIAP))[0][0]
voiIndexPA = np.where(dosePAArray[0,:] == int(VOIPA))[0][0]

doseAP = np.sum(doseAPArray[1:,voiIndexAP]) * 9.76E14 * 0.45
dosePA = np.sum(dosePAArray[1:,voiIndexPA]) * 9.76E14 * 0.45

dose = doseAP +dosePA

print( "Summed dose: " + str(dose))

for dst in [0.002,0.02,0.2,0.5]:
    doseAP = 0
    timeDoseAP = 0
    dosePA = 0
    timeDosePA = 0
    for time in np.arange(0,(0.45*60), 0.002):
         if int(time*1000) % int(dst*1000) == 0 :
             doseStep = DoseDataSampler (doseAPArray,time,VOIAP )
             #if doseStep > 0:
             doseAP = doseAP + doseStep
             timeDoseAP = timeDoseAP + dst
    for time in np.arange(0,(0.45*60), 0.002):
         if int(time*1000) % int(dst*1000) == 0 :
             doseStep = DoseDataSampler (dosePAArray,time,VOIPA )
             #if doseStep > 0:
             dosePA = dosePA + doseStep
             timeDosePA = timeDosePA + dst   
    doseFinalAP = doseAP * (timeDoseAP/60) * 9.76E14
    doseFinalPA = dosePA * (timeDosePA/60) * 9.76E14
    doseFinal =  doseFinalAP +  doseFinalPA
    print( str(dst))     
    print(str(doseFinal))     
        
        
        
    
    
    

#%%
# dose = doseArray[:,2]
        
# # Calculate mean and standard deviation
# mean = np.mean(dose)
# std_dev = np.std(dose)

# print("mean: " + str(mean))
# print("std: " + str(std_dev))

# # Create the histogram
# plt.hist(dose, bins=25, alpha=0.7, color='skyblue', edgecolor='black', label='Blood Volume Dose')

# # Add vertical lines for mean and standard deviation
# plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2E}')
# plt.axvline(mean - std_dev, color='green', linestyle='dotted', linewidth=2, label=f'1 Std Dev: {std_dev:.2E}')
# plt.axvline(mean + std_dev, color='green', linestyle='dotted', linewidth=2)

# # Add labels and title
# plt.xlabel('Dose (Gy)')
# plt.ylabel('Blood Volume Count')
# plt.title('Co60 One Frac VOI Summed')
# plt.legend()
# plt.grid(axis='y', alpha=0.75)

# # Display the plot
# plt.show()
    