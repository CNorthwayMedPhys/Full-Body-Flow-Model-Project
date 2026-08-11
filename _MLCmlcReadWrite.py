# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:54:38 2026

@author: Cassidy.Northway
"""
import numpy as np
import os

def normalizeMU (muArray):
    sMU = muArray[0]
    eMU = muArray[-1]
    normMU = (muArray - sMU)/(eMU-sMU)
    return np.round(normMU, decimals = 4)

#%%Write function
def writeMlc(filename,sI,eI,lines):
    
    cpCount = 0
    muArray = []
    for line in lines:
        content = line.split("=")[0]
        if "Index" in content:
            mu = float(line.split("=")[1])
            cpCount +=1
            muArray.append(mu)
            
            if mu == eI:
                break
    
    muArray = normalizeMU(np.array(muArray))
    writename = os.path.join(cd,"allVMAT_Files/MCFiles/",filename + "_plan04.mlc")
    with open(writename, 'w') as fid:
        for i in range(5):
            fid.write(preamble[i])
        newField =  "Number of Fields = "+str(cpCount)+" \n"
        fid.write(newField)
        for i in range(6,9):
            fid.write(preamble[i])
        for i in range(1,cpCount+1):
            fid.write("Field = "+str(i)+"\n")
            fid.write("Index = " + str(muArray[i-1]) + "\n")
            for j in range(2,129):
                fid.write(lines[j])
            lines = lines[129:]    
        
        fid.write("\n")
        fid.write("CRC = 8415\n")
        fid.close()
        
        return lines

#%%
cd = os.getcwd()

filename = "allVMAT_Files\MCFiles\VCCN_XCAT_plan01.mlc"
filepath = os.path.join(cd,filename)

cpkeyname = "allVMAT_Files\MCFiles\cpKey.npy"
cpKey = np.load(os.path.join(cd,cpkeyname))


#%% Read in sequence
with open(filepath) as x:
    lines = x.readlines()
preamble = lines  [0:9]
lines = lines[9:]

#%%
filename = "VCCN_XCAT_Head"
sI = 0 
eI = float(cpKey[0])
lines = writeMlc(filename,sI,eI,lines)

 #%%
filename = "VCCN_XCAT_Chest"
sI = eI 
eI = float(cpKey[1])
lines = writeMlc(filename,sI,eI,lines)

#%%
filename = "VCCN_XCAT_Abdomen"
sI = eI 
eI = float(cpKey[2])

lines = writeMlc(filename,sI,eI,lines)

#%%
filename = "VCCN_XCAT_Pelvis"
sI = eI 
eI = float(cpKey[3])

lines = writeMlc(filename,sI,eI,lines)
#%%
filename = "VCCN_XCAT_Knee"
sI = eI 
eI = float(cpKey[4])

lines = writeMlc(filename,sI,eI,lines)      

#%%
filename = "VCCN_XCAT_Feet"
sI = eI 
eI = 1

lines = writeMlc(filename,sI,eI,lines)

