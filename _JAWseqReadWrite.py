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
def writeSeq(filename,sI,eI,muList, dataList):
    writeArray = []
    writeflag = 0
    for i in range(len(muList)):
        mu = muList[i].strip()
        mu = np.round(float(mu),decimals=4)
        if mu == sI:
            writeflag = 1
        if writeflag ==1:
            writeArray.append(muList[i])
            writeArray.append(dataList[i*2])
            writeArray.append(dataList[(i*2)+1])
        if mu == eI:
            muList = muList[i+1:]
            dataList = dataList[(i*2 + 2):]
            break
    muArray = []
    
    for i in range(len(writeArray)):
        if i%3 == 0:
            mu = writeArray[i].strip()
            mu = np.round(float(mu),decimals=4)
            muArray.append(mu)
    
    muArray = np.array(muArray)
    muArray = normalizeMU(muArray)
    
    j= 0
    for i in range(len(writeArray)):
        if i%3 == 0:
            writeArray[i] = str(muArray[j]) + "\n"
            j += 1
    
    
    preamble = filename + "  04"
    cpnumber = int(len(writeArray)/3)

    writename = os.path.join(cd,"allVMAT_Files/MCFiles/",filename + "_plan04_JAW.sequence")

    with open(writename, 'w') as fid:
        fid.write(preamble)
        fid.write("\n")
        fid.write(str(cpnumber))
        fid.write("\n")
        for i in range(len(writeArray)):
            line = writeArray[i]
            fid.write(line)
    fid.close()
    return muList, dataList

#%%
cd = os.getcwd()

filename = "allVMAT_Files\MCFiles\VCCN_XCAT_plan01_JAW.sequence"
filepath = os.path.join(cd,filename)

cpkeyname = "allVMAT_Files\MCFiles\cpKey.npy"
cpKey = np.load(os.path.join(cd,cpkeyname))


#%% Read in sequence
with open(filepath) as x:
    lines = x.readlines()
lines = lines[2:]

muList = []
dataList = []
for i in range(len(lines)):
    if i%3 == 0:
        muList.append(lines[i])
    else:
        dataList.append(lines[i])
#%%
filename = "VCCN_XCAT_Head"
sI = 0 
eI = float(cpKey[0])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)
  

 #%%
filename = "VCCN_XCAT_Chest"
sI = eI 
eI = float(cpKey[1])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)  

#%%
filename = "VCCN_XCAT_Abdomen"
sI = eI 
eI = float(cpKey[2])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)  

#%%
filename = "VCCN_XCAT_Pelvis"
sI = eI 
eI = float(cpKey[3])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)  
#%%
filename = "VCCN_XCAT_Knee"
sI = eI 
eI = float(cpKey[4])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)         

#%%
filename = "VCCN_XCAT_Feet"
sI = eI 
eI = 1

muList, dataList = writeSeq(filename,sI,eI,muList, dataList) 
    

