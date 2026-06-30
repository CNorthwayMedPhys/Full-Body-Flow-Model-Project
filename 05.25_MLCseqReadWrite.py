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
            for j in range(60):
                writeArray.append(dataList[(i*2) + j])
        if mu == eI:
            muList = muList[i+1:]
            dataList = dataList[(i*2 + 2):]
            break
    muArray = []
    
    for i in range(len(writeArray)):
        if i%61 == 0:
            mu = writeArray[i]
            muArray.append(float(mu))
    
    muArray = np.array(muArray)
    muArray = normalizeMU(muArray)
    
    j= 0
    for i in range(len(writeArray)):
        if i%61 == 0:
            writeArray[i] = str(muArray[j]) + "\n"
            j += 1
    
    
   
    cpnumber = int(len(writeArray)/61)

    writename = os.path.join(cd,"VMAT_Files/MCFiles/",filename + "_plan02_MLC.sequence")

    with open(writename, 'w') as fid:
        fid.write("0\n")
        fid.write(str(cpnumber))
        fid.write("\n")
        fid.write("0\n")
        for i in range(len(writeArray)):
            line = writeArray[i]
            fid.write(line)
    fid.close()
    return muList, dataList

#%%
cd = os.getcwd()

filename = "VCCN_XCAT_plan01_MLC.sequence"
filepath = os.path.join(cd,filename)

cpkeyname = "VMAT_Files\MCFiles\cpKey.npy"
cpKey = np.load(os.path.join(cd,cpkeyname))


#%% Read in sequence
with open(filepath) as x:
    lines = x.readlines()
lines = lines [2:]

muList = []
dataList = []
for i in range(len(lines)):
    if i%61 == 0:
        muList.append(lines[i])
    else:
        dataList.append(lines[i])
#%%
filename = "VCCN_XCAT_HeadR"
sI = 0 
eI = float(cpKey[0])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)

#%%
filename = "VCCN_XCAT_HeadL"
sI = eI 
eI = float(cpKey[1])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)        

 #%%
filename = "VCCN_XCAT_Chest"
sI = eI 
eI = float(cpKey[2])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)  

#%%
filename = "VCCN_XCAT_Abdomen"
sI = eI 
eI = float(cpKey[3])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)  

#%%
filename = "VCCN_XCAT_Pelvis"
sI = eI 
eI = float(cpKey[4])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)  
#%%
filename = "VCCN_XCAT_KneeAnt"
sI = eI 
eI = float(cpKey[5])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)         

#%%
filename = "VCCN_XCAT_KneePost"
sI = eI 
eI = float(cpKey[6])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)    

#%%
filename = "VCCN_XCAT_RFootMAO"
sI = eI 
eI = float(cpKey[7])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList)    

#%%
filename = "VCCN_XCAT_RFootRPO"
sI = eI 
eI = float(cpKey[8])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList) 

#%%
filename = "VCCN_XCAT_LFootMAO"
sI = eI 
eI = float(cpKey[9])

muList, dataList = writeSeq(filename,sI,eI,muList, dataList) 

#%%
filename = "VCCN_XCAT_LFootLPO"
sI = eI 
eI = 1

muList, dataList = writeSeq(filename,sI,eI,muList, dataList) 
    

