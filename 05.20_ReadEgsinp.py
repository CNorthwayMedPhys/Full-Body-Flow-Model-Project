# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:23:50 2026
I HAVE MANUALLY CODED ALL TRANSITIONS FROM ONE BEAM TO ANOTHER USING THE INDEX VALUES
OF THE controlArray rather than writing a search function
@author: Cassidy.Northway
"""

#NOTE TO SELF EDIT THE PHASESPACE WRITING THERE IS AN ERROR!!!!!!!!!




import os
import numpy as np

#%%Normalize MU function
def normalizeMU (muArray):
    sMU = muArray[0]
    eMU = muArray[-1]
    normMU = (muArray - sMU)/(eMU-sMU)
    return np.round(normMU, decimals = 4)
    
#%% Write .egsinp function
def writeEgsinp(name,preamble,parameters,array, postamble):
    
    writename = os.path.join(cd,"VMAT_Files/MCFiles/",name + ".egsinp")

    newPost = postamble[1]
    newPost = newPost.split(',')
    
    newWriteLocation = newPost[2]
    newWriteLocation = newWriteLocation.split("/")
    newWriteLocation[6] = "02"
    newWriteLocation[7] = name + ".txt"
    newWriteLocation = "/".join(newWriteLocation)
    newPost[2] = newWriteLocation
    
    newPhspLocation = newPost[1]
    newPhspLocation = newPhspLocation.split("/")
    newPhsp = newPhspLocation[7].split("_")
    newPhsp = name +"_"+newPhsp[3]+"_"+newPhsp[4]
    newPhspLocation[7] = newPhsp
    newPost[1] = "/".join(newPhspLocation)
    
    postamble[1] = ",".join(newPost)+"\n"
    

    with open(writename, 'w') as fid:
        for line in preamble:
            fid.write(line)
        fid.write(",".join(parameters))
        fid.write("\n")
        for i in range(np.shape(array)[0]):
            line = array[i,:]
            line = [str(x) for x in line]
            fid.write(",".join(line))
            fid.write("\n")
        for line in postamble:
            fid.write(line)
    fid.close()   
    
#%%
cd = os.getcwd()
filename = "VMAT_Files\MCFiles\Plan01Files\VCCN_XCAT_plan01.egsinp"
filepath = os.path.join(cd,filename)

#Will contain all Control Values of interest for splitting up ALL files

cpkey=[]

#%%Read in all file data and parse
with open(filepath) as x:
    lines = x.readlines()
    

preamble = lines[0:5]
postamble = lines[-29:]
#Parameters has number of control points will need to be edited at time of printing
parameters = lines[5]
parameters = [x.strip() for x in parameters.split(",")]

controlPoints = lines[6:-29]

#%%Turn control points into numpy array for easy of searching 
controlArray = np.zeros([len(controlPoints),8])
i = 0

for line in controlPoints:
    string = [x.strip() for x in line.split(",")]
    array = np.array([float(x) for x in string])
    controlArray[i,:] = array
    i += 1

#%% First seperate out the right lat head field
sI = 0
eI = 18

head1Array = controlArray[sI:eI,:]
cpkey.append(head1Array[-1,-1])

newMU = normalizeMU(head1Array[:,-1])
head1Array[:,-1] = newMU

newCP = np.shape(head1Array)[0]
parameters[2] = str(newCP)

name = "VCCN_XCAT_HeadR_plan02"
writeEgsinp(name, preamble, parameters, head1Array, postamble) 

#%% Second left lat head field   
sI = eI
eI = 36

newArray = controlArray[sI:eI,:]
cpkey.append(newArray[-1,-1])

newMU = normalizeMU(newArray[:,-1])
newArray[:,-1] = newMU

newCP = np.shape(newArray)[0]
parameters[2] = str(newCP)

name = "VCCN_XCAT_HeadL_plan02"
writeEgsinp(name, preamble, parameters, newArray, postamble) 

#%% 4 Arcs around the chest
sI = eI
eI = 764

newArray = controlArray[sI:eI,:]
cpkey.append(newArray[-1,-1])

newMU = normalizeMU(newArray[:,-1])
newArray[:,-1] = newMU

newCP = np.shape(newArray)[0]
parameters[2] = str(newCP)

name = "VCCN_XCAT_Chest_plan02"
writeEgsinp(name, preamble, parameters, newArray, postamble) 

#%% 4 Arcs around the Abdomen
sI = eI
eI = 1310

newArray = controlArray[sI:eI,:]
cpkey.append(newArray[-1,-1])

newMU = normalizeMU(newArray[:,-1])
newArray[:,-1] = newMU

newCP = np.shape(newArray)[0]
parameters[2] = str(newCP)

name = "VCCN_XCAT_Abdomen_plan02"
writeEgsinp(name, preamble, parameters, newArray, postamble) 

#%% 4 Arcs around the Pelvis
sI = eI
eI = 1856

newArray = controlArray[sI:eI,:]
cpkey.append(newArray[-1,-1])

newMU = normalizeMU(newArray[:,-1])
newArray[:,-1] = newMU

newCP = np.shape(newArray)[0]
parameters[2] = str(newCP)

name = "VCCN_XCAT_Pelvis_plan02"
writeEgsinp(name, preamble, parameters, newArray, postamble) 

#%%Knee Ant Field
sI = eI
eI = 1872

newArray = controlArray[sI:eI,:]
cpkey.append(newArray[-1,-1])

newMU = normalizeMU(newArray[:,-1])
newArray[:,-1] = newMU

newCP = np.shape(newArray)[0]
parameters[2] = str(newCP)

name = "VCCN_XCAT_KneeAnt_plan02"
writeEgsinp(name, preamble, parameters, newArray, postamble) 

#%%Knee Post Field
sI = eI
eI = 1888

newArray = controlArray[sI:eI,:]
cpkey.append(newArray[-1,-1])

newMU = normalizeMU(newArray[:,-1])
newArray[:,-1] = newMU

newCP = np.shape(newArray)[0]
parameters[2] = str(newCP)

name = "VCCN_XCAT_KneePost_plan02"
writeEgsinp(name, preamble, parameters, newArray, postamble) 

#%%RFoot MAO
sI = eI
eI = 1904

newArray = controlArray[sI:eI,:]
cpkey.append(newArray[-1,-1])

newMU = normalizeMU(newArray[:,-1])
newArray[:,-1] = newMU

newCP = np.shape(newArray)[0]
parameters[2] = str(newCP)

name = "VCCN_XCAT_RFootMAO_plan02"
writeEgsinp(name, preamble, parameters, newArray, postamble) 

#%%RFoot RPO
sI = eI
eI = 1920

newArray = controlArray[sI:eI,:]
cpkey.append(newArray[-1,-1])

newMU = normalizeMU(newArray[:,-1])
newArray[:,-1] = newMU

newCP = np.shape(newArray)[0]
parameters[2] = str(newCP)

name = "VCCN_XCAT_RFootRPO_plan02"
writeEgsinp(name, preamble, parameters, newArray, postamble) 

#%%LFoot MAO
sI = eI
eI = 1936

newArray = controlArray[sI:eI,:]
cpkey.append(newArray[-1,-1])

newMU = normalizeMU(newArray[:,-1])
newArray[:,-1] = newMU

newCP = np.shape(newArray)[0]
parameters[2] = str(newCP)

name = "VCCN_XCAT_LFootMAO_plan02"
writeEgsinp(name, preamble, parameters, newArray, postamble) 

#%%LFoot LPO
sI = eI
eI = -1

newArray = controlArray[sI:eI,:]
#cpkey.append(newArray[-1,-1])

newMU = normalizeMU(newArray[:,-1])
newArray[:,-1] = newMU

newCP = np.shape(newArray)[0]
parameters[2] = str(newCP)

name = "VCCN_XCAT_LFootLPO_plan02"
writeEgsinp(name, preamble, parameters, newArray, postamble) 

#%% Finally save the cpKEY
writename = os.path.join(cd,"VMAT_Files/MCFiles/","cpKey.npy") 

np.save(writename,cpkey)


 
     