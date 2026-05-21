# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:23:50 2026
I HAVE MANUALLY CODED ALL TRANSITIONS FROM ONE BEAM TO ANOTHER USING THE INDEX VALUES
OF THE controlArray rather than writing a search function
@author: Cassidy.Northway
"""

import os
import numpy as np

#%%Normalize MU function
def normalizeMU (muArray):
    sMU = muArray[0]
    eMU = muArray[-1]
    normMU = (muArray - sMU)/(eMU-sMU)
    return np.round(normMU, decimals = 4)
    

#%%
cd = os.getcwd()
filename = "VCCN_XCAT_plan01.egsinp"
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

#%% First seperate out the first head field
sI = 0
eI = 18

head1Array = controlArray[sI:eI,:]
cpkey.append(head1Array[-1,-1])

newMU = normalizeMU(head1Array[:,-1])
head1Array[:,-1] = newMU

newCP = np.shape(head1Array)[0]
parameters[2] = str(newCP)

writename = os.path.join(cd,"VMAT_Files/MCFiles/Head1/VC_XCAT_Head1_Plan02.egsinp")
with open(writename, 'w') as fid:
    for line in preamble:
        fid.write(line)
    fid.write(",".join(parameters))    
fid.close()    





 
     