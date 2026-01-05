# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:07:20 2025

@author: Cassidy.Northway
"""

import os
import numpy as np
import physt 
import matplotlib.pyplot as plt
#%%Files names and manual data 
cd = os.getcwd()
filename = "CompartmentData\\PA\\RawData\\XCAT_PA"
writelocation = "CompartmentData\\PA\\"
compartmentnum = 27

num_files  = 80

#Treatment time (min/field)
rxTime= 6.81
rxTime = rxTime * 60 #(s)

#%% FCN: Read edepheader
def read_edepheader(headerfile):
    # read phsp source data: xsrc, ysrc, muindx and num of voxel
    with open(headerfile, 'r') as fid:
        header = fid.read().split()
        header = list(map(float, header))
    header = [header[i:i+4] for i in range(0, len(header), 4)]
    return header

#%% FCN: Read edepdat
def read_edepdat(datafile):
    with open(datafile, 'r') as fid:
        line1 = fid.readline().strip().split(' ')
        try: 
            line1 =list(filter(None,line1))
        except:
            print('issue')
        dat1 = [float(x) for x in line1]
        ncol = len(dat1)

        edep = []
        for line in fid:
            edep.extend([float(x) for x in line.split()])

        nrow = len(edep) // ncol
        edep = [edep[i * ncol:(i + 1) * ncol] for i in range(nrow)]

        edep = [dat1] + edep

    return edep

#%% FCN: Check for duplicate location, time events
def has_duplicates(arr):
    seen = {}
    dup_keys = []
    for i in range(np.shape(arr)[1]):
        key = str(arr[:,i])
        if key in seen:
            index = seen.get(key)
            index.append(i)
            seen[key] = index
            dup_keys.append(key)
        else:    
            seen[key] = [i]
    newdict = {k: seen[k] for k in dup_keys}        
    return newdict

#%%Begin file processing 
for sub_files in range(1,num_files+1):
    headerfile = os.path.join(cd,filename +"_w" + str(sub_files) + '.edepheader')
    datafile = os.path.join(cd,filename +"_w" + str(sub_files)+ '.edepdat')
    header = read_edepheader(headerfile)
    data = read_edepdat(datafile)

#%%Create array of for all interactions
    eventArray = np.zeros((3, len(data)))
    
    #for each incident particle in header, write particle with MU tag
    j = 0
    for i in range(len(header)):
        particle = header[i]
        muIndex = particle[2]
        numEvents = int(particle[3])
        for k in range(numEvents):
            event = data[j]
            eventArray[:,j] = np.hstack((event,muIndex))
            j += 1 
            
#%%Convert MU to time     

    #Converting treatment time to seconds
    eventArray[2,:] = eventArray[2,:] * rxTime #s
    
#%%Create / update histrogram

    #Convert data to histogram format
    hist_data = eventArray[2,:]
    
    if sub_files == 1:
        #Create first histogram to fill
        hist = physt.h1(hist_data,'fixed_width', bin_width = 0.002) #bin width in ms
        
    else:
        #Update hist with new data
        hist.fill_n(hist_data)
    
#%%Now we have a filled histogram of events 
total = hist.total
hist = hist/total
freq_data = hist.frequencies
bin_data = hist.bin_left_edges
event_trace = np.concatenate(([bin_data], [freq_data]),axis=0)

#Save the data
np.save(os.path.join(cd,writelocation + str(compartmentnum) +'eventTrace.npy'),event_trace)

