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
fieldnames = ['Abdomen', 'Chest', 'Pelvis', 'KneeAnt', 'KneePost']#, 'LFoot', 'RFoot']#'HeadL', 'HeadR'
compartmentnumbers = [0,1,2,3,4,5,6,7,8,9,10,12,15,18,19,20,21,22,23,24,25,26,27]

cd = os.getcwd()
filelocation = "VMAT_Files\\4DDose_Raw_Data"
writelocation = "VMAT_Files\\CompartmentData\\"
MUtimeloc = os.path.join(cd,"VMAT_Files\\MUvsTimeArray")

num_files  = 80
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

#%% FCN: Convert MU Index to Time
def MU2Time(MUarray,key):
    timeArray = np.zeros_like(MUarray)
    keyTime = key[2,:]
    keyMU = key[0,:]
    for i in range(len(MUarray)):
        MU = MUarray[i]
        time = np.interp(MU, keyMU, keyTime)
        timeArray[i] = time
        
    return(timeArray)
#%%Begin file processing 
for compartmentnum in compartmentnumbers:
    filename = os.path.join(cd,filelocation,str(compartmentnum))
    for iso in fieldnames:
        isoname = os.path.join(filename,'VCCN_XCAT_'+iso+"_plan02")
        muTime = np.load(os.path.join(MUtimeloc,iso+".npy" ))
        for sub_files in range(1,num_files+1):
            headerfile = os.path.join(cd,isoname +"_w" + str(sub_files) + '.edepheader')
            datafile = os.path.join(cd,isoname +"_w" + str(sub_files)+ '.edepdat')
            header = read_edepheader(headerfile)
            if header == []:
                continue
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
            eventArray[2,:] = MU2Time(eventArray[2,:],muTime)
        
    #%%Create / update histrogram
    
            #Convert data to histogram format
            hist_data = eventArray[2,:]
        
            if sub_files == 1:
            #Create first histogram to fill
                hist = physt.h1(hist_data,'fixed_width', bin_width = 0.05) #bin width in ms
            
            else:
            #Update hist with new data
                hist.fill_n(hist_data)
        
    #%%Now we have a filled histogram of events 
        total = hist.total
        freq_data = hist.frequencies
        norm_freq = freq_data / total
        bin_data = hist.bin_left_edges
        event_trace = np.concatenate(([bin_data], [norm_freq]),axis=0)
    
        plt.plot(event_trace[0,:],event_trace[1,:])
        plt.show()

        #Save the data
        savefilename = os.path.join(cd,writelocation,iso,str(compartmentnum) +'eventTrace.npy')
        np.save(savefilename,event_trace)

#%% For L feet need to sum both arrays
for compartmentnum in compartmentnumbers:
    filename = os.path.join(cd,filelocation,str(compartmentnum))
    tag = 0
    for iso in ['LFootMAO','LFootLPO']: 
        isoname = os.path.join(filename,'VCCN_XCAT_'+iso+"_plan02")
        muTime = np.load(os.path.join(MUtimeloc,"LFoot.npy" ))
        for sub_files in range(1,num_files+1):
            headerfile = os.path.join(cd,isoname +"_w" + str(sub_files) + '.edepheader')
            datafile = os.path.join(cd,isoname +"_w" + str(sub_files)+ '.edepdat')
            header = read_edepheader(headerfile)
            if header == []:
                continue
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
            eventArray[2,:] = MU2Time(eventArray[2,:],muTime)
        
    #%%Create / update histrogram
    
            #Convert data to histogram format
            hist_data = eventArray[2,:]
        
            if sub_files == 1 and tag == 0:
            #Create first histogram to fill
                hist = physt.h1(hist_data,'fixed_width', bin_width = 0.5) #bin width in ms
                tag = 1
            
            else:
            #Update hist with new data
                hist.fill_n(hist_data)
        
#%%Now we have a filled histogram of events 
    total = hist.total
    freq_data = hist.frequencies
    norm_freq = freq_data / total
    bin_data = hist.bin_left_edges
    event_trace = np.concatenate(([bin_data], [norm_freq]),axis=0)

    #Save the data
    savefilename = os.path.join(cd,writelocation,'LFoot',str(compartmentnum) +'eventTrace.npy')
    np.save(savefilename,event_trace)
    
#%% For R feet need to sum both arrays
for compartmentnum in compartmentnumbers:
    filename = os.path.join(cd,filelocation,str(compartmentnum))
    tag = 0
    for iso in ['RFootMAO','RFootRPO']: 
        isoname = os.path.join(filename,'VCCN_XCAT_'+iso+"_plan02")
        muTime = np.load(os.path.join(MUtimeloc,"RFoot.npy" ))
        for sub_files in range(1,num_files+1):
            headerfile = os.path.join(cd,isoname +"_w" + str(sub_files) + '.edepheader')
            datafile = os.path.join(cd,isoname +"_w" + str(sub_files)+ '.edepdat')
            header = read_edepheader(headerfile)
            if header == []:
                continue
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
            eventArray[2,:] = MU2Time(eventArray[2,:],muTime)
        
    #%%Create / update histrogram
    
            #Convert data to histogram format
            hist_data = eventArray[2,:]
        
            if sub_files == 1 and tag == 0:
            #Create first histogram to fill
                hist = physt.h1(hist_data,'fixed_width', bin_width = 0.5) #bin width in ms
                tag = 1
            
            else:
            #Update hist with new data
                hist.fill_n(hist_data)
        
#%%Now we have a filled histogram of events 
    total = hist.total
    freq_data = hist.frequencies
    norm_freq = freq_data / total
    bin_data = hist.bin_left_edges
    event_trace = np.concatenate(([bin_data], [norm_freq]),axis=0)

    #Save the data
    savefilename = os.path.join(cd,writelocation,'RFoot',str(compartmentnum) +'eventTrace.npy')
    np.save(savefilename,event_trace)    