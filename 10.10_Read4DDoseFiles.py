# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:07:20 2025

@author: Cassidy.Northway
"""

import os
import numpy as np
#%%Files names and manual data 
cd = os.getcwd()
filename = "MCDAO_pt9_plan02"
num_files  = 3
egsphantfile = os.path.join(cd,'MCDAO_pt9'+ '.egsphant')

#Treatment time (min)
rxTime= 8.14
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
        dat1 = [float(x) for x in line1]
        ncol = len(dat1)

        edep = []
        for line in fid:
            edep.extend([float(x) for x in line.split()])

        nrow = len(edep) // ncol
        edep = [edep[i * ncol:(i + 1) * ncol] for i in range(nrow)]

        edep = [dat1] + edep

    return edep

#%% FCN: Read ainflu
def read_ainflu(ainflufile):
    # read ainflu of each parallel job
    with open(ainflufile, 'r') as fid:
        ainflu = float(fid.read().strip().split()[0])
    return ainflu

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
#%% FCN and class: Read egsphant and voi
class egsphant:
    
    def __init__(self, nmaterials, materialList, dimensions, edges, centers, materialArray, densityArray, VOIArray):
        self._nmaterials = nmaterials
        self._materialList = materialList
        self._dimensions = dimensions
        self._edges = edges
        self._centers = centers
        self._materialArray = materialArray
        self._densityArray = densityArray
        self._VOIArray = VOIArray
        
    @property
    def dimensions(self):
        return self._dimensions
    @property
    def edges(self):
        return self._edges
    @property
    def materialArray(self):
        return self._materialArray
    @property
    def densityArray(self):
        return self._densityArray
    @property
    def VOIArray(self):
        return self._VOIArray
    @VOIArray.setter
    def VOIArray(self,value):
        self._VOIArray = value

def readEgsphant(egsphantfile):    


    #Parses file line by line
    with open(egsphantfile) as x:
        
        #Number of materials
        nmaterials = int(x.readline().strip())
        
        #List materials
        materialList = []
        for i in range(nmaterials):
            material = x.readline().strip()
            materialList.append(material)
            
        #Placeholder values
        estepe = x.readline().strip()
             
        #Read dimensions
        xdim, ydim, zdim = x.readline().split()
        xdim = int(xdim)
        ydim = int(ydim)
        zdim = int(zdim)
        
        #Read voxel edges
        xedgesList = x.readline().split()
        xedgesList = [float(item) for item in xedgesList]
        yedgesList = x.readline().split()
        yedgesList = [float(item) for item in yedgesList]
        zedgesList = x.readline().split()
        zedgesList = [float(item) for item in zedgesList]
        
        #Determine the center of the voxels
        xcenterList = []
        ycenterList = []
        zcenterList = []
        for i in range(xdim):
            edge1 = xedgesList[i]
            edge2 = xedgesList[i+1]
            xcenterList.append((edge1+edge2)/2)
        for i in range(ydim):
            edge1 = yedgesList[i]
            edge2 = yedgesList[i+1]
            ycenterList.append((edge1+edge2)/2)
        for i in range(zdim):
            edge1 = zedgesList[i]
            edge2 = zedgesList[i+1]
            zcenterList.append((edge1+edge2)/2)
        
        #Build material image
        materialArray = np.zeros((xdim,ydim,zdim))
        for k in range(zdim):
            for j in range(ydim+1):
                xRow = x.readline().strip()
                if len(xRow) == xdim:
                    for i in range(xdim):
                        materialArray[i,j,k] = int(xRow[i])
                else:
                    m = 0
                    for i in range(len(xRow)):
                        try:
                            xValue = int(xRow[i])
                            if int(xRow[i]) in range(1,nmaterials):
                                materialArray[m,j,k] = int(xRow[i])
                                m += 1
                        except:
                            pass
        
        #Build density image
        densityArray = np.zeros((xdim,ydim,zdim))
        for k in range(zdim):
            for j in range(ydim+1):
                xRow = x.readline().strip().split()
                xRow = [float(item) for item in xRow]
                if len(xRow) == xdim:
                    for i in range(xdim):
                        densityArray[i,j,k] = xRow[i]


    egsphantObject=egsphant(nmaterials, materialList, [xdim, ydim, zdim], [xedgesList, yedgesList, zedgesList], [xcenterList, ycenterList, zcenterList], materialArray, densityArray, []) 
    return egsphantObject



#%%Begin file processing 
egsphantom = readEgsphant(egsphantfile)

for sub_files in range(1,num_files+1):
    headerfile = os.path.join(cd,filename +"_w" + str(sub_files) + '.edepheader')
    datafile = os.path.join(cd,filename +"_w" + str(sub_files)+ '.edepdat')
    ainflufile = os.path.join(cd,filename +"_w" + str(sub_files) +'.ainflu')
    header = read_edepheader(headerfile)
    data = read_edepdat(datafile)
    ainflu = read_ainflu(ainflufile)



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
    
    
#%%Convert energy to Gy/particle
    
    #Determine mass of each voxel. 
    #For this need to use the the volume and density of each voxel
    [xdim,ydim,zdim] = egsphantom.dimensions
    [xbnds,ybnds,zbnds] = egsphantom.edges
    
    for i in range(len(data)):
        lin_ind = eventArray[1,i]
        x_ind = int((lin_ind-1) % xdim)
        y_ind = int(((lin_ind-1) //xdim) % ydim)
        z_ind = int(((lin_ind-1)//xdim) // ydim)
        voxelVolume = (xbnds[x_ind+1]-xbnds[x_ind])*(ybnds[y_ind+1]-ybnds[y_ind])*(zbnds[z_ind+1]-zbnds[z_ind]) #cm^3
        voxelDensity = egsphantom.densityArray[x_ind,y_ind,z_ind] #g/cm
        massVoxel = voxelDensity / voxelVolume # g
        voxelDose = (1.602E-10 / (massVoxel * ainflu))*eventArray[0,i] #Gy/particle
        eventArray[0,i] = voxelDose #Gy/particle
        
#%%Sum identical locations and times
    dup_dict = has_duplicates(eventArray[1:3,:])
    rem_ind = []
    
    if dup_dict != {}:
        for key, value in dup_dict.items():
            sumValues = []
            for ind in value:
                sumValues.append(eventArray[0,ind])
                location = eventArray[1,ind]
                time = eventArray [2,ind]
                rem_ind.append(ind)
            eventArray = np.append(eventArray,np.vstack([sum(sumValues), location, time]), axis = 1)
            
    #Remove summed values
    eventArray = np.delete(eventArray, rem_ind, axis = 1)        
#%%Convert MU to time and sort in descending order    
    
    #Sort by MU index, aka in time
    ind = np.argsort(eventArray[2,:])
    eventArray = eventArray[:,ind]
    
    #Start by converting treatment time to seconds
    rxTime = rxTime * 60 #(s)
    eventArray[2,:] = eventArray[2,:] * rxTime #s


    np.save(os.path.join(cd,"4DDoseFiles",filename +"_w" + str(sub_files) +'.npy'),eventArray)
    
    
    

