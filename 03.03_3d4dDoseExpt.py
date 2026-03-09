# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:07:20 2025

@author: Cassidy.Northway
"""

import os
import numpy as np
import matplotlib.pyplot as plt

#%%Files names and manual data 
cd = os.getcwd()
filename = "4DDoseData\VCData\VC1731501_plan04"
num_files  = 80
egsphantfile = os.path.join(cd,'4DDoseData\VCData\VC1731501.egsphant')
dosefile = os.path.join(cd,"4DDoseData\VCData\VC1731501_plan04.3ddose")

#Treatment time (min/field)
rxTime= 0.45
rxTime = rxTime * 60 #(s)

#Is the volume stacked (Co-60 with filters)?
stacked_flag = 0 #set to 1 if true, set to zero otherwise
stackmap_filename = "StackToUnstackMaps\\APVOIUnstackToStackMapHiRes.npy"
stackmap_path = os.path.join(cd,stackmap_filename)

#Load in the relevant voxel indices
mapping_filename = "VOIMappingArrays\\Hi-Res\\AP\\"
voxelMap = np.zeros((0,2))
for i in range(28,81):
    voxelMapSub =np.load(os.path.join(cd,mapping_filename+str(i)+".npy"))
    voxelMap = np.append(voxelMap,voxelMapSub,axis =0)
mapVoxels = np.unique(voxelMap[:,1])  

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
#%% FCN: "Unstack the VOI values for the phantom"
if stacked_flag == 1:
    voiMap = np.load(stackmap_path)
            
def unstack(voiMap,StackedVoiValue):
    ind = np.where(voiMap[1,:] == int(StackedVoiValue))[0]
    unStackedVOI = voiMap[0,ind[0]]

    return int(unStackedVOI)

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

class dose:
    
    def __init__(self, dimensions, edges, centers, doseArray, unCertArray):
        self._dimensions = dimensions
        self._edges = edges
        self._centers = centers
        self._doseArray = doseArray
        self._unCertArray = unCertArray
    @property
    def dimensions(self):
        return self._dimensions
    @property
    def doseArray(self):
        return self._doseArray

    @property
    def edges(self):
        return self._edges  
    @property
    def unCertArray(self):
        
        return self._unCertArray
def readDose(dosefile):

    #Parses file line by line
    with open(dosefile) as x:
    
        #Read dimensions
        xdim, ydim, zdim = x.readline().split()
        xdim = int(xdim)
        ydim = int(ydim)
        zdim = int(zdim)
        
        #Read voxel edges, not in one consitent line in the complete version so this looks for 
        #lines going from pos to negative
        xedgesList = []
        yedgesList = []
        zedgesList = []
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
        
        #Build dose image
        doseArray = np.zeros((xdim,ydim,zdim))
        doseList = x.readline().strip().split()
        m = 0

        for k in range(zdim):
            for j in range(ydim):
                    for i in range(xdim):
                            doseArray[i,j,k] = doseList[m]
                            m += 1
        unCertArray = np.zeros((xdim,ydim,zdim))                    
        unCertList = x.readline().strip().split()
        m = 0

        for k in range(zdim):
            for j in range(ydim):
                    for i in range(xdim):
                            unCertArray[i,j,k] = unCertList[m]
                            m += 1                    

        
        


        doseObject = dose([xdim, ydim, zdim], [xedgesList, yedgesList, zedgesList], [xcenterList, ycenterList, zcenterList], doseArray,unCertArray) 
    return doseObject

#%%Begin file processing 
egsphantom = readEgsphant(egsphantfile)
doseObj = readDose(dosefile)
   
#For this need to use the the volume and density of each voxel
[xdim,ydim,zdim] = egsphantom.dimensions
[xbnds,ybnds,zbnds] = egsphantom.edges

#What is the uncert at the VOI of interest
[xdimD,ydimD,zdimD] = doseObj.dimensions


#%%

voiIndex = 1934170 #unstacked

x_indD = int((voiIndex-1) % xdimD)
y_indD = int(((voiIndex-1) //xdimD) % ydimD)
z_indD = int(((voiIndex-1)//xdimD) // ydimD)
print("dose to VOI: " + str(doseObj.doseArray[x_indD,y_indD,z_indD]))
print("uncert to VOI: " + str(doseObj.unCertArray[x_indD,y_indD,z_indD]))
#%%
step_size = 0.9
dose_total = 0
sum_E = 0
#%%
for sub_files in range(1,num_files+1):
    headerfile = os.path.join(cd,filename +"_w" + str(sub_files) + '.edepheader')
    datafile = os.path.join(cd,filename +"_w" + str(sub_files)+ '.edepdat')
    ainflufile = os.path.join(cd,filename +"_w" + str(sub_files) +'.ainflu')
    header = read_edepheader(headerfile)
    data = read_edepdat(datafile)
    ainflu = read_ainflu(ainflufile)



#%%Create array of for all interactions 
    eventArray = np.zeros((4, len(data)))
    
    #for each incident particle in header, write particle with MU tag
    j = 0
    for i in range(len(header)):
        particle = header[i]
        muIndex = particle[2]
        numEvents = int(particle[3])
        for k in range(numEvents):
            event = data[j]
            eventArray[:,j] = np.hstack((event,muIndex,0))
            j += 1

#%%Convert MU to time and step size and sort in descending order    
    
    #Sort by MU index, aka in time
    ind = np.argsort(eventArray[2,:])
    eventArray = eventArray[:,ind]
    
    #Converting treatment time to seconds
    eventArray[2,:] = eventArray[2,:] * rxTime #s
    
    # for j in range(np.size(eventArray, axis = 1)):
    #     new_value = round(eventArray[2,j] / step_size)*step_size
    #     eventArray[2,j] = new_value


    
#%%Clean and filter by VOI
    if stacked_flag == 1:
        for i in range(eventArray.shape[1]):
            unstacked = unstack(voiMap,eventArray[1,i])
            eventArray[1,i] = unstacked

    #Remove events which aren't in the voxel of interest 
    eventVoxels = eventArray[1,:]
    mask = np.isin(eventVoxels,voiIndex)
    eventArray = eventArray[:,mask]

#%%Sum identical locations and times to create dose and rel uncert    
    sum_E = np.sum(eventArray[0,:]) 

    location = eventArray[1,0]
    x_ind = int((location-1) % xdim)
    y_ind = int(((location-1) //xdim) % ydim)
    z_ind = int(((location-1)//xdim) // ydim)
    voxelVolume = (xbnds[x_ind+1]-xbnds[x_ind])*(ybnds[y_ind+1]-ybnds[y_ind])*(zbnds[z_ind+1]-zbnds[z_ind]) #cm^3
    voxelDensity = egsphantom.densityArray[x_ind,y_ind,z_ind] #g/cm
    massVoxel = voxelDensity * voxelVolume # g
    voxelDose = sum_E * 1.602E-10 / ainflu / voxelDensity/voxelVolume
    dose_total = voxelDose + dose_total
    print(str(sub_files))
print("Summed 4d dose: " + str(dose_total))           

#%% Plot

density = egsphantom.densityArray 
density[x_ind,y_ind,z_ind] = 7
plt.figure()         
plt.imshow(density[:,:,z_ind], cmap='hot')  

   

print(dose_total *9.76E14 *rxTime )   
print(doseObj.doseArray[x_indD,y_indD,z_indD] * 9.76E14 *rxTime )


    
    

