# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:07:20 2025

@author: Cassidy.Northway
"""

import os
import numpy as np



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

#%% FCN: Read edepheader
def read_edepheader(headerfile):
    # read phsp source data: xsrc, ysrc, muindx and num of voxel
    with open(headerfile, 'r') as fid:
        header = fid.read().split()
        header = list(map(float, header))
    header = [header[i:i+4] for i in range(0, len(header), 4)]
    return header

#%% FCN: Read ainflu
def read_ainflu(ainflufile):
    # read ainflu of each parallel job
    with open(ainflufile, 'r') as fid:
        ainflu = float(fid.read().strip().split()[0])
    return ainflu

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






#%%Files names and manual data 

cd = os.getcwd()
num_files  = 80
num_fields = 11
filePath = "VMAT_Files\\4DDose_Raw_Data\\Vessels"

majorDir = os.path.join(cd,filePath)


egsphantfile = os.path.join(cd,'CN_XCAT.egsphant')
egsvoifile = os.path.join(cd,"VMAT_Files\\VOI_Files\\Vessels.egsvoi")
treatmentPlan = os.path.join(cd,"VMAT_Files\\TreatmentPlan.txt")

step_size = 0.02

#%%Load in and process treatmentPlan
fieldsArray = np.zeros([11,2],dtype=object)
with open(treatmentPlan, 'r') as x:
    for i in range(num_fields):
        line = x.readline().split(',')
        fieldsArray[i,0] = line[0]
        fieldsArray[i,1] = float(line[1].strip())



#%%Begin file processing 
egsphantom = readEgsphant(egsphantfile)
  
#For this need to use the the volume and density of each voxel
[xdim,ydim,zdim] = egsphantom.dimensions
[xbnds,ybnds,zbnds] = egsphantom.edges

#%%Read number of VOI
voiArray = []

with open(egsvoifile) as x:
    numVOI = int(x.readline().split()[1])
    for i in range(numVOI):
        voiArray.append(int(x.readline()))
voiArray = np.array(voiArray)        

for i in range(0,7):
    rxTime = fieldsArray[i,1]

    #Determine number of time intervals
    timeArray = np.arange(0,rxTime + step_size, step_size)

    #Create empty aray
    E_array = np.zeros([len(timeArray),len(voiArray)],dtype= 'f')
    uE_array = np.zeros([len(timeArray),len(voiArray)], dtype= 'f')
    ainflu_sum = 0
    fieldname = fieldsArray[i,0] 
    filename = 'VCCN_XCAT_'+fieldname+"_plan02"
    
    if fieldname != 'Chest' or "Foot" not in fieldname: 
        for sub_files in range(1,num_files+1):
        
            headerfile = os.path.join(cd,filePath,filename +"_w" + str(sub_files) + '.edepheader')
            datafile = os.path.join(cd,filePath,filename +"_w" + str(sub_files) + '.edepdat')
            ainflufile = os.path.join(cd,filePath,filename +"_w" + str(sub_files) + '.ainflu')
            header = read_edepheader(headerfile)
            data = read_edepdat(datafile)
            ainflu = read_ainflu(ainflufile)
            
            ainflu_sum = ainflu_sum + ainflu
            
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
            
            for j in range(np.size(eventArray, axis = 1)):
                 new_value = round(eventArray[2,j] / step_size)*step_size
                 eventArray[2,j] = new_value
    
    
    #%% Fill E_Array
            for i in range(np.size(eventArray, axis = 1)):
                voiIndex = np.where(voiArray == eventArray[1,i])[0][0]
                
                timeIndex = np.where(timeArray == eventArray[2,i])[0][0]
                E_array[timeIndex,voiIndex] = E_array[timeIndex,voiIndex] + eventArray[0,i]
                uE_array[timeIndex,voiIndex] = uE_array[timeIndex,voiIndex] + eventArray[0,i]**2
       
            
            #print(sub_files) 
    #%% Generate Mass Array
    massArray = np.zeros_like(E_array)
    
    for i in range(np.size(voiArray)):
        location = voiArray[i]
        x_ind = int((location-1) % xdim)
        y_ind = int(((location-1) //xdim) % ydim)
        z_ind = int(((location-1)//xdim) // ydim)
        voxelVolume = (xbnds[x_ind+1]-xbnds[x_ind])*(ybnds[y_ind+1]-ybnds[y_ind])*(zbnds[z_ind+1]-zbnds[z_ind]) #cm^3
        voxelDensity = egsphantom.densityArray[x_ind,y_ind,z_ind] #g/cm
        voxelMass = voxelDensity * voxelVolume # g
        massArray[:,i] = voxelMass
        
    
    
    #%% Convert from E to dose
    doseArray = (1.602E-10 * E_array) / (massArray * ainflu_sum)    
    unCertArray = (1/E_array)*np.sqrt((ainflu_sum/(ainflu_sum-1))*(uE_array - (E_array**2/ainflu_sum)))
    
       
    finalArray = np.zeros([np.size(timeArray)+1, np.size(voiArray)+1]) 
    finalArray[1:,0] = timeArray
    finalArray[0,1:] = voiArray
    finalArray[1:,1:] = doseArray  
    
    arrayname = os.path.join(cd,fieldname+"_Vessel.npy")
    np.save(arrayname,finalArray)
    print(fieldname)

#%%Feet Arrays

rxTime = fieldsArray[7,1]

#Determine number of time intervals
timeArray = np.arange(0,rxTime + step_size, step_size)

#Create empty array
E_array = np.zeros([len(timeArray),len(voiArray)],dtype= 'f')
uE_array = np.zeros([len(timeArray),len(voiArray)], dtype= 'f')
ainflu_sum = 0
fieldname = fieldsArray[7,0] 
filename = 'VCCN_XCAT_'+fieldname+"_plan02"
for i in range(7,9):
    if "LFoot" in fieldname: 
        for sub_files in range(1,num_files+1):
        
            headerfile = os.path.join(cd,filePath,filename +"_w" + str(sub_files) + '.edepheader')
            datafile = os.path.join(cd,filePath,filename +"_w" + str(sub_files) + '.edepdat')
            ainflufile = os.path.join(cd,filePath,filename +"_w" + str(sub_files) + '.ainflu')
            header = read_edepheader(headerfile)
            data = read_edepdat(datafile)
            ainflu = read_ainflu(ainflufile)
            
            ainflu_sum = ainflu_sum + ainflu
            
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
            
            for j in range(np.size(eventArray, axis = 1)):
                 new_value = round(eventArray[2,j] / step_size)*step_size
                 eventArray[2,j] = new_value
    
    
    #%% Fill E_Array
            for i in range(np.size(eventArray, axis = 1)):
                voiIndex = np.where(voiArray == eventArray[1,i])[0][0]
                
                timeIndex = np.where(timeArray == eventArray[2,i])[0][0]
                E_array[timeIndex,voiIndex] = E_array[timeIndex,voiIndex] + eventArray[0,i]
                uE_array[timeIndex,voiIndex] = uE_array[timeIndex,voiIndex] + eventArray[0,i]**2
       
            
            #print(sub_files) 
    #%% Generate Mass Array
massArray = np.zeros_like(E_array)

for i in range(np.size(voiArray)):
    location = voiArray[i]
    x_ind = int((location-1) % xdim)
    y_ind = int(((location-1) //xdim) % ydim)
    z_ind = int(((location-1)//xdim) // ydim)
    voxelVolume = (xbnds[x_ind+1]-xbnds[x_ind])*(ybnds[y_ind+1]-ybnds[y_ind])*(zbnds[z_ind+1]-zbnds[z_ind]) #cm^3
    voxelDensity = egsphantom.densityArray[x_ind,y_ind,z_ind] #g/cm
    voxelMass = voxelDensity * voxelVolume # g
    massArray[:,i] = voxelMass
    


#%% Convert from E to dose
doseArray = (1.602E-10 * E_array) / (massArray * ainflu_sum)    
unCertArray = (1/E_array)*np.sqrt((ainflu_sum/(ainflu_sum-1))*(uE_array - (E_array**2/ainflu_sum)))

   
finalArray = np.zeros([np.size(timeArray)+1, np.size(voiArray)+1]) 
finalArray[1:,0] = timeArray
finalArray[0,1:] = voiArray
finalArray[1:,1:] = doseArray  

arrayname = os.path.join(cd,"VMAT_Files\\4DDoseData\\" + fieldname + "_Vessel.npy")
np.save(arrayname,finalArray)
print(fieldname)

#%%Feet Arrays
rxTime = fieldsArray[9,1]

#Determine number of time intervals
timeArray = np.arange(0,rxTime + step_size, step_size)

#Create empty aray
E_array = np.zeros([len(timeArray),len(voiArray)],dtype= 'f')
uE_array = np.zeros([len(timeArray),len(voiArray)], dtype= 'f')
ainflu_sum = 0
fieldname = fieldsArray[9,0] 
filename = 'VCCN_XCAT_'+fieldname+"_plan02"
for i in range(9,11):
    if "RFoot" in fieldname: 
        for sub_files in range(1,num_files+1):
        
            headerfile = os.path.join(cd,filePath,filename +"_w" + str(sub_files) + '.edepheader')
            datafile = os.path.join(cd,filePath,filename +"_w" + str(sub_files) + '.edepdat')
            ainflufile = os.path.join(cd,filePath,filename +"_w" + str(sub_files) + '.ainflu')
            header = read_edepheader(headerfile)
            data = read_edepdat(datafile)
            ainflu = read_ainflu(ainflufile)
            
            ainflu_sum = ainflu_sum + ainflu
            
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
            
            for j in range(np.size(eventArray, axis = 1)):
                 new_value = round(eventArray[2,j] / step_size)*step_size
                 eventArray[2,j] = new_value
    
    
    #%% Fill E_Array
            for i in range(np.size(eventArray, axis = 1)):
                voiIndex = np.where(voiArray == eventArray[1,i])[0][0]
                
                timeIndex = np.where(timeArray == eventArray[2,i])[0][0]
                E_array[timeIndex,voiIndex] = E_array[timeIndex,voiIndex] + eventArray[0,i]
                uE_array[timeIndex,voiIndex] = uE_array[timeIndex,voiIndex] + eventArray[0,i]**2
       
            
            #print(sub_files) 
    #%% Generate Mass Array
massArray = np.zeros_like(E_array)

for i in range(np.size(voiArray)):
    location = voiArray[i]
    x_ind = int((location-1) % xdim)
    y_ind = int(((location-1) //xdim) % ydim)
    z_ind = int(((location-1)//xdim) // ydim)
    voxelVolume = (xbnds[x_ind+1]-xbnds[x_ind])*(ybnds[y_ind+1]-ybnds[y_ind])*(zbnds[z_ind+1]-zbnds[z_ind]) #cm^3
    voxelDensity = egsphantom.densityArray[x_ind,y_ind,z_ind] #g/cm
    voxelMass = voxelDensity * voxelVolume # g
    massArray[:,i] = voxelMass
    


#%% Convert from E to dose
doseArray = (1.602E-10 * E_array) / (massArray * ainflu_sum)    
unCertArray = (1/E_array)*np.sqrt((ainflu_sum/(ainflu_sum-1))*(uE_array - (E_array**2/ainflu_sum)))

   
finalArray = np.zeros([np.size(timeArray)+1, np.size(voiArray)+1]) 
finalArray[1:,0] = timeArray
finalArray[0,1:] = voiArray
finalArray[1:,1:] = doseArray  

arrayname = os.path.join(cd,"VMAT_Files\\4DDoseData\\" + fieldname + "_Vessel.npy")
np.save(arrayname,finalArray)
print(fieldname)        
        
       
        
       
        