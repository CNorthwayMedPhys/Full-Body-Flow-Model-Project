# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:46:42 2024

@author: cbnor

Linearly interpolates for event frequency from 0.5s histogram data
"""

import numpy as np
import pandas as pd
import os
import sys
import random
import matplotlib.pyplot as plt
from pytictoc import TicToc
t = TicToc()          # Create an instance
t.tic()   
#%%Parameters
dt = 0.002 #BFS Time step size (s)
dst = 0.1 #Dose sample time step size (s)
T = 0.955 #Length of one period (s)
BV_num = 1E1 #total number BV
Tss =  round(400*T,3) #Time to reach Steady State (s) #Needs to be a round nubmer!!!
Tfield = round((0.45*15) * 60, 3) #Time per field (s)

#Dose file locations 
cd = os.getcwd()
dose_filename_AP = "4DDoseData\\Individual Sweeps\\PA\\XCAT_PA_Sweep"

#Get Voxel Mapping Array
mapping_filename_AP = "VOIMappingArrays\\Hi-Res\\PA\\"

#Compartment data locations
compartment_AP = "CompartmentData\\PA\\"


#%% Utility functions

def colToExcel(col): # col is 1 based
    excelCol = str()
    div = col 
    while div:
        (div, mod) = divmod(div-1, 26) # will return (x, 0 .. 25)
        excelCol = chr(mod + 65) + excelCol

    return excelCol

def periodic(t, T):
    """
    Returns equivalent time of the first period if more than one period is simulated.
    
    :param t: Time.
    :param T: Period length.
    """
    while t/T > 1.0:
        t = t - T
    return np.round(t,decimals=3)
        
def interp(x,x0,x1,y0,y1):
    
    y = y0 + (x-x0)*((y1-y0)/(x1-x0))
    
    return y

def velocity_interp(flowdata,t,x):
    xarray = flowdata[0,1:]
    tarray = flowdata[1:,0]
    varray = flowdata[1:,1:]
    it = np.searchsorted(tarray,t,side = 'left')
    ix = np.searchsorted(xarray,x,side = 'right')
    x1 = xarray[ix]
    x0 = xarray[ix-1]
    t1 = tarray[it]
    t0 = tarray[it-1]
    if it == 1 and ix == 1:
        p00 = 0

    else:
        p00 = varray[it-1,ix-1]
    p10 = varray[it,ix-1]
    p01 = varray[it-1,ix]
    p11 = varray[it,ix]
    p = p00 + (p10-p00)*((t-t0)/(t1-t0))+(p01-p00)*((x-x0)/(x1-x0))+\
        (p11-p01-p10+p00)*((t-t0)/(t1-t0))*((x-x0)/(x1-x0))
    return p


def pathselecter(outflow,splittingratios,time):
    #need to pull time indexs and interpolate
    num = random.random()
    timearray = splittingratios[:,0]
    index = np.searchsorted(timearray,time,side = 'right')
    aSR = splittingratios[:,1]
    bSR = splittingratios[:,2]
    if index == 478:
        pa = float(aSR[-1])
        pb = float(bSR[-1])
    else:
        pa = interp(time,timearray[index-1],timearray[index],aSR[index-1],aSR[index])
        pb = interp(time,timearray[index-1],timearray[index],bSR[index-1],bSR[index])
    if num <= pa:
        key = outflow[0]
    elif num > pa and num <= pa + pb:
        key = outflow[1]
    else:
        key = outflow[2]
    return key
        
def VOISelector (distance, VOIMapArray):
    #First look for two possible values
    index = np.searchsorted(VOIMapArray[:,0],distance, side ='right') 
    VOI = VOIMapArray[index-1,1]
    return VOI  

def DVHSampler(DVHArray):
    volume = random.random()*100 #convert to percentages
    DVHArray = np.fliplr(DVHArray)
    index = np.searchsorted(DVHArray[0,:],volume)
    value = np.interp(volume, [DVHArray[0,index-1], DVHArray[0,index]],[DVHArray[1,index-1],DVHArray[1,index]])
    return value #Gy

def DoseRateSampler(EventFreqArray, time):
    if time < EventFreqArray[0,0]:
        DoseRate = 0
    elif time >= EventFreqArray[0,-1]:
        DoseRate = 0
    else: 
        index = np.searchsorted(EventFreqArray[0,:],time, side ='right')
        DoseRate =  EventFreqArray[1,index-1]
    return DoseRate #1/s
    
def DoseDataSampler (doseArray,time,voi):
    voiArray = doseArray[:,(doseArray[1,:]==voi)]
    if np.shape(voiArray)[1] == 0:
        dose = 0 
    else:    
        index = np.searchsorted(voiArray[0,:],time,side='left')
        if index == np.shape(voiArray)[1]:
            dose = 0 
        else:    
            dose = voiArray[2,index]
    return dose 
#%%Location class

class Location (object):
    """
    Class for the locations at which all blood volumes pass through.
    """
    
    def __init__(self, IDnum, outflow, dwelltime, splittingratiokey):
        """
        Location constructors
        """
        self._IDnum = IDnum
        self._outflow = outflow
        self._dwelltime = dwelltime
        self._splittingratiokey = splittingratiokey
        self._flowdata = None
        self._splittingratios = None
        self._VOIMap = None
        self._DVH = None
        self._EventFreq = None
        
    def IntFlowData (self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '\\BVSimulationFiles\\'+ str(self.IDnum) + '.xlsx'
        df = pd.read_excel(path, header = None)
        self._flowdata = df.to_numpy()

            
        
    def IntSplittingRatio (self, SRdf):
        SRarray = np.array(SRdf.loc[2:,'A'],dtype = float, ndmin=2).T
        SRdata = SRdf.loc[2:,self.splittingratiokey]
        SRdata = [i.split(';') for i in SRdata]
        SRdata = np.array(SRdata, dtype = float)
        SRarray = np.append(SRarray,SRdata,axis=1)
        self._splittingratios = SRarray
        
    def IntVOIMap (self, number,xx): 
        if xx == 'AP':
            VOIMappingArrayPath = os.path.join(cd,mapping_filename_AP,str(number)+'.npy')
        VOIMappingArray = np.load(VOIMappingArrayPath)
        VOIMappingArray[:,0] = VOIMappingArray[:,0] / 1000 #mm -> m
        self._VOIMap = VOIMappingArray
        
    def IntDVH (self,number, xx):
        if number == 11:
            number = 10
        elif number == 13 or number == 14:
            number = 12
        elif number == 16 or number == 17:
            number = 15    
        if xx == 'AP':
            DVHArrayPath = os.path.join(cd,compartment_AP,str(number)+'DVH.npy')
        DVHArray = np.load(DVHArrayPath)  
        self._DVH = DVHArray
    def IntEventFreq (self,number, xx):
            if number == 11:
                number = 10
            elif number == 13 or number == 14:
                number = 12
            elif number == 16 or number == 17:
                number = 15    
            if xx == 'AP':
                EventFreqArrayPath = os.path.join(cd,compartment_AP,str(number)+'eventTrace100ms.npy')
            EventFreqArray = np.load(EventFreqArrayPath)  
            self._EventFreq = EventFreqArray
        
    @property
    def IDnum(self):
        """
        The ID number used to represent the location
        """
        return self._IDnum
    
    @property
    def outflow(self):
     """
     An array of 2 or 3 integers which are the IDnum keys for the outflow 
     of this location
     """
     return self._outflow

    @property
    def dwelltime (self):
     """
     The dwell time (s) of this location (=0 if a vessel with flowrate)
     """
     return self._dwelltime  
    
    
    @property
    def splittingratiokey (self):
     """
     The column ID for the splitting ratio values needs for this array (=0 if not applicable)
     """
     return self._splittingratiokey

    @property
    def flowdata(self):
        """
        An 2D array containing the flow data where row 0 = location (m) and
        col 0 = time(s). Note: 0,0 = None 
        """
        return self._flowdata
    
    @property
    def splittingratios(self):
      """
      A 2D array containing the splitting ratios for the outflow, col = 0 is 
      time (s) and col= 1,2,3 are the splitting ratios (%)
      """  
      return self._splittingratios
  
    @property
    def VOIMap(self):
        """
        A 2D array containing the distance to voxel maps, col = 0 is distance 
        along the vessel (m) and col =1 is the voxel index for the phantom
        """
        return self._VOIMap
    
    @property
    def DVH(self):
        """
        A 2D array of the DVH for each compartment [2,n]
        [0,:] is the Volume in percentage [100,0] w/ 0.0001 prescision
        [1,:] is the dose value in Gy, w/ 0.0025 step size

        """
        return self._DVH
    
    @property
    def EventFreq(self):
        """
       A 2D Compartment "dose rate" arrays. Every energy event is binned into 0.002 s bins
       size is [2,n]
       [1,:]  is the number of events/total number of events
       [2,:] is the LEFT bin edge (recall bin width = 2ms) 
        """
        return self._EventFreq 
    
#%% BV class

class BloodVolume (object):
    """
    Class represents a single Blood Volume to be tracked
    
    """
    
    def __init__ (self, ID, location):
        """
        Blood volume constructor
        """
        self._ID =ID
        self._location = location
        self._tottime = 0
        self._dwelltime = 0
        self._distance = 0
        self._dose = 0 
        
        
    @property
    def ID(self):
        """
        ID number of the BV
        """
        return self._ID
    
    @property
    def location(self):
        """
        ID number of the current location of the BV
        """
        return self._location
    
    @property
    def tottime(self):
        """
        Total time BV has been within the simulation
        """
        return self._tottime
    
    @property
    def dwelltime(self):
        """
        Time spent dwelling within a location
        """
        return self._dwelltime
    
    @property
    def distance(self):
        """
        Distance down the length of a vessel
        """
        return self._distance
    
    @distance.setter
    def distance(self, value):
        if np.isnan(value):
            print('nan value for distance')
        self._distance = value  
    
    @property 
    def dose(self):
        """
        Dose to the BV
        """
        return self._dose
    
    @dose.setter 
    def dose(self,value):
        self._dose = value


    
#%% Define Network Class

class Network (object):
    """
    Class representing the entire network of blood volumes and locations
    """
    def __init__ (self, dt, BV_num):
        self._Nettime = 0
        self._progress = 0
        self._dt = dt
        self._BVcount = 0
        self._BVs = []
        self._locations = []
        self._currentDoseData = None
        
    def intializeLocations (self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + "\\FlowTracker.xlsx"
        df = pd.read_excel(path)
        for index, row in df.iterrows():
            data = row['LocationIDNumber;OutFlowA;OutFlowB;OutFlowC;DwellTime(s);SplittingRatioKey']
            data = data.split(';')
            if int(data[0]) == 12 or int(data[0]) == 15:
                DTmod = 1.16222463
            elif int(data[0]) == 13 or int(data[0]) == 16:
                DTmod = 1.21396462
            elif int(data[0]) == 14 or int(data[0]) == 17:
                DTmod = 1.07799579
            elif int(data[0]) == 18 or int(data[0]) == 22:
                DTmod = 0.76160822
            elif int(data[0]) == 19 or int(data[0]) == 23:
                DTmod = 0.65386433
            elif int(data[0]) == 20 or int(data[0]) == 24:
                DTmod = 1.0146046
            elif int(data[0]) == 21 or int(data[0]) == 25:
                DTmod =  1.00537915    
            elif int(data[0]) == 26 or int(data[0]) == 27:
                DTmod = 0.58857567 
            elif int(data[0]) == 2:
                DTmod = 1.5
            elif int(data[0]) == 3:
                DTmod = 0.69882772
            elif int(data[0]) == 5:
                DTmod = 0.51878497   
            else:
                DTmod = 1
            self.locations.append(Location(int(data[0]),[int(data[1]),int(data[2]),int(data[3])],float(data[4])*DTmod,data[5]))
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '\\SplittingRatios.xlsx' 
        SRdf =  pd.read_excel(path, header = None)
        SRdf = SRdf.rename(index=lambda x: x+1, columns=lambda y: colToExcel(y+1) )
        for location in self.locations:
            if location.IDnum > 27:
                location.IntFlowData()
                location.IntVOIMap (location.IDnum, 'AP')
            else:
                location.IntDVH(location.IDnum, 'AP')
                location.IntEventFreq(location.IDnum, 'AP')
            if location.splittingratiokey != '0':
                location.IntSplittingRatio (SRdf)
        print('\n Location intialization complete')     
        
    def intializeBV(self):
        self.BVs.append(BloodVolume(self.BVcount, 28))
        self._BVcount += 1
            
    def runNT (self):
        flag = 0
    ##################################Establish Steady State#####################
        #Run until we have reached our steady state
        while self.Nettime < self.Tss:
            
            #Check to see if we have the desired number of BVs. 
            #If not release another BV into the system 
            if self.BVcount < BV_num:
       
                self.intializeBV()
            if self.BVcount == BV_num and flag == 0:
                flag = 1
            
            #Calculate the t w/in the period for table look ups
            pt = periodic(self.Nettime, self.T) 
            
            #Iterate through each BV
            for BV in self.BVs:
                
                #BV determine their location type
                clocation = self.locations[BV.location]
                
                if clocation.IDnum > 27: #outside of an organ
                   #Determine velocity value at exact position and time
                    velocity = velocity_interp(clocation.flowdata,pt,BV.distance)
                   
                   #Update position and exp time
                    newdistance = velocity * self.dt + BV.distance
    
                   #Determine wether the BV is in the vessel
                    if newdistance > clocation.flowdata[0,-1]: 
                       
                       #Move BV to next location
                       if clocation.splittingratios is None:
                           BV._location = clocation.outflow[0]
                           BV._dwelltime = 0
                           BV._distance = 0
                       else:
                           BV._location = pathselecter(clocation.outflow,clocation.splittingratios,pt)
                           BV._dwelltime = 0
                           BV._distance = 0
                   #Advance BV down vessel        
                    else:
                       BV._distance = newdistance
                       
                
                #BV within organ
                else:
                    #Stays in organ
                    if BV.dwelltime < clocation.dwelltime:
                        BV._dwelltime += self.dt
                    #Leaves organ    
                    else:
                        if clocation.splittingratios is None:
                            BV._location = clocation.outflow[0]
                            BV._dwelltime = 0
                        else:
                            BV._location = pathselecter(clocation.outflow,clocation.splittingratios,pt)
                            BV._dwelltime = 0
                BV._tottime += self.dt    
            self.timestep()
            self.print_status(self.Nettime,self.Tss)
################ Steady State Established ########################
        print( '\n Steady State Established')
        self._progress = 0
        localTime = 0
        localTimeSweep = 0
        #Load in the first set of dose data
        current_file = 2
        self._currentDoseData = np.load(os.path.join(cd,dose_filename_AP+"1.npy"))
     
############# Begin AP Field ###########################
        while localTime < self.Tfield:
    
            #Check to see if we need to update the dose date
            if localTimeSweep > (0.45*60) and current_file != 16:
                 self._currentDoseData = np.load(os.path.join(cd,dose_filename_AP+str(current_file)+".npy" ) )
                 current_file += 1
                 localTimeSweep = 0.002
                 flag = 0


            #Calculate the t w/in the period for table look ups
            pt = periodic(self.Nettime, self.T) 
            
            #Iterate through each BV
            for BV in self.BVs:
                # #BV determine their location type
                # clocation = self.locations[BV.location]
                
                # if clocation.IDnum > 27: #outside of an organ
                #    #Determine velocity value at exact position and time
                #     velocity = velocity_interp(clocation.flowdata,pt,BV.distance)
                   
                #    #Update position and exp time
                #     newdistance = velocity * self.dt + BV.distance
    
                #    #Determine wether the BV is in the vessel
                #     if newdistance > clocation.flowdata[0,-1]: 
                       
                #        #Move BV to next location
                #        if clocation.splittingratios is None:
                #            BV._location = clocation.outflow[0]
                #            BV._dwelltime = 0
                #            BV._distance = 0
                #        else:
                #            BV._location = pathselecter(clocation.outflow,clocation.splittingratios,pt)
                #            BV._dwelltime = 0
                #            BV._distance = 0
                #    #Advance BV down vessel        
                #     else:
                #        BV._distance = newdistance

                # #BV within organ
                # else:
                #     #Stays in organ
                #     if BV.dwelltime < clocation.dwelltime:
                #         BV._dwelltime += self.dt
                #     #Leaves organ    
                #     else:
                #         if clocation.splittingratios is None:
                #             BV._location = clocation.outflow[0]
                #             BV._dwelltime = 0
                #         else:
                #             BV._location = pathselecter(clocation.outflow,clocation.splittingratios,pt)
                #             BV._dwelltime = 0 
                            
                ######ADD DOSE HERE #####                
                if int(localTime*100) % int(dst*100) == 0:
                    if BV.location > 27:
                        newlocation = self.locations[BV.location]
                        VOIArray = newlocation.VOIMap
                        VOI = VOISelector(BV.distance, VOIArray)
                        energy_index = np.where((self.currentDoseData[1,:] == VOI) & (self.currentDoseData[2,:] == localTimeSweep))[0]
                        if energy_index.size > 0:
                            doseStep = self.currentDoseData[0,energy_index] * 9.76E14  * (dst) #/60)
                            BV._dose = BV.dose + doseStep

                    else:
                       doselocation = self.locations[BV.location]
                       DVHArray = doselocation.DVH
                       doseDVH = DVHSampler(DVHArray) #Gy
                       DoseRateArray = doselocation.EventFreq
                       doseRate = DoseRateSampler(DoseRateArray, localTime) #1/s
                       doseStep = doseDVH * doseRate * dst 
                       BV._dose = BV.dose + doseStep

                   
                   
                   

            self.timestep()
            localTime += self.dt
            localTime = round(localTime,3)
            localTimeSweep += self.dt
            localTimeSweep = round(localTimeSweep,3)
            self.print_status(localTime,self.Tfield)
        
######## DONE ##############
    def shuffleBVs (self):
        """
        Take the BVs current positions and shuffle them amongst themselves to
        simulate 
        """
        BV_data = self.BVs
        position_data = np.zeros((len(BV_data),4))
        i = 0
        #Write existing BV into an array
        for BV in BV_data:
            loc = BV.location
            if loc > 27: #In blood vessels
                dist = BV.distance
                tag = 0
                position_data[i,:] = [loc, 0, dist, tag]
            else: #In comp
                dt = BV.dwelltime
                tag = 1
                position_data[i,:] =[loc, dt, 0, tag]
            i += 1
        #Shuffle position data   
        np.random.shuffle(position_data)
        
        #Redist locations
        i = 0
        for BV in BV_data:
            loc = position_data[i,0]
            tag = position_data[i,3]
            
            if tag == 0: #Vessels
                BV._loction = int(loc)
                BV._dwelltime = 0
                BV._distance = position_data[i,2]
            else: #Compartment
                BV._location = int(loc)
                BV._distance = 0
                BV._dwelltime = position_data[i,1]
        self._BVs = BV_data         
            
            
        
        
        
        
    def setTimes(self, T, Tss, Tfield):
        """
        Sets timing parameters for the network 
        :param Tss: Time to elapse to reach Steady State
        :param T: Length of one periodic cycle.
        :param Tfield: Time to elapse for each field
        """
        self._Tss = Tss
        self._T = T
        self._Tfield = Tfield
        self._Ttrans = 0
                 
    def timestep(self):
        self._Nettime += self.dt   
        self._Nettime = np.round(self._Nettime, decimals =3) 
        
    @staticmethod
    def _printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
        formatStr       = "{0:." + str(decimals) + "f}"
        percents        = formatStr.format(100 * (iteration / float(total)))
        filledLength    = int(round(barLength * iteration / float(total)))
        bar             = '█' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()
             
    def print_status(self, time, timegoal):
        """
        Prints a status bar to the terminal in 2% increments.
        From VamPy code base. Same with _printProgress
        """
        it = 2
        if time % (timegoal/(100/it)) < self.dt:
            Network._printProgress(self.progress, 100,
                    prefix = 'Progress:', suffix = 'Complete', barLength = 50)
            self.progress += it    
    
    @property
    def dt(self):
        """
        Time step size
        """
        return self._dt
    
    @property
    def Tss(self):
        """
        Total amount of time we want to simulation to run before reaching steady state (s)
        """
        return self._Tss
    
    @property
    def T(self):
        """
        Length of one period 
        """
        return self._T
    
    @property
    def Tfield(self):
        """
        Time to elapse for a field
        """
        return self._Tfield
    
    @property
    def Nettime(self):
        """
        Time that the network has been running
        """
        return self._Nettime
    
    @property 
    def Ttrans(self):
        """
        Time to transition from AP to PA field
        """
        return self._Ttrans
        
    @property
    def BVcount(self):
        """
        Number of BV in the network
        """
        return self._BVcount
    
    
    @property
    def BVs(self):
        """
        Array containing all BVs
        """
        return self._BVs
 
    @property
    def locations(self):
        """
        Array containing all locations
        """
        return self._locations  
    @property
    def progress(self):
        """
        Simulation progress
        """
        return self._progress
        
    @progress.setter
    def progress(self, value): 
        self._progress = value
        
    @property
    def currentDoseData(self):
         """
         Array of processed 4D Dose data, row = 0 is the dose to the voxel, 
         row = 1 is the voxel index, row = 2 is the time of the event
         """ 
         return self._currentDoseData
          
    @currentDoseData.setter
    def currentDoseData(self,value):
        self._currentDoseData = value
        
 
#%% Excute Simulation
def runSimulation(dummy_input):
    nt = Network(dt, BV_num)
    nt.setTimes(T, Tss, Tfield)
    nt.intializeLocations()
  
    nt.runNT()

    BVolumes = nt.BVs
    #%% Process Data 
    BV_data = np.zeros([len(BVolumes),2])
    i=0
    for BV in BVolumes:
        dose = BV.dose
        location = BV.location
        try:
            BV_data[i,0] = dose[0]
        except:
            BV_data[i,0] = dose
        BV_data[i,1] = location
        i += 1   
    return BV_data, BVolumes    
    
results, BVolumes = runSimulation(0)
 
BV_data = results
dose = BV_data[:,0]
# Calculate mean and standard deviation
mean = np.mean(dose)
std_dev = np.std(dose)

print("mean: " + str(mean))
print("std: " + str(std_dev))

# Create the histogram
plt.hist(dose, bins=25, alpha=0.7, color='skyblue', edgecolor='black', label='Blood Volume Dose')

# Add vertical lines for mean and standard deviation
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2E}')
plt.axvline(mean - std_dev, color='green', linestyle='dotted', linewidth=2, label=f'1 Std Dev: {std_dev:.2E}')
plt.axvline(mean + std_dev, color='green', linestyle='dotted', linewidth=2)

# Add labels and title
plt.xlabel('Dose (Gy)')
plt.ylabel('Blood Volume Count')
plt.title('Single AP Fields 0.1s')
plt.legend()
plt.grid(axis='y', alpha=0.75)

# Display the plot
plt.show()

#np.save("APOnly2msBinning.npy", BV_data)
t.toc()   

#%% Scratch Pad
vessel_dose = []
comp_data = []
for BV in BVolumes:
    location = BV.location
    if location > 27:
        try:
            vessel_dose.append(round(BV.dose[0],3))
        except:
            
            vessel_dose.append(round(BV.dose,3))
    else:
        try:
            comp_data.append(round(BV.dose[0],3))
        except:
            
            comp_data.append(round(BV.dose,3))
        
vessel_mean = np.mean(vessel_dose)
comp_mean = np.mean(comp_data)

plt. hist(vessel_dose, bins=25 )
plt.title('Vessels')
print(np.mean(vessel_dose))
plt.show()
plt.hist(comp_data,  bins=25)
plt.title('Comp')
print(np.mean(comp_data))
plt.show()       

