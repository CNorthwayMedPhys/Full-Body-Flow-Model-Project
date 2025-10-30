# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:46:42 2024

@author: cbnor
"""
"""
Notes to self

15/01/2025 
I think I'll want a location class which will include all the values in FlowTracker.xlsx,
the flow data and the splitting data. 

I'll need to load in the splitting ratio data some where to pass to
the intializer fcn

21/01/25
I want to determine the where all the BVs all so let's write a fcn for
the network to bin them into the categories from legget
"""

#%%Import

import pandas as pd
import os
import numpy as np
import sys
import random



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
    
    
#%% Location class

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
  

#%% Start by defining BV class

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
        self._exptime = 0
        self._tottime = 0
        self._dwelltime = 0
        self._distance = 0
        #EDITS
        self._traveltime = 0
        self._travelflag = 0
    
    
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
    def exptime(self):
        """
        Total time that the BV has been "exposed" to dose (s)
        """
        return self._exptime
    
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
        
    #EDITS
    @property 
    def traveltime(self):
        """
        Time from L.Heart to L.Heart
        """
        return self._traveltime
    
    @property
    def travelflag(self):
        """
        Flag used to determine the stage of travel
        """
        return self._travelflag
    
    @traveltime.setter
    def traveltime(self, value):
        self._traveltime = value
        
    @travelflag.setter
    def travelflag(self,value):
        self._travelflag = value
#%% Define Network Class

class Network (object):
    """
    Class representing the entire network of blood volumes and locations
    """
    def __init__ (self, dt, dx, BV_num):
        self._Nettime = 0
        self._progress = 0
        self._dt = dt
        self._dx = dx
        self._BVcount = 0
        self._BVs = []
        self._locations = []
        
    def intializeLocations (self,DTmodifier):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + "\\FlowTracker.xlsx"
        df = pd.read_excel(path)
        for index, row in df.iterrows():
            data = row['LocationIDNumber;OutFlowA;OutFlowB;OutFlowC;DwellTime(s);SplittingRatioKey']
            data = data.split(';')
            if int(data[0]) == 12 or int(data[0]) == 15:
                DTmod = DTmodifier[0]
            elif int(data[0]) == 13 or int(data[0]) == 16:
                DTmod = DTmodifier[1]
            elif int(data[0]) == 14 or int(data[0]) == 17:
                DTmod = DTmodifier[2]
            elif int(data[0]) == 18 or int(data[0]) == 22:
                DTmod = DTmodifier[3]
            elif int(data[0]) == 19 or int(data[0]) == 23:
                DTmod = DTmodifier[4]
            elif int(data[0]) == 20 or int(data[0]) == 24:
                DTmod = DTmodifier[5]
            elif int(data[0]) == 21 or int(data[0]) == 25:
                DTmod = DTmodifier[6]    
            elif int(data[0]) == 26 or int(data[0]) == 27:
                DTmod = DTmodifier[7]  
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
            if location.splittingratiokey != '0':
                location.IntSplittingRatio (SRdf)
        print('\n Location intialization complete')     
        
    def intializeBV(self):
        self.BVs.append(BloodVolume(self.BVcount, 28))
        self._BVcount += 1
             
    def runNT (self):
        flag = 0
        #Run until we have completed the desired number of cycles
        while self.Nettime < self.tf:
            
            #Check to see if we have the desired number of BVs. 
            #If not release another BV into the system 
            if self.BVcount < BV_num:
   
                self.intializeBV()
            if self.BVcount == BV_num and flag == 0:
                print('\n BV Intialized')
                flag = 1
            
            #Calculate the t w/in the period for table look ups
            pt = periodic(self.Nettime, self.T) 
            
            #Iterate through each BV
            for BV in self.BVs:
                
                #EDIT: have I returned to the L.heart?
                if BV.location == 1 and BV.travelflag == 0:
                    BV.travelflag = 1
                    
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
                       BV._exptime += self.dt
                
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
                if BV.travelflag == 0:
                    BV._traveltime += self.dt
            self.timestep()
            self.print_status()
           
    def setTime(self, T, tc):
        """
        Sets timing parameters for the network 
        :param dt: Time step size.
        :param T: Length of one periodic cycle.
        :param tc: Number of cycles.
        """
        self._tf = T*tc
        self._T = T
        self._tc = tc
                 
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
             
    def print_status(self):
        """
        Prints a status bar to the terminal in 2% increments.
        From VamPy code base. Same with _printProgress
        """
        it = 2
        if self.Nettime % (self.tf/(100/it)) < self.dt:
            Network._printProgress(self.progress, 100,
                    prefix = 'Progress:', suffix = 'Complete', barLength = 50)
            self.progress += it    
    
    def binBVs(self):
        dictBV = {"Brain": 0, 
                "Stomach": 0 ,
                "S. Intestine": 0,
                "L. Intestine": 0,
                "Heart": 0 ,
                "Kidneys": 0,
                "Liver": 0,
                "Pulmonary": 0,
                "Pancreas": 0,
                "Spleen": 0,
                "Aorta and L. Arteries": 0,
                "L. Veins": 0,
                "Distributed Tissues": 0}
        for BV in self.BVs:
            ID = int(BV.location)
            if ID >= 28 and ID <= 52: #arteries
                dictBV["Aorta and L. Arteries"] += 1
            elif ID >= 104 and ID <= 125: #arteries
                dictBV["Aorta and L. Arteries"] += 1
            elif ID >= 53 and ID <= 97: #veins
                dictBV["L. Veins"] += 1
            elif ID == 0 or ID == 1: #heart
                dictBV["Heart"] += 1
            elif ID == 2: #brain
                dictBV["Brain"] += 1
            elif ID == 3: #s. int
                dictBV["S. Intestine"] += 1
            elif ID == 4: #pancreas
                dictBV["Pancreas"] += 1
            elif ID == 5: #stomach
                dictBV["Stomach"] += 1
            elif ID == 7: #stomach
                dictBV["Spleen"] += 1
            elif ID == 6 or ID == 9: #kidney
                dictBV["Kidneys"] += 1
            elif ID == 8: #stomach
                dictBV["L. Intestine"] += 1            
            elif ID == 10 or ID == 11: #liver
                dictBV["Liver"] += 1
            elif ID >= 12 and ID <= 25: #Dist tiss
                dictBV["Distributed Tissues"] += 1
            else: #pulm
                dictBV["Pulmonary"] += 1
        for item in dictBV:
            dictBV[item] = (dictBV[item]/self.BVcount) * 100
        return dictBV
    @property
    def dt(self):
        """
        Time step size
        """
        return self._dt
    
    @property
    def tf(self):
        """
        Total amount of time we want to simulation to run (s)
        """
        return self._tf
    
    @property
    def T(self):
        """
        Length of one period 
        """
        return self._T
    
    @property
    def tc(self):
        """
        Number of periods
        """
        return self._tc
    
    @property
    def Nettime(self):
        """
        Time that the network has been running
        """
        return self._Nettime
    
    @property
    def BVcount(self):
        """
        Number of BV in the network
        """
        return self._BVcount
    
    @property
    def dx(self):
        """ 
        Spatial step size (m)
        """
        return self._dx
    
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
 
    
#%%Parameters 
dx = 1e-4 # Distance step size (m)
dt = 0.002 #Time step size (s)
T = 0.955 #Length of one period (s)
BV_num = 1e2 #total number BV
ToR = BV_num/((T/dt)) # nubmer of cycles before all BVs released
tc = np.round(200 + ToR) #Number of cycles to be simulated

DTmodifier = np.asarray([1.16222463, 1.21396462, 1.07799579, 0.76160822, 0.65386433, 1.0146046,\
 1.00537915, 0.58857567, 0.8])

nt = Network(dt, dx, BV_num)
nt.setTime(T, tc)
nt.intializeLocations(DTmodifier)
nt.runNT()

print('Complete!')

#%%Compute travel time avg
TravelTime = []

for BV in nt.BVs:
    TravelTime = np.append(TravelTime,BV.traveltime)

avg = np.mean(TravelTime)
std = np.std(TravelTime)
 

