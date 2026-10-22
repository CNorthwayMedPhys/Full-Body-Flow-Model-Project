# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 12:50:49 2025

@author: Cassidy.Northway
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import random

#%%
#Dose file locations 
cd = os.getcwd()
dose_filename_AP = "4DDoseData\\AP\\Sorted\\XCAT_AP"
dose_filename_PA = "4DDoseData\\PA\\Sorted\\XCAT_PA"

#Get Voxel Mapping Array
mapping_filename_AP = "VOIMappingArrays\\Hi-Res\\AP\\"
mapping_filename_PA = "VOIMappingArrays\\Hi-Res\\PA\\"  

#Compartment data locations
compartment_AP = "CompartmentData\\AP\\"
compartment_PA = "CompartmentData\\PA\\"


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
        else:
            VOIMappingArrayPath = os.path.join(cd,mapping_filename_PA,str(number)+'.npy')
        VOIMappingArray = np.load(VOIMappingArrayPath)
        VOIMappingArray[:,0] = VOIMappingArray[:,0] / 1000 #mm -> m
        self._VOIMap = VOIMappingArray
        
    def IntDVH (self,number, xx):
        if str(number) == 11:
            number = 10
        elif str(number) == 13 or str(number) == 14:
            number = 12
        elif str(number) == 16 or str(number) == 17:
            number = 15    
        if xx == 'AP':
            DVHArrayPath = os.path.join(cd,compartment_AP,str(number)+'DVH.npy')
        else:
            DVHArrayPath = os.path.join(cd,compartment_PA,str(number)+'DVH.npy')
        DVHArray = np.load(DVHArrayPath)  
        self._DVH = DVHArray
    def IntEventFreq (self,number, xx):
            if str(number) == 11:
                number = 10
            elif str(number) == 13 or str(number) == 14:
                number = 12
            elif str(number) == 16 or str(number) == 17:
                number = 15    
            if xx == 'AP':
                EventFreqArrayPath = os.path.join(cd,compartment_AP,str(number)+'eventTrace.npy')
            else:
               EventFreqArrayPath = os.path.join(cd,compartment_PA,str(number)+'eventTrace.npy')
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
