# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:46:42 2024

@author: cbnor
"""

#%% Start by defining BV class

class BloodVolume (object):
    """
    Class represents a single Blood Volume to be tracked
    
    """
    
    def __init__ (self,ID, volume, locationID):
        """
        Blood volume constructor
        """
        self._ID =ID
        self._volume = volume
        self._locationID = locationID
        self._locationpos = locationpos
        self._exptime = 0
        self._tottime = 0
        self._dwelltime = 0
        #self._dose = dose
        
    @property
    def ID(self):
        """
        ID number of the BV
        """
        return self._ID
    
#%%
    
class Network (object):
    """
    """
    def __init__ (self, dt, volume, BV_num):
        self._Nettime = 0
        self._dt = dt
        self._BVcount = 0
        self._volume = volume
        self._BVs = []
        
    def intializeBV(self, ID):
        self.BVs.append(BloodVolume(ID, self.volume, 0, 0, 0, 0))
        self._BVcount += 1
        #currently locationID = 0 ==heart need to map
    
    def timestep(self):
        self._Nettime += self.dt
        
    def intializeNT (self):
        ID_num = 0
        
        while nt. BVcount < BV_num:
            nt.intializeBV(ID_num)
            ID_num += 1

            for BV in self.BVs:
                #need to access current location and determine the path fro
                #there I'll need to build out a spread sheet to handle this 
                
                #Step 1. determine current location that will decide whether 
                #we are dwelling or moving
                
                #If dwell than check if time<dwell time if not than put into next
                #else increase dwell time by time step
                
                #If in vessel sample the flow rate
            self.timestep()
            
            
        
        
    @property
    def dt(self):
        """
        Time step size
        """
        return self._dt
    
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
    def volume(self):
        """ 
        Volume of each blood volume
        """
        return self._volume
    
    @property
    def BVs(self):
        """
        Array containing all BVs
        """
        return self._BVs
#%%
volume = 1 #mL
sys_time = 0 #sec
dt = 1 #sec
BV_num = 10 #total number BV

nt = Network(dt, volume, BV_num)
nt.intializeNT()


    
    
    
            