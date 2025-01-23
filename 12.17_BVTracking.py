# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:46:42 2024

@author: cbnor
"""
"""
"""


#%% Utility functions
def periodic(t, T):
    """
    Returns equivalent time of the first period if more than one period is simulated.
    
    :param t: Time.
    :param T: Period length.
    """
    while t/T > 1.0:
        t = t - T
    return t
        
def extrapolate(x0, x, y):
    """
    Returns extrapolated data point given two adjacent data points.
    
    :param x0: Data point to be extrapolated to.
    :param x: x-coordinates of known data points.
    :param y: y-coordinates of known data points.
    """
    return y[0] + (y[1]-y[0]) * (x0 - x[0])/(x[1] - x[0])

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
      def set_time(self, dt, T, tc=1):
          """
          Sets timing parameters for the artery network and invokes
          boundary_layer_thickness(T) in each artery.
          
          :param dt: Time step size.
          :param T: Length of one periodic cycle.
          :param tc: Number of cycles.
          """
          self._dt = dt
          self._tf = T*tc
          self._dtr = self.tf/self.ntr
          self._T = T
          self._tc = tc
          for artery in self.arteries:
              artery.boundary_layer_thickness(self.nu, T)
              
              
      def timestep(self):
          """
          Increases time by dt.
          """
          self._t += self.dt       
            
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
                    """
                    it = 2
                    if self.t % (self.tf/(100/it)) < self.dt:
                        ArteryNetwork._printProgress(self.progress, 100,
                                prefix = 'Progress:', suffix = 'Complete', barLength = 50)
                        self.progress += it    
    
        tr = np.linspace(self.tf-self.T, self.tf, self.ntr)
        i = 0
        ii=0
        
        self.print_status()
        self.timestep()       
        bc_in = np.zeros((len(self.arteries), 2))
        
        
        self.timestep()
        ii = ii + 1
        self.print_status()
        tt.toc()
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


    
    
    
            