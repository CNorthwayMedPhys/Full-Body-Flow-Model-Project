# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:19:01 2024

Goal of this file is to be able to calc z_n array for our BC in a seperate  file
and save them as a dictionary
 
@author: Cassidy.Northway
"""

#%% Load packages
 
import numpy as np
import pandas as pd
import math
import warnings
import os
import pickle
import multiprocessing as mp
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

#%%

def impedance_weights(r_root, dt, T, tc, rc, qc, nu):
    acc = 1e-12 #numerical accuracy of impedance fcn
    r_root = r_root*rc
    N = math.ceil(1/dt)
    eta = acc**(1/(2*N))
    
    m = np.linspace(0,2*np.pi,(2*N)+1) #actual [0:2N-1] the size of 2N
    zeta = eta * np.exp(1j*m)
    Xi = 0.5*(zeta**2) - (2*zeta) + (3/2)
    
    Z_impedance = np.zeros(np.size(Xi), dtype = np.complex_)
    for ii in range(0,np.size(Xi)):
        Z_impedance[ii] = getImpedance(Xi[ii]/(dt), r_root,rc, qc ,nu)

    z_n = np.zeros(np.size(Xi), dtype = np.complex_) 
    weighting = np.concatenate(([1],2*np.ones((2*N)-1),[1]))/ (4*N)
    for n in range(0,N+1): # actual range [0,N]
        z_n[n] = np.sum(weighting * Z_impedance * np.exp(-1j*n*m))
        z_n[n] = z_n[n] / (eta ** n)
    z_n = np.real(z_n)

    return z_n

def getImpedance(s, r_root,rc, qc ,nu):
    #maxGens = math.ceil(math.log(self.r_min / r_root) / math.log(self.alpha)) + 1
    empty_table = np.empty((1000, 1000)) #replace max gen with 1000
    empty_table[:] = np.nan
    [Z, table] = impedance(s, r_root,0, 0, empty_table, rc, qc , nu)
    return Z
    
    
def impedance(s, r_root, N_alpha, N_beta, table, rc, qc , nu):
    
    if r_root > 0.025:
        xi = 2.5
        zeta = 0.4
        lrr = 10
    elif r_root <= 0.005:
        xi = 2.9
        zeta = 0.9  
        lrr = 30
    else:
        xi = 2.76
        zeta = 0.6
        lrr = 20
    alpha = (1+zeta**(xi/2))**(-1/xi)
    beta = alpha * np.sqrt(zeta)
    
    r_0 = r_root * (alpha ** (N_alpha)) *(beta ** (N_beta))
    
    if r_0 < r_min:
        ZL = 0
    else:
        if np.isnan(table[N_alpha + 1, N_beta]):
            [ZD1, table] = impedance( s, r_root, N_alpha+1 , N_beta,table, rc,qc,nu)
        else:
            ZD1 = table[N_alpha + 1, N_beta]
 
        if np.isnan(table[N_alpha, N_beta +1]):
            [ZD2, table] = impedance( s,r_root,N_alpha , N_beta + 1,table, rc,qc,nu)
        else:
            ZD2 = table[N_alpha , N_beta + 1]
   
        ZL = (ZD1 * ZD2) / (ZD1 + ZD2)
    
    Z0 = singleVesselImpedance(ZL, s ,r_0 , rc, qc , nu, lrr )
    table [N_alpha, N_beta] = Z0
    return [Z0, table]
                 
def singleVesselImpedance(ZL, s, r_0, rc,qc, nu, lrr):
    
    gamma = 2 #velocity profile 
    nu_temp = nu * qc / rc 
    L = r_0 * lrr
    A0 = np.pi * (r_0 ** 2)
    Ehr = (2e7 *np.exp( -22.53*r_0) + 8.65e5) #Youngs Modulus * vessel thickness/radius
    C = (3/2) *(A0)/(Ehr) #complaince
    delta_s = (2 * nu_temp*(gamma +2))/ (rho * (r_0**2))
   
    

    if s == 0:
        Z0 = ZL + (2*(gamma +2)*nu_temp* lrr) / (np.pi * r_0**3)
        
            
    else:
        d_s = (A0/(C*rho*s*(s+delta_s)))**(0.5)
        num = ZL + ((np.tanh(L/d_s, dtype=np.longcomplex))/(s*d_s*C))

        denom = s*d_s*C*ZL*np.tanh(L/d_s, dtype=np.longcomplex) + 1   
        Z0 = num/denom
                       
    return Z0

def determine_tree(pos,Rd, T, tc,rc,qc,nu):
    """
    Intiate the tree calculcation for the artery
    
    """
    if dataframe.at[pos,'End Condition'] == 'ST':
        zn = impedance_weights(Rd, dt, T, tc,rc,qc,nu)
    else:
        zn = 0
    return zn

def main_calc (xy):
    zn = determine_tree(xy[0],xy[1],T, tc,rc,qc,nu)
    
    return {xy[0]: zn}
#%% Load in dataframe
path = os.getcwd()
file_name = path + '\\SysArteries.pkl' #switch to \ for slurm
vessel_df = pd.read_pickle (file_name)

    
#%% Define parameters
rc = 1  #cm
qc = 10 #cm3/s
rho = 1.055 #g/cm3
nu = 0.049 #cm2/s

T = 1 #s
tc = 1 #Normally 4 #s
dt = 0.25e-5 #normally 1e-5 #s
dx = 0.015 #normally 0.1 cm  (unitless)

Re = qc/(nu*rc) 
T = T * qc / rc**3 # time of one cycle
tc = tc # number of cycles to simulate
dt = dt * qc / rc**3 # time step size
ntr = 50 # number of time steps to be stored
dx = dx / rc # spatial step size
nu = nu*rc/qc # viscosity

kc = rho*qc**2/rc**4
k1 = 2.0e7 #g/s2 cm
k2 = -22.53 # 1/cm 
k3 = 8.65e5 #g/s2 cm

k = (k1/kc, k2*rc, k3/kc) # elasticity model parameters (Eh/r) 
out_args =[0]
out_bc = 'ST'
p0 =((85 * 1333.22365) * rc**4/(rho*qc**2)) # zero transmural pressure intial 85 *
  
dataframe = vessel_df
r_min =0.003 #2014_Cousins
Z_term = 0 #Terminal Impedance 8    
    
#%% Intialize calculation and dictionary

indices = np.arange(0,len(dataframe))
Rd_array = dataframe.loc[:,'Radius Values']
Rd_array=np.array(Rd_array)
lst = []
for i in indices:
    [Ru,Rd] = Rd_array[i] #From mm to cm
    lst.append((i,Rd/10)) 

pool = mp.Pool (processes = 2) #temp for current work desktop
BC_dict = pool.map(main_calc, lst)
    
# create a binary pickle file 
f = open("StBC.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(BC_dict,f)

# close file
f.close()

       
 