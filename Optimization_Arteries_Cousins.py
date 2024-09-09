#Installations and Functions


from __future__ import division
import scipy.optimize as optimize
import numpy as np
import os
from RunSimulation_Arteries_Cousins_Corn import runSim
import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)
#%% Find Optimal Lambda (Length to Radius Ratio)
#Note this is differnt from my intial testing runs where r_min
#was the tuning variable

def findOptLambda(vessel_dict):
    

    #Define number of end vessels !WILL NEED TO EDIT LATER!!!!!CURRENTLY HARD CODED TO ASSUME ALL VESSELS HAVE THE SAME L-RR THIS WILL NEED TO BE UPDATED!!!
    n_vessels = np.shape(vessel_dict)[0]
    intial_guess = 50*np.ones(n_vessels)
    
    #Run optimization
    results = optimize.least_squares(ModelError, intial_guess , max_nfev = 100 )
    
    #Parse results
    lambda_vals = results.x
    E = results.fun
    print(E)
    print(lambda_vals)
    
    
    
    
    
    return [lambda_vals, E]

#%% Function to be Optimized 

def ModelError(lambda_vals):
    vessel_dict = np.loadtxt('C:\\Users\\Cassidy.Northway\\Remote Git\\lrr_factor_CA_corn.txt')
    #Run simulation
    n_vessels = np.shape(vessel_dict)[0]
    E = np.ones(n_vessels)
    d1,d2 = runSim(lambda_vals)
    if d1 != 1 and d2 != 1:
        index_d1 = 0
        index_d2 = 0
        try: 
            index_d1 = np.where(vessel_dict[:,0]== d1)
        except: 
            index_d2 = np.where(vessel_dict[:,0]== d2)
        try:
            index_d2 = np.where(vessel_dict[:,0]== d2)
        except:
            index_d1 = np.where(vessel_dict[:,0]== d1) 
            
    if index_d1> 0:
        E[index_d1]=10
    if index_d2 > 0:
        E[index_d2]=10
    return E 

#%%
vessel_dict = np.loadtxt('C:\\Users\\Cassidy.Northway\\Remote Git\\lrr_factor_CA_corn.txt')

[lambda_vals,E] = findOptLambda(vessel_dict)
print(lambda_vals)

