#Installations and Functions


from __future__ import division
import scipy.optimize as optimize
import numpy as np
import os
from runSimulation import runSimulation
import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)
#%% Find Optimal Lambda (Length to Radius Ratio)
#Note this is differnt from my intial testing runs where r_min
#was the tuning variable

def findOptLambda():
    

    #Define number of end vessels !WILL NEED TO EDIT LATER!!!!!CURRENTLY HARD CODED TO ASSUME ALL VESSELS HAVE THE SAME L-RR THIS WILL NEED TO BE UPDATED!!!
    n_vessels = 2
    
    #Run optimization
    results = optimize.least_squares(ModelError, 50, max_nfev = 1000, ftol = 1e-4)
    
    #Parse results
    lambda_vals = results.x
    E = results.fun
    
    
    
    
    
    return [lambda_vals, E]

#%% Function to be Optimized 

def ModelError(lambda_vals):
    
    #Run simulation
    runSimulation(lambda_vals)
    
    #Prep for error calcs !HARD CODED!
    v_number = [1,2] 
    EU = np.zeros ([50, 2]) #!HARD CODED! 2 = num vessles, 50 = number time steps
    EP = np.zeros ([50, 2])
    E = np.zeros([4,]) #4 = num vessels * 2 (for EU and EP)
    #Load simulation data !CURRENTLY HARD CODED, EDIT LATER!
   
    for ii in v_number:
        os.chdir('C:\\Users\\Cassidy.Northway\\RemoteGit\\VamPy_ST')
        
        base_name = str(ii) + '_VamPy_ST.csv'
        
        A_sim = np.loadtxt('a' + base_name, delimiter=',')[:,-1]
        U_sim = np.loadtxt('u' + base_name, delimiter=',')[:,-1]
        P_sim = np.loadtxt('p' + base_name, delimiter=',')[:,-1]
        
        #Q data is the Velocity (cm/s)
        Q_sim = U_sim / A_sim


    #Load ground truth data !CURRENTLY HARD CODED, EDIT LATER!
        os.chdir('C:\\Users\\Cassidy.Northway\\RemoteGit\\VamPy_3wk')

        base_name = str(ii) + '_VamPy_3wk.csv'
        A_data = np.loadtxt('a' + base_name, delimiter=',')[:,-1]
        U_data = np.loadtxt('u' + base_name, delimiter=',')[:,-1]
        P_data = np.loadtxt('p' + base_name, delimiter=',')[:,-1]
        
        #Q data is the Velocity (cm/s)
        Q_data = U_data / A_data
        os.chdir('C:\\Users\\Cassidy.Northway\\RemoteGit')
    #Compute error 
        EU[:,ii-1] =(Q_sim - Q_data) / np.linalg.norm(Q_data, ord =1)
        EP[:,ii-1] =(P_sim - P_data) / np.linalg.norm(P_data, ord =1)
        
    for ii in range(0, len(v_number)):
        E[2*ii] = np.linalg.norm(EU[:,ii], ord =1)
        E[(2*ii)+1] = np.linalg.norm(EP[:,ii], ord =1)

    return E 

#%%
[lambda_vals,E] = findOptLambda()
print(lambda_vals)

