#Installations and Functions


from __future__ import division
import scipy.optimize as optimize
import numpy as np
import os
from RunSimulation_April import runSim
import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)
#%% Find Optimal Lambda (Length to Radius Ratio)
#Note this is differnt from my intial testing runs where r_min
#was the tuning variable

def findOptLambda(mirror_dict):
    

    #Define number of end vessels !WILL NEED TO EDIT LATER!!!!!CURRENTLY HARD CODED TO ASSUME ALL VESSELS HAVE THE SAME L-RR THIS WILL NEED TO BE UPDATED!!!
    n_vessels = np.count_nonzero(mirror_dict)
    intial_guess = 6*np.ones(n_vessels)
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
    
    #Run simulation
    print('Current lambda val ' + str(lambda_vals))
    try:
        runSim(lambda_vals)
        #Prep for error calcs !HARD CODED!
        v_number = [0,1,2] 
        EU = np.zeros ([50, 3]) #!HARD CODED! 2 = num vessles, 50 = number time steps
        EP = np.zeros ([50, 3])
        E = np.zeros([6,]) #4 = num vessels * 2 (for EU and EP)
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
    except:
        print('Error occured')
        E=[100,100,100,100,100,100]
    
    
    

    return E 

#%%
mirroring_dict = np.loadtxt('C:\\Users\\Cassidy.Northway\\RemoteGit\\MirroredVessels.txt')

[lambda_vals,E] = findOptLambda(mirroring_dict)
print(lambda_vals)

