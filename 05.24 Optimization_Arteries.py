#Installations and Functions


from __future__ import division
import scipy.optimize as optimize
import numpy as np
import os
from RunSimulation_Arteries_May_29 import runSim
import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)
#%% Find Optimal Lambda (Length to Radius Ratio)
#Note this is differnt from my intial testing runs where r_min
#was the tuning variable

def findOptLambda(mirror_dict):
    

    #Define number of end vessels !WILL NEED TO EDIT LATER!!!!!CURRENTLY HARD CODED TO ASSUME ALL VESSELS HAVE THE SAME L-RR THIS WILL NEED TO BE UPDATED!!!
    #n_vessels = np.shape(mirror_dict)[0]
    intial_guess = np.loadtxt('intial_guess.txt')
    #Run optimization
    results = optimize.least_squares(ModelError, intial_guess , max_nfev = 1 )
    
    #Parse results
    lambda_vals = results.x
    E = results.fun
    print(E)
    print(lambda_vals)
    
    
    
    
    
    return [lambda_vals, E]

#%% Function to be Optimized 

def ModelError(lambda_vals):
    path = os.getcwd()
    new_path_ST = os.path.join(path,'VamPy_ST')
    new_path_ADAN = os.path.join(path,'ADAN')
    
    tuning_dict = np.loadtxt('TuningVessels.txt')
    
    #Run simulation
    print('Current lambda val ' + str(lambda_vals))
    try:
        runSim(lambda_vals)
        #Prep for error calcs 
        v_number = np.shape(tuning_dict)[0]
        EU = np.zeros ([50, v_number]) #!HARD CODED! 50 = number time steps
        EP = np.zeros ([50, v_number])
        E = np.zeros([v_number*2,]) #(for EU and EP)
       
        for i in range(0,v_number+1):
            
            vessel_ID = tuning_dict[i,0]
            ADAN_ID = str(tuning_dict[i,1])
            location = tuning_dict[i,2]
            
            #Load simulation data
            os.chdir(new_path_ST)    
            base_name = str(vessel_ID) + '_VamPy_ST.csv'
            
            #Determine where along the length of the vessel we need to sample from.
            if location == 0 or -1:
                ii = int(location)
            else:    
                A_sheet = np.loadtxt('a' + base_name, delimiter=',')
                sheet_length = np.shape(A_sheet)[1]
                ii = int(sheet_length*location)

            A_sim = np.loadtxt('a' + base_name, delimiter=',')[:,ii]
            U_sim = np.loadtxt('u' + base_name, delimiter=',')[:,ii]
            P_sim = np.loadtxt('p' + base_name, delimiter=',')[:,ii]
            
            #Q data is the Velocity (cm/s)
            Q_sim = U_sim / A_sim


        #Load tuning data 
            os.chdir(new_path_ADAN)
            
            ADAN_folder = os.path.join(new_path_ADAN,ADAN_ID)
            os.chdir(ADAN_folder)

            base_name = str(ADAN_ID) + '.csv'
            A_data = np.loadtxt('a' + ADAN_ID, delimiter=',')[:]
            U_data = np.loadtxt('u' + ADAN_ID, delimiter=',')[:]
            P_data = np.loadtxt('p' + ADAN_ID, delimiter=',')[:]
            
            #Q data is the Velocity (cm/s)
            Q_data = U_data / A_data
            os.chdir(path)
        #Compute error 
            EU[:,i-1] =(Q_sim - Q_data) / np.linalg.norm(Q_data, ord =1)
            EP[:,i-1] =(P_sim - P_data) / np.linalg.norm(P_data, ord =1)
            
        for i in range(0, v_number):
            E[2*i] = np.linalg.norm(EU[:,i], ord =1)
            E[(2*i)+1] = np.linalg.norm(EP[:,i], ord =1)
    except:
        print('Error occured')
        E = 100*np.ones([v_number*2,])
    
    
    

    return E 

#%%
mirroring_dict = np.loadtxt('MirroredVessels.txt')

[lambda_vals,E] = findOptLambda(mirroring_dict)
print(lambda_vals)

