# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:07:10 2023

@author: Cassidy.Northway
"""
def runSimulation(lambda_val):
    #%% Load fcns and packages
    from scipy.interpolate import interp1d
    import numpy as np
    import pandas as pd
    from artery_network_modified_numba import ArteryNetwork
    
    
    #%%Define inlet function
    def inlet(qc, rc, f_inlet):
        """
        Function describing the inlet boundary condition. Returns a function.
        """
        Q = np.loadtxt(f_inlet, delimiter=',')
        t = [(elem) * qc / rc**3 for elem in Q[:,0]]
        q = [elem / qc for elem in Q[:,1]] #added that 10
        return interp1d(t, q, kind='linear', bounds_error=False, fill_value=q[0])
    
    #%%Load in Vessel Dataframe
    try:
        vessel_df = pd.read_pickle ('C:\\Users\\cbnor\\Documents\\Full Body Flow Model Project\\larm.pkl')
    except:
        vessel_df = pd.read_pickle ('C:\\Users\\Cassidy.Northway\\RemoteGit\\larm.pkl')
    
    #!MODIFICATIONS FOR SIMPLE BIFURCATION!
    vessel_df = vessel_df.loc[0:2]
    vessel_df.at[1,'End Condition'] = 'LW'
    vessel_df.at[2,'End Condition'] = 'LW'
    vessel_df.at[0,'Radius Values'] = [0.37, 0.37]
    vessel_df.at[1,'Radius Values'] = [0.177, 0.17]
    vessel_df.at[2,'Radius Values'] = [0.177, 0.17]
    vessel_df.at[0, 'lam'] =56.22
    vessel_df.at[1, 'lam'] =100
    vessel_df.at[2, 'lam'] =99.4
    
    #%% Define parameters and passing variables
    
    #Dimension parameters
    rc = 1  
    qc = 10 
    
    #Physical Parameters
    rho = 1.055 #blood density
    nu = 0.046 #blood viscosity
    p0 = (85 * 1333.22365) * rc**4/(rho*qc**2) # zero transmural pressure
    Re = qc/(nu*rc) #Reynold's number
    
    #Simulation parameters
    T = 0.917 #length of period (s)
    T = T * qc / rc**3 # time of one cycle
    tc = 4 #number of periods
    dt = 1e-5 #time step size
    dx = 0.1 #physical step size
    ntr = 50 # number of time steps to be stored
    
    #Elasticity Parameters
    kc = rho*qc**2/rc**4
    k1 = 2.0e7 
    k2 = -22.53 
    k3 = 8.65e5
    k = (k1/kc, k2*rc, k3/kc) # elasticity model parameters (Eh/r)
    
    #Un-dimesnionalize step
    T = T * qc / rc**3 # time of one cycle
    dt = dt * qc / rc**3 # time step size
    dx = dx / rc # spatial step size
    
    
    #3wk parameters
    R1 = 25300
    R2 = 13900
    Ct = 1.3384e-6
    out_args = [R1*rc**4/(qc*rho), R2*rc**4/(qc*rho), Ct*rho*qc**2/rc**7] # Windkessel parameters
    
    #Structured Tree parameters
    alpha = 0.88 #branching ratio
    beta =0.66 #branching ratio
    l_rr = lambda_val #length to radius ratio 
    r_min = 0.025 #minimum radius 
    Z_term = 0 #terminal impedance value 
    out_arg = [] #place holder
    
    #Create parameter variables
    
    phys_parameter = [rho, nu, p0, Re, k]
    
    st_parameter = [Z_term, alpha, beta, r_min, l_rr]
    
    #Decide Outlet BCs
    out_bc = 'ST'
    
    #Dataframe is the panada date frame containing the vessel geometeries and connections
    dataframe = vessel_df
    
    #%% Prepare Inlet BC
    q_in = inlet(qc, rc, 'example_inlet.csv')
    
    #%% Create system 
    an = ArteryNetwork(phys_parameter, dataframe, st_parameter, ntr)
    an.mesh(dx)
    an.set_time(dt, T, tc)
    an.initial_conditions(0, dataframe)
    
    #%% Solve and redimensionalize
    # run solver
    an.solve(q_in, out_bc, out_arg)
    
    
    # redimensionalise
    an.redimensionalise(rc, qc)
    
    #%% Name file and dump results
    #Define file name
    if out_bc == '3wk':
        file_name = 'VamPy_3wk'
    elif out_bc == 'ST':
        file_name = 'Vampy_ST'
    try:
        an.dump_results(file_name,'C:\\Users\\cbnor\\Documents\\Full Body Flow Model Project')   
    except:
         an.dump_results(file_name,'C:\\Users\\Cassidy.Northway\\GitRemoteRepo')

import time


runSimulation(50)
