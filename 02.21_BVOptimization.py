#Installations and Functions


import matplotlib.pyplot as plt
import scipy.optimize as optimize
import numpy as np
from BVTrackingforOpt_MP import BVsim
"""
Notes to self

02/21 I want to modify this code to allow for pooling across the BV population.
I want to intialize all the basics and then disperse the BV across CPUS
After the BVs have been sent out I want their current locations to be pulled in
and I then want them combined and the BV % done
"""

#%% Find Optimal Dwell Times for Distributed Tissues 

def findOptDT():

    intial_guess = np.ones(7)*100
    #Run optimization
    results = optimize.least_squares(ModelError, intial_guess ,bounds=([0, 1500]), max_nfev = 100, diff_step = 1, xtol = None, verbose = 2 )
    
    #Parse results
    DT_modifiers = results.x
    E = results.fun
    return [DT_modifiers, E]

#%% Function to be Optimized 


def ModelError(DT_modifiers):
    
    #Run BV simulation
    dictSim = BVsim(DT_modifiers)
    
    #Ground truth percentile value dictionary
    dictGT = {"Brain": 1.24, 
            "Stomach": 1.03 ,
            "S. Intestine": 3.93,
            "L. Intestine": 2.27,
            "Heart": 9.3 ,
            "Kidneys": 2.07,
            "Liver": 10.34,
            "Pulmonary": 10.85,
            "Pancreas": 0.62,
            "Spleen": 1.45,
            "Aorta and L. Arteries": 6.2,
            "L. Veins": 18.61,
            "Distributed Tissues": 32.09}
    sim = []
    gt = []
    for item in dictGT:
        gt.append(dictGT[item])
        sim.append(dictSim[item])
    sim = np.array(sim)
    gt = np.array(gt)
        

    #Compute percintile error 
    E = (np.abs(sim - gt) / gt) * 100

    return E 

#%%
[DT_modifiers,E] = findOptDT()

#Run final results and plot
dictSim = BVsim(DT_modifiers)

plt.bar(range(len(dictSim)), list(dictSim.values()), align='center')
plt.xticks(range(len(dictSim)), list(dictSim.keys()))
plt.show()

print(DT_modifiers)

