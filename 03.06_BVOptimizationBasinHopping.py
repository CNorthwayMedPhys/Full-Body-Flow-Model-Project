#Installations and Functions


import matplotlib.pyplot as plt
import scipy.optimize as optimize
import numpy as np
from BVTrackingAll import BVsim


"""
Notes to self

"""

#%% Find Optimal Dwell Times for Distributed Tissues 

def findOptDT():

    intial_guess = np.asarray([9.4, 10.1, 10.1, 9.4, 9.7, 10.1, 8.3, 10.6, 3.4, 1.2, 0.5, 4.2, 1.1, 4.5, 3.3, 4.5,3.8])
    
    #Run optimization
    results = optimize.basinhopping(ModelError, intial_guess, 
        niter= 20, stepsize = 1, interval = 20 )
    
    #Parse results
    DT_modifiers = results.x
    E = results.fun

    return [DT_modifiers, E]


def ModelError(DT_modifiers):
    
    #Run BV simulation in mp for all BVs
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
    E = np.nanmax(np.abs(gt-sim))
    print(E)
    print(DT_modifiers)
    return E 

#%%Prepare MP and run

 
[DT_modifiers,E] = findOptDT()
print(DT_modifiers)
print(E)
#plt.bar(range(len(dictSim)), list(dictSim.values()), align='center')
#plt.xticks(range(len(dictSim)), list(dictSim.keys()))
#plt.show()



