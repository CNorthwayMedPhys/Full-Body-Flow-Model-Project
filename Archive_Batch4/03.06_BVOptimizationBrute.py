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

    
    rrange = (slice(0,2,0.5),slice(0,2,0.5),slice(0,2,0.5),
              slice(0,2,0.5), slice(0,2,0.5),slice(0,2,0.5),
              slice(0,2,0.5),slice(0,2,0.5), 
              slice(0,2,0.5),slice(0,2,0.5),slice(0,2,0.5),
              slice(0,2,0.5),slice(0,2,0.5),slice(0,2,0.5),
              slice(0,2,0.5),slice(0,2,0.5))
    #Run optimization
    results = optimize.brute(ModelError, rrange, Ns = 3,
        full_output = True)
    
    #Parse results
    DT_modifiers = results[0]
    E = results[1]

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
    E = np.sqrt(np.sum(np.abs(gt-sim)**2))
    print(E)
    print(DT_modifiers)
    return E 

#%%Prepare MP and run

 
results = findOptDT()
print(results[0])
print(results[1])
#plt.bar(range(len(dictSim)), list(dictSim.values()), align='center')
#plt.xticks(range(len(dictSim)), list(dictSim.keys()))
#plt.show()



