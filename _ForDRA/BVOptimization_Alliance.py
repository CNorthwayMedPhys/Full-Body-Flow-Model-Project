#Installations and Functions


import os
import scipy.optimize as optimize
import numpy as np
import multiprocessing as mp
from BVTrackingforOpt_MP import BVsim
from functools import partial

"""
Notes to self

02/21 I want to modify this code to allow for pooling across the BV population.
I want to intialize all the basics and then disperse the BV across CPUS
After the BVs have been sent out I want their current locations to be pulled in
and I then want them combined and the BV % done
"""

#%% Find Optimal Dwell Times for Distributed Tissues 

def findOptDT():

    intial_guess = np.asarray(np.ones(7)*100)
    
    #Run optimization
    results = optimize.least_squares(ModelError, intial_guess ,bounds=([0, 1500]), max_nfev = 100, diff_step = 1, xtol = None, verbose = 2 )
    
    #Parse results
    DT_modifiers = results.x
    E = results.fun
    
    return [DT_modifiers, E]

#%% Function to be Optimized 
def mp_handler(DT_modifiers):
    BVnum = 1e5
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    p = mp.Pool(processes=ncpus)
    iterations = int(BVnum/100)
    BVsimPart = partial (BVsim, DTmodifier = DT_modifiers)
    BVlocations = p.map(BVsimPart , range(iterations))
    return BVlocations

def ModelError(DT_modifiers):
    
    #Run BV simulation in mp for all BVs
    BVlocations = mp_handler(DT_modifiers)
    
    dictSim = binBVs(BVlocations)
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
    E = np.nanmax((np.abs(gt-sim)*100)/gt)

    return E 

def binBVs(BVlocations):
    dictBV = {"Brain": 0, 
            "Stomach": 0 ,
            "S. Intestine": 0,
            "L. Intestine": 0,
            "Heart": 0 ,
            "Kidneys": 0,
            "Liver": 0,
            "Pulmonary": 0,
            "Pancreas": 0,
            "Spleen": 0,
            "Aorta and L. Arteries": 0,
            "L. Veins": 0,
            "Distributed Tissues": 0}
    for chunks in BVlocations:
        for ID in chunks:
            if ID >= 28 and ID <= 54: #arteries
                dictBV["Aorta and L. Arteries"] += 1
            elif ID >= 106 and ID <= 127: #arteries
                dictBV["Aorta and L. Arteries"] += 1
            elif ID >= 55 and ID <= 99: #veins
                dictBV["L. Veins"] += 1
            elif ID == 0 or ID == 1: #heart
                dictBV["Heart"] += 1
            elif ID == 2: #brain
                dictBV["Brain"] += 1
            elif ID == 3: #s. int
                dictBV["S. Intestine"] += 1
            elif ID == 4: #pancreas
                dictBV["Pancreas"] += 1
            elif ID == 5: #stomach
                dictBV["Stomach"] += 1
            elif ID == 7: #stomach
                dictBV["Spleen"] += 1
            elif ID == 6 or ID == 9: #kidney
                dictBV["Kidneys"] += 1
            elif ID == 8: #stomach
                dictBV["L. Intestine"] += 1            
            elif ID == 10 or ID == 11: #liver
                dictBV["Liver"] += 1
            elif ID >= 12 and ID <= 25: #Dist tiss
                dictBV["Distributed Tissues"] += 1
            else: #pulm
                dictBV["Pulmonary"] += 1
    for item in dictBV:
        dictBV[item] = (dictBV[item]/1e5) * 100
    return dictBV

#%%Prepare MP and run

if __name__ == '__main__': 
    [DT_modifiers,E] = findOptDT()

#dictSim = BVsim(DT_modifiers)

#plt.bar(range(len(dictSim)), list(dictSim.values()), align='center')
#plt.xticks(range(len(dictSim)), list(dictSim.keys()))
#plt.show()

#print(DT_modifiers)

