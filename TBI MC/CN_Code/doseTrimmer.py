# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 14:39:54 2025

@author: Cassidy.Northway
Converted from work by Levi Burns dose_strip_final.cc
"""

import numpy as np
OK =     0
ERROR =-1
FAIL  =-1
ON    = 1
OFF   = 0
MAX_MED = 100
MAX_STR_LEN = 100

class PHANT_STRUCT:
    def __init__(self):
        self.x_num = 0
        self.y_num = 0
        self.z_num = 0
        self.num_mat = 0
        self.x_bound = [0.0] * 512
        self.y_bound = [0.0] * 512
        self.z_bound = [0.0] * 512
        self.x_size = 0.0
        self.y_size = 0.0
        self.z_size = 0.0
        self.x_start = 0.0
        self.y_start = 0.0
        self.z_start = 0.0
        self.med_name = np.chararray((MAX_MED, MAX_STR_LEN), itemsize=1)
        self.estep = np.zeros(MAX_MED, dtype=float)
        self.mednum = None
        self.densval = None
        
        @property
        def x_bound(self):
            return self._x_bound
        @x_bound.setter 
        def x_bound(self, a): 
            x_bound = a 
        @property
        def y_bound(self):
            return self._y_bound            
        @y_bound.setter 
        def y_bound(self, a): 
            y_bound = a
        @property
        def z_bound(self):
            return self._z_bound    
        @z_bound.setter 
        def z_bound(self, a): 
            z_bound = a    

def readPhantomBoundaries(istrm, n_bounds, bounds):
    tmp_value = istrm.readline().strip().split()
    if not tmp_value:
        print("\n ERROR: readPhantomBoundaries")
        return ERROR
    bounds = np.array(tmp_value)
    return bounds


def read_phant(fname, p):
    # print(f"\n Reading In {fname}")
        with open(fname, "r") as fp:
            # Read number of voxels in each direction X,Y,Z
            values = fp.readline().strip().split()
            if len(values) != 3:
                print("\n ERROR: fscan: reading in x_num,y_num,z_num")
                return FAIL
            
            p.x_num, p.y_num, p.z_num = map(int, values)

            
            print(f"\n xvox={p.x_num} yvox={p.y_num} zvox={p.z_num} \n")

            # Read in the x_bounds
            p.x_bound = readPhantomBoundaries(fp, p.x_num + 1, p.x_bound)

            # Read in the y_bounds
            p.y_bound = readPhantomBoundaries(fp, p.y_num + 1, p.y_bound)

            # Read in the z_bounds
            p.z_bound = readPhantomBoundaries(fp, p.z_num + 1, p.z_bound)



            p.mednum = np.zeros(p.x_num * p.y_num * p.z_num, dtype=np.float64)
            if p.mednum is None:
                print("\n ERROR: Allocating Memory for Int Array")
                print("\n\t x {} y {} z {}".format(p.x_num, p.y_num, p.z_num))
                return FAIL
    
            p.densval = np.zeros(p.x_num * p.y_num * p.z_num, dtype=np.float32)
            if p.densval is None:
                print("\n ERROR: Allocating Memory for Float Array")
                return FAIL
    
            print("\n Reading In Medium Numbers")
            nread = 0
    
            for k in range(p.z_num):
                for j in range(p.y_num):
                    for i in range(p.x_num):
                        value = fp.readline().strip()
                        if value:
                            p.mednum[k * p.x_num * p.y_num + j * p.x_num + i] = float(value)
                            nread += 1
                        else:
                            print("i={} j={} k={}".format(i, j, k))
    
            if nread != p.z_num * p.y_num * p.x_num:
                print("\n ERROR: reading in mednum, read in {}, expected {}".format(nread, p.z_num * p.y_num * p.x_num))
    
            nread = 0
    
            for i in range(p.x_num * p.y_num * p.z_num):
                value = fp.readline().strip()
                if value:
                    p.densval[i] = float(value)
                    nread += 1
    
            if nread != p.z_num * p.y_num * p.x_num:
                print("\n ERROR: reading in densval, read in {}, expected {}".format(nread, p.z_num * p.y_num * p.x_num))
    
            fp.close()
        print("\n For {}".format(fname))
        print("\n Number of voxels {} {} {}".format(p.x_num, p.y_num, p.z_num))
        print("\n Size of voxels   {} {} {}".format(p.x_bound[1] - p.x_bound[0], p.y_bound[1] - p.y_bound[0], p.z_bound[1] - p.z_bound[0]))
        print("\n Start of voxels  {} {} {}".format(p.x_bound[0], p.y_bound[0], p.z_bound[0]))
        return OK  

def write_phant(fname, p):
    try:
        with open(fname, "w") as fp:
            print(f"\n Opened file {fname}")

            extend_pixel = 0
            extend_pixel_front = 0  # LB Sept 18 2017 to deal with front extension

            
            fp.write(f"{p.x_num:5d}{p.y_num - 28:5d}{p.z_num - extend_pixel - extend_pixel_front:5d}\n")

            for i in range(p.x_num + 1):  # LB DO NOT CHANGE
                fp.write(f"{p.x_bound[i]:7.5f} ")

            fp.write("\n")

            for i in range(28, p.y_num + 1):  # LB CHANGE
                fp.write(f"{p.y_bound[i]:7.5f} ")

            fp.write("\n")

            for i in range(extend_pixel, p.z_num + 1 - extend_pixel_front):  # LB CHANGE
                fp.write(f"{p.z_bound[i]:7.5f} ")

            fp.write("\n")

            # LB ADJUST INCREMENTING
            # DOSE_VALS
            for k in range(extend_pixel, p.z_num - extend_pixel_front):  # LB: made each z-slice twice
                for j in range(28, p.y_num):
                    for i in range(p.x_num):
                        fp.write(f"{p.mednum[k * p.x_num * p.y_num + j * p.x_num + i]:9.7E} ")

                    fp.write("\n")
                fp.write("\n")

            # LB ADJUST INCREMENTING

            # UNCERTS
            for k in range(extend_pixel, p.z_num - extend_pixel_front):  # LB: made each z-slice twice
                for j in range(28, p.y_num):
                    for i in range(p.x_num):
                        fp.write(f"{p.densval[k * p.x_num * p.y_num + j * p.x_num + i]:.8f} ")
                    fp.write("\n")
                fp.write("\n")

    except Exception as e:
        print(f"\n ERROR: opening file >{fname}: {e}")
        return "FAIL"

    print(f"\n For {fname}")
    print(f"\n Number of voxels {p.x_num} {p.y_num - 28} {p.z_num - extend_pixel - extend_pixel_front}")  # reschange
    print(f"\n Size of voxels   {p.x_bound[1] - p.x_bound[0]} {p.y_bound[1] - p.y_bound[0]} {p.z_bound[1] - p.z_bound[0]}")  # reschange
    print(f"\n Start of voxels  {p.x_bound[0]} {p.y_bound[0]} {p.z_bound[0]}")
    
    return "OK"

    
# ***********************************
# CHANGE FILENAMES HERE
# ***********************************
egsphantFileName1 = "TBI_XCAT_Full_AP_complete.3ddose"
outputegsphantFileName = "TBI_XCAT_Full_AP_complete_trimmed.3ddose"

# CHANGE PIXEL RESOLUTION HERE IF NEEDED
pixel_size = 0.5  # 2.5 mm fixed by hand but will need to change this
air_density = 0.0012048

# read First EGS4Phant ....
TBIFilter = PHANT_STRUCT()

# *****  Reading both Phantoms ************************
print(f"\n Loading information from {egsphantFileName1}")
if read_phant(egsphantFileName1, TBIFilter) != OK:  # also defined in phantomStructure.h warning &phantom for pointer
    print(f"\n ERROR: Reading EGS4 Phantom File {egsphantFileName1}")
    exit(FAIL)

y_res = TBIFilter.y_bound[2] - TBIFilter.y_bound[1]  # maybe not needed

print(f"\n y_res= {y_res} \n")

print(f"\n Writing information into {outputegsphantFileName}")
if write_phant(outputegsphantFileName, TBIFilter) != OK:  # also defined in phantomStructure.h warning &phantom for pointer
    print(f"\n ERROR: Writing EGS4 Phantom File {outputegsphantFileName}")
    exit(FAIL)

print("\n")    