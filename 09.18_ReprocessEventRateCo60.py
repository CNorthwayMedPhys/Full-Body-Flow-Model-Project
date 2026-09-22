# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:02:54 2026

@author: Cassidy.Northway
"""
import os
import numpy as np
from pathlib import Path

cd = os.getcwd()

# Define the directory to search
directory_path = directory_path = Path(cd+'//CompartmentData\\PA\\')

# Find all files matching the pattern recursively (e.g., all .txt files)
# .rglob() returns a generator yielding Path objects
matching_files = directory_path.rglob("*500ms.npy")

# Convert to list of strings or iterate directly
file_list = [str(file) for file in matching_files if file.is_file()]

for file in file_list:
    event_trace = np.load(file)
    event_trace_og = event_trace[1,:]
    event_trace[1,:] = event_trace_og/0.5
    np.save(file,event_trace)
