# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 11:04:16 2025
Load in RS file with empty structures and add in the stl files data
Save for import to Eclipse
Adding organ contours to XCAT phantom
@author: Cassidy.Northway
"""

import os
import numpy as np
from stl import mesh

stl_dir = r'\\PHSAhome2.phsabc.ehcnet.ca\Cassidy.Northway\Remote Git\OrganContours\OrganStlFiles'
FileList = [f for f in os.listdir(stl_dir) if f.endswith('.stl')]



organDic = {}

for name in FileList:
    TR = mesh.Mesh.from_file(os.path.join(stl_dir, name))
    name_no_ext = name.replace('.stl', '')
    # You can add TR to organDic or process as needed
    # organDic[name_no_ext] = TR