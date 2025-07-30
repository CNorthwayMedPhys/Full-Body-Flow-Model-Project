# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:04:01 2025

@author: Cassidy.Northway
"""



import os

class ContourSet:
    def __init__(self):
        self.x1 = []
        self.x2 = []
        self.z1 = []
        self.z2 = []

    def add(self, block_num, raw_vals):
        if block_num == 0:
            for i in range(0, len(raw_vals), 2):
                self.x1.append(raw_vals[i] / 10.0)
                self.z1.append(raw_vals[i + 1] / 10.0)
        else:
            for i in range(0, len(raw_vals), 2):
                self.x2.append(raw_vals[i] / 10.0)
                self.z2.append(raw_vals[i + 1] / 10.0)

    def write(self, file_name):
        writer = None
        try:
            writer = open(file_name, 'w')
            for line in range(1, 5):
                print(self.get_array_size(line), end=' ')
                for index in range(self.get_array_size(line)):
                    writer.write(self.get_string(line, index))
                writer.write("** \n")
        except FileNotFoundError as e:
            print(e)
        except IOError as e:
            print(e)
        finally:
            if writer is not None:
                writer.close()

    def get_array_size(self, order_num):
        if order_num == 1:
            return len(self.x1)
        elif order_num == 2:
            return len(self.z1)
        elif order_num == 3:
            return len(self.x2)
        else:
            return len(self.z2)

    def get_string(self, array, index):
        if array == 1:
            tmp = str(self.x1[index])
        elif array == 2:
            tmp = str(self.z1[index])
        elif array == 3:
            tmp = str(self.x2[index])
        else:
            tmp = str(self.z2[index])
        
        return tmp + " "
    
 #%%

import os
import re
from pydicom import dcmread
from pydicom.sequence import Sequence
from pydicom.errors import InvalidDicomError

# class ContourSet:
#     def __init__(self):
#         self.contours = {}

#     def add(self, index, points):
#         self.contours[index] = points

#     def write(self, output_file_name):
#         with open(output_file_name, 'w') as f:
#             for index, points in self.contours.items():
#                 f.write(f"Contour {index}: {points}\n")

def main(args):
    print("**************************************")
    print("   Getting Contours from DICOM file")
    print("**************************************")
    dicom_file_name = "RP.CN_XCAT.TBI_Crop_AP.dcm"
    output_file_name = "LOG.contourPoints"
    print(f"  Number of Inputs {len(args)}")
    
    if len(args) >= 2:
        dicom_file_name = args[0]
        output_file_name = args[1]
    else:
        print("  ERROR - Not enough input parameters, using defaults")

    whole_list = get_whole_list(dicom_file_name)
    num_blocks = extract_num_blocks(whole_list)
    print(f"  Number of Blocks found = {num_blocks}")
    
    if num_blocks < 2:
        print("  ERROR - Not enough blocks found, exiting...")
    else:
        beam_list = whole_list.BeamSequence[0]
        block_seq = beam_list.BlockSequence
        contour = ContourSet()
        num_lung_blocks_found = 0

        for block_counter in range(num_blocks):
            tmp_list = block_seq[block_counter]
            print(tmp_list)
            curr_name = tmp_list.BlockName if 'BlockName' in tmp_list else "NOT FOUND"
            if is_lung_block(curr_name) and num_lung_blocks_found < 2:
                points = tmp_list.BlockData
                print(f"  Block {curr_name}  has  {len(points)}  points")
                contour.add(num_lung_blocks_found, points)
                num_lung_blocks_found += 1

        if num_lung_blocks_found == 2:
            contour.write(output_file_name)
        else:
            print("  WARNING - Not enough lung blocks found")

        print("\n  DONE getting Contour")

def get_whole_list(input_file_name):
    try:
        dicom = dcmread(input_file_name)
        return dicom
    except InvalidDicomError as e:
        print(e)
        return None

def extract_num_blocks(dicom):
    try:
        return dicom.BeamSequence[0].NumberOfBlocks
    except (IndexError, AttributeError):
        return 0

def is_lung_block(name):
    return re.search(r'lung', name, re.IGNORECASE) is not None

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])   