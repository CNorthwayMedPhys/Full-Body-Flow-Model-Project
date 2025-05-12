# -*- coding: utf-8 -*-
"""
Created on Tue May  6 10:48:33 2025

@author: cbnor
based on: https://github.com/InsightSoftwareConsortium/ITK/issues/2884
"""

import itk
import os
import tkinter
from tkinter import filedialog
import numpy as np
#%%
#Import mesh file
bb_flag = 1

tkinter.Tk().withdraw() # prevents an empty tkinter window from appearing
file_names = filedialog.askopenfilenames()
summed_array = []
flag = 1
for file_name in file_names:
    name = file_name.split('/')[-1]
    name = name.split('.')[0]
    
    #No idea what this does frankly
    TCoordinate = itk.F
    Dimension = 3
    TMesh = itk.Mesh[TCoordinate, Dimension].New()
    
    MeshType = itk.Mesh[itk.F, 3]
    reader = itk.MeshFileReader[MeshType].New()
    meshIO = itk.STLMeshIO.New()
    reader.SetMeshIO(meshIO)
    reader.SetFileName(file_name)
    reader.Update()
    mesh = reader.GetOutput()
    
    mesh_writer = itk.MeshFileWriter[TMesh].New()
    mesh_writer.SetFileName(os.getcwd() + "\\TempVTK\\"+ name + ".vtk")
    mesh_writer.SetInput(mesh)
    mesh_writer.Update()
    
    TMesh = itk.Mesh[itk.SS, Dimension].New()
    mesh_reader = itk.MeshFileReader[TMesh].New()
    mesh_reader.SetFileName(os.getcwd() + "\\TempVTK\\" + name + ".vtk")
    mesh_reader.Update()
    mesh = mesh_reader.GetOutput()
    
    
    # # Determine BB of entire vessel set to determine the image size 
    # bounding_box = mesh.GetBoundingBox()
    # min_point = bounding_box.GetMinimum()
    # max_point = bounding_box.GetMaximum()
    # [x_min,y_min,z_min] = np.array(min_point)
    # [x_max,y_max,z_max] = np.array(max_point)
    
    # if bb_flag == 1:
    #     x_0 = x_min
    #     y_0 = y_min
    #     z_0 = z_min
    #     x_1 = x_max
    #     y_1 = y_max
    #     z_1 = z_max
    #     bb_flag = 0
    # if x_min < x_0:
    #     x_0 = x_min
    # if y_min < y_0:
    #     y_0 = y_min
    # if z_min < z_0:
    #     z_0 = z_min
    # if x_max > x_1:
    #     x_1 = x_max
    # if y_max > y_1:
    #     y_1 = y_max
    # if z_max > z_1:
    #     z_1 = z_max

    
    #%%
    TPixel = itk.SS                                                                     
    TImage = itk.Image[TPixel, Dimension]                                               
                                                                                        
    image = itk.Image[TPixel, Dimension].New()                                          
    region = itk.ImageRegion[Dimension]()                                               
    region.SetSize([650, 235, 1630])                                                     
    region.SetIndex([0, 0, 0])                                                          
    image.SetRegions(region)                                                            
    image.Allocate()                                                                    
    image.SetOrigin([-310, -145, -1070])                                              
    image.SetSpacing([1, 1, 1])   
    
    #%%                                                  
    mesh_to_image_filter = itk.TriangleMeshToBinaryImageFilter[TMesh, TImage].New() 
    mesh_to_image_filter.SetInput(mesh)                                      
    mesh_to_image_filter.SetInfoImage(image)                                        
    mesh_to_image_filter.Update()
    filtered = mesh_to_image_filter.GetOutput()
    array = itk.GetArrayFromImage(filtered)
    if np.max(array) != 1:
        print(name)
    else:
        itk.imwrite(mesh_to_image_filter.GetOutput(), os.getcwd() + "\\VesselFiles\\Utilized Subset\\gzFiles\\" + name +".nii.gz") 
        if flag == 1:
            summed_array = array
            flag = 0
        else:
            summed_array = summed_array + array

itk.imwrite(itk.GetImageFromArray(summed_array), os.getcwd() + "\\VesselFiles\\Utilized Subset\\gzFiles\\summed.nii.gz")             
#%%
