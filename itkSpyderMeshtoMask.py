# -*- coding: utf-8 -*-
"""
Created on Tue May  6 10:48:33 2025

@author: cbnor
based on: https://github.com/InsightSoftwareConsortium/ITK/issues/2884
"""

import itk
import os


#%%
#Import mesh file
path = os.getcwd() + '\\VesselFiles\\Utilized Subset\\dias_aorta.stl'

#No idea what this does frankly
TCoordinate = itk.F
Dimension = 3
TMesh = itk.Mesh[TCoordinate, Dimension].New()

MeshType = itk.Mesh[itk.F, 3]
reader = itk.MeshFileReader[MeshType].New()
meshIO = itk.STLMeshIO.New()
reader.SetMeshIO(meshIO)
reader.SetFileName(path)
reader.Update()
mesh = reader.GetOutput()

mesh_writer = itk.MeshFileWriter[TMesh].New()
mesh_writer.SetFileName(os.getcwd() + "\\TempVTK\\Cleaned_Aorta.vtk")
mesh_writer.SetInput(mesh)
mesh_writer.Update()

TMesh = itk.Mesh[itk.SS, Dimension].New()
mesh_reader = itk.MeshFileReader[TMesh].New()
mesh_reader.SetFileName(os.getcwd() + "\\TempVTK\\Cleaned_Aorta.vtk")
mesh_reader.Update()
mesh = mesh_reader.GetOutput()

#%%
TPixel = itk.SS                                                                     
TImage = itk.Image[TPixel, Dimension]                                               
                                                                                    
image = itk.Image[TPixel, Dimension].New()                                          
region = itk.ImageRegion[Dimension]()                                               
region.SetSize([256, 256, 350])                                                     
region.SetIndex([0, 0, 0])                                                          
image.SetRegions(region)                                                            
image.Allocate()                                                                    
image.SetOrigin([-15, -30, -5])                                              
image.SetSpacing([1, 1, 1])   

#%%                                                  
mesh_to_image_filter = itk.TriangleMeshToBinaryImageFilter[TMesh, TImage].New() 
mesh_to_image_filter.SetInput(mesh)                                      
mesh_to_image_filter.SetInfoImage(image)                                        
mesh_to_image_filter.Update()
filtered = mesh_to_image_filter.GetOutput()
itk.imwrite(mesh_to_image_filter.GetOutput(), os.getcwd() + "\\VesselFiles\\Utilized Subset\\gzFiles\\cleaned.nii.gz") 

#%%
