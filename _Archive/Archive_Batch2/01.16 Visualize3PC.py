# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:44:16 2024

@author: Cassidy.Northway
"""
import pandas as pd
import numpy as np
import scipy
import open3d as o3d
import matplotlib.pyplot as plt

def display_three_pointclouds(array_1,array_2,array_3,windowname):
    p1_pcd = o3d.geometry.PointCloud()
    p1_pcd.points = o3d.utility.Vector3dVector(array_1)
    p1_pcd.paint_uniform_color([0, 0, 0])


    p2_pcd = o3d.geometry.PointCloud()
    p2_pcd.points = o3d.utility.Vector3dVector(array_2)
    p2_pcd.paint_uniform_color([38/255, 191/255, 214/255])

    p3_pcd = o3d.geometry.PointCloud()
    p3_pcd.points = o3d.utility.Vector3dVector(array_3)
    p3_pcd.paint_uniform_color([140/225, 76/255, 164/225]) 
    
    
    concate_pc = np.concatenate((array_1, array_2,array_3),axis = 0)
    p1_color = np.asarray(p1_pcd.colors)
    p2_color = np.asarray(p2_pcd.colors)
    p3_color = np.asarray(p3_pcd.colors)
    p4_color = np.concatenate((p1_color,p2_color,p3_color), axis=0)

    p4_pcd = o3d.geometry.PointCloud()
    p4_pcd.points = o3d.utility.Vector3dVector(concate_pc)
    p4_pcd.colors = o3d.utility.Vector3dVector(p4_color)
    o3d.visualization.draw_geometries([p4_pcd],window_name = windowname)
    
    
main_file1 = 'arteries_rarm12_fitted_data.npy'
main_file2 = 'arteries_rarm3_fitted_data.npy'
main_file3 = 'arteries7_fitted_data.npy'
    
#Load the file data
array_1 = np.load('C:\\Users\\Cassidy.Northway\\RemoteGit\\FittedVesselsFiles\\' + main_file1)
center_array_1 = array_1[:,0:3 ]

array_2 = np.load('C:\\Users\\Cassidy.Northway\\RemoteGit\\FittedVesselsFiles\\' + main_file2)
center_array_2 = array_2[:,0:3 ]

array_3 = np.load('C:\\Users\\Cassidy.Northway\\RemoteGit\\FittedVesselsFiles\\' + main_file3)
center_array_3 = array_3[:,0:3 ]

display_three_pointclouds(center_array_1,center_array_2,center_array_3,'windowname')

