# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:37:53 2025

@author: Cassidy.Northway
"""

import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import easygui
import openpyxl
import os

#%%
#Set time interval (0, 0.955) s
t=[0,0.955] #(s)

#Ask for file name
indexNum = str(easygui.enterbox('Enter index number for the vessel'))

#Ask for Q Distal and Q Prox ml/s
Qdist = float(easygui.enterbox('Enter the distal flow rate')) #ml/s
Qprox = float(easygui.enterbox('Enter the proximal flow rate')) #ml/s

#Determine Q array
Qarray = np.linspace(Qdist, Qprox , 30)

#Ask for R Distal, R Prox and  Vessel Length cm
L = float(easygui.enterbox('Enter the vessel length in cm')) #cm
Rdist = float(easygui.enterbox('Enter the distal radius in mm'))/10 #cm
Rprox = float(easygui.enterbox('Enter the proximal radius in mm'))/10 #cm

#Calc 20 R increments along 0.2 cm uses exp 
X = np.linspace(0.0, L , 30) #cm

#Calc Area increments (cm^2)
Rarray = Rdist * np.power((Rprox/Rdist), X/L) #cm

crop_flag = int(easygui.enterbox('Enter 1 if crop is requried'))
if crop_flag == 1:
    new_L = float(easygui.enterbox('Enter cropped length of vein'))
    X= X[X<=new_L]
    L = new_L
    Rarray = np.append(Rarray[-X.shape[0]:-1],Rprox)
    Qarray = np.linspace(Qdist,Qprox,X.shape[0])
Aarray =np.pi* (Rarray**2) #cm^2

#Convert Q to V m/s 
Varray = Qarray/Aarray #cm/s


#%%#Create Excel sheet
wb = openpyxl.Workbook()
ws = wb.active

#Write time array
ws['A2'].value = 0 #s
ws['A3'].value = 0.955 #s

#Write distance array
i=2
for value in X:
        ws.cell(1,i).value = value/100 #m
        i += 1 
#Write V values
i=2
for value in Varray:
    ws.cell(2,i).value = value/100 #m/s
    ws.cell(3,i).value = value/100 #m/s
    i += 1 
#Save Excel sheet
dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path + '\\BVSimulationFiles\\'+ indexNum + '.xlsx'
wb.save(path)
wb.close()
