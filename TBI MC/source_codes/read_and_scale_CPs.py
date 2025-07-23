# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:53:14 2017

@author: levi.burns

This code uses the contourPoint file.
It starts with the .contourPoints file that is produced by runTBI.
It finishes with a text file (.txt). 
Matlab will convert this text file to an .egsphant

Need to input the t0, t1, t2 z-values from the Excel sheet (the MC_scaled values)
Also adjust the file name
(Doesn't actually use t1...)
"""


import matplotlib.pyplot as plt

#CHANGE THE FILE NAME HERE!
points = open("LOG.contourPoints","r")
data = points.readlines()

#CHANGE t0, t1, t2
t0z = 0.00
t1z = 8.35
t2z = 13.81

scal = (139.7/100.0)/0.0357746  #make the numbers this much bigger
#last float value is to convert to pixels instead of cm

shiftz = (t2z-t0z)*scal


lxstr = data[0]
lzstr = data[1]
rxstr = data[2]
rzstr = data[3]


#convert them to lists (also take off ** at end from the .contourPoints format)
#Note - the "l" and "r" here refer to AP... for PA, the Matlab code will swap them
rx = rxstr.split(" ")
rx.remove('**')
rx_flt = []
for i in range(len(rx)-1):
    rx_flt.append(float(rx[i]))

rz = rzstr.split(" ")
rz.remove('**')
rz_flt = []
for i in range(len(rz)-1):
    rz_flt.append(float(rz[i]))

lx = lxstr.split(" ")
lx.remove('**')
lx_flt = []
for i in range(len(lx)-1):
    lx_flt.append(float(lx[i]))

lz = lzstr.split(" ")
lz.remove('**')
lz_flt = []
for i in range(len(lz)-1):
    lz_flt.append(float(lz[i]))



#Now actually scale the values
rx_scal = []
for i in rx_flt:
    rx_scal.append(i*scal)
    
rz_scal = []
for i in rz_flt:
    rz_scal.append(i*scal)
    
lx_scal = []
for i in lx_flt:
    lx_scal.append(i*scal)
    
lz_scal = []
for i in lz_flt:
    lz_scal.append(i*scal)

#plt.plot(rx_flt,rz_flt,'ro')
#plt.plot(lx_flt,lz_flt,'ro')
#plt.plot(0.0,0.0,'s')
#plt.xlim([-10,10])
#plt.ylim([-10,10])
lz_scal_shift = []
rz_scal_shift = []

for i in lz_scal:
    lz_scal_shift.append(i+shiftz)
    
for i in rz_scal:
    rz_scal_shift.append(i+shiftz)

#Produce a plot of the values

plt.plot(rx_scal,rz_scal_shift,'ro')
plt.plot(lx_scal,lz_scal_shift,'ro')
plt.plot(0.0,0.0,'s')
plt.xlim([-400,400])
plt.ylim([0,1050])
plt.title("Scaled, shifted")



#Need to write these to a file now. Floats.
#Best way to handle the import in matlab:

outfile = open("contourstomask_res.txt","w")

with outfile as f:
    for i in rx_scal:
        f.write(str(i)+" ")
    f.write("\n")
    for i in rz_scal_shift:
        f.write(str(i)+" ")
    f.write("\n")
    for i in lx_scal:
        f.write(str(i)+" ")
    f.write("\n")
    for i in lz_scal_shift:
        f.write(str(i)+" ")
    f.write("\n")  

outfile.close()
points.close()
