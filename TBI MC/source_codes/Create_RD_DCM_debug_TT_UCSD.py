# -*- coding: utf-8 -*-
#		Author:  Tony Teke
#  Do not distribute without author's authorization
# 
# python Create_RD_dcm RD_file_name 3ddose_File_name path_to_folder New_RD_DICOM_FILENAME
import dicom
from numpy import *
import sys 

toto=dicom.ReadFile(sys.argv[1])

f = open(sys.argv[3]+ sys.argv[2] ,'r')
line=f.readline()
print line
#while line[0] == ' ':
#  line=line[1:]
line=line.lstrip()
line=line.rstrip()
line=line.split(' ')

#print line 
#print line
coord = [int(P) for P in line]
print coord

# Reading X boundaries
X = zeros( (coord[0]+1) )
Y = zeros( (coord[1]+1) )
Z = zeros( (coord[2]+1) )

XX=f.readline()
XX=XX.lstrip()
XX=XX.rstrip()
XX=XX.split(' ')

for i in range(0,coord[0]+1):
  X[i]=float(XX[i])
#  print i
#  print X[i]
#  print " "
#print "X = " 
#print X

YY=f.readline()
YY=YY.lstrip()
YY=YY.rstrip()
YY=YY.split(' ')
for i in range(0,coord[1]+1):
  Y[i]=float(YY[i])
#print "\n \n Y = " 
#print Y

ZZ=f.readline()
ZZ=ZZ.lstrip()
ZZ=ZZ.rstrip()
ZZ=ZZ.split(' ')
for i in range(0,coord[2]+1):
  Z[i]=float(ZZ[i])
#print "\n \n Z = " 
#print Z


# numpy array can't handle one coord->2D counter, have to specify i and j
counter = 0
sizec=coord[0]   #94
sizel=coord[1]   #79
sizez=coord[2]   #137

#data = int32(zeros( (sizel,sizec,sizez) )) # for debug/testing purposes
data = zeros( (sizez,sizel,sizec) ) # for debug/testing purposes changed l and c

zslice=0
while counter <  (coord[0]*coord[1]*coord[2]):#sum(coord):
  #readline and take care of first space and convert into list of float

  line = f.readline()
  #while line[0] == " ":
  #   line = line [1:]
  #print line
  line=line.lstrip()
  line=line.rstrip()
  line = line.split(" ")
  #print line
  #line=float(line[0:2]) #.split(" "))
  #print line
  line = [float(P) for P in line]
  #print line
  for l in range(len(line)):
     #print "counter %d " %counter
     zslice=(counter)/(sizel*sizec)
     j=( (counter-zslice*sizel*sizec) /(sizec))
     i=( (counter-zslice*sizel*sizec) -(j*sizec))
     #print i,j,k
     data[zslice,j,i]=line[l] # changed i,j
     counter = counter + 1
print data.max()
print "counter %d" %counter
dose_scale= 5.e8/data.max()
print "dose scale %e" %dose_scale
f.close

for i in range(sizel):
    for j in range(sizec):
        for k in range(sizez):
            #data[k,i,j]=int(data[k,i,j]*dose_scale)
            data[k,i,j]=data[k,i,j]*dose_scale
#print data
data = int32(data)
#print data
#update DICOM structure stuff
zpixelsize= (Z[1]-Z[0])*10.
xpixelsize = (X[1]-X[0])*10. #2.5
ypixelsize = (Y[1]-Y[0])*10. #2.5
print "Pixel Size"
print "X=%2.1f" %xpixelsize
print "Y=%2.1f" %ypixelsize
print "Z=%2.1f" %zpixelsize

toto.ImagePositionPatient = [X[0]*10+xpixelsize/2 , Y[0]*10 + ypixelsize/2 , Z[0]*10 + zpixelsize/2 ]
#print toto.ImagePositionPatient
GridFrameOffsetVector = []
print data.max()
for i in range(sizez):
    GridFrameOffsetVector.append(str(i*zpixelsize))
toto.GridFrameOffsetVector = GridFrameOffsetVector
#print  toto.GridFrameOffsetVector
# MODIFY HERE THE PIXELSPACING DICOM TAG 
#*******   TO   DO *********************
# Need to put pixel size in a list
toto.Rows=sizel
toto.Columns=sizec
toto.NumberofFrames=sizez
toto.DoseGridScaling=1./dose_scale
toto.PixelData=data.tostring()
#********************   TO  DO *****************
# take new name of DICOM RD file as an argument
#    DONE
#toto.SaveAs(sys.argv[3] + 'RD_TT.dcm')
toto.SaveAs(sys.argv[3] + sys.argv[4])
