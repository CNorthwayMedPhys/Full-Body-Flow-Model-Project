# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:55:58 2024

@author: cnorthway
"""

#%%Import section
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.filedialog as fd
import pydicom._storage_sopclass_uids
import os
#%% load the DICOM files
tk.Tk().withdraw()
file_names = fd.askopenfilenames(title='Choose a file')

files = []
for fname in file_names:
    print(f"loading: {fname}")
    files.append(pydicom.dcmread(fname))

print(f"file count: {len(files)}")


print("Combining Files...")
pixels = []
i = 0
for f in files:
    #fFirst file
    if i == 0:
        pixels = f.pixel_array
        pixels = pixels.astype('int32')
        i = i + 1
    else:    
        pixel_set = f.pixel_array
        pixel_set = pixel_set.astype('int32')
        pixels = np.append(pixels, pixel_set, axis = 0)


# pixel aspects, assuming all slices are the same
ps = [0.8,0.8] #mm
ss = 0.8 #mm
ax_aspect = ps[1] / ps[0]
sag_aspect = ps[1] / ss
cor_aspect = ss / ps[0]

#%% Plot
img_shape = np.shape(pixels)
img3d = pixels

# plot 3 orthogonal slices
a1 = plt.subplot(2, 2, 1)
plt.imshow(img3d[:, :, img_shape[2] // 4])
a1.set_aspect(ax_aspect)

a2 = plt.subplot(2, 2, 2)
plt.imshow(img3d[:, img_shape[1] // 2, :])
a2.set_aspect(sag_aspect)

a3 = plt.subplot(2, 2, 3)
plt.imshow(img3d[img_shape[0] // 2, :, :].T)
a3.set_aspect(cor_aspect)

plt.show()

#%%Write file

VS = [0.8,0.8,0.8] #mm
V = np.uint16(pixels)
fname = 'Combined'

print('Writing DICOM Files')
fileMeta = pydicom.Dataset()
fileMeta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
fileMeta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
fileMeta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

# dataset
ds = pydicom.Dataset()
ds.file_meta = fileMeta

ds.Rows = V.shape[1]
ds.Columns = V.shape[2]
ds.NumberOfFrames = V.shape[0]
ds.ImagerPixelSpacing = VS[0:2]
ds.PixelSpacing = VS[0:2] # in mm
ds.SliceThickness = VS[2] # in mm
ds.BitsAllocated = 16
ds.PixelRepresentation = 1
ds.Width = V.shape[1]
ds.Height = V.shape[2]
ds.ColorType = 'grayscale'
ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
ds.PhotometricInterpretation = 'MONOCHROME2'

os.chdir('C:\\Users\cbnor\OneDrive\Desktop\FullBodyPhantom Files\CombinedFiles')
# create dicom slices
fnames = []
for i in range(V.shape[0]):
    I = V[i, :, :]
    I = np.flipud(I)
    cfn = f'img_{i + 1}.dcm'
    # Create DICOM file
    ds.PixelData = I.tobytes()
    ds.save_as(cfn, write_like_original=False)
    fnames.append(cfn)

print('Done!')


