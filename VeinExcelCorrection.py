# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:29:53 2025

@author: Cassidy.Northway
"""

import tkinter as tk
import tkinter.filedialog as fd
import openpyxl

tk.Tk().withdraw()
files = fd.askopenfilenames()

for file in files:
    wb = openpyxl.load_workbook(file,data_only=True)
    ws = wb.worksheets[0]
    
    for row in ws.iter_rows(min_row = 2, min_col = 2):
        for cell in row:
            v = cell.value
            new_v = v*2
            cell.value = new_v
    wb.save(file)
    wb.close()