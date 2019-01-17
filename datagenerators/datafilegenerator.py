# -*- coding: utf-8 -*-
"""
Created on: Mon Jan 14 16:06:46 2019
@author: Anidhya Bhatnagar
@description: Program to generate data file with random numbers
"""

import numpy as np

filename = str(input("Enter the file name: "))
row = int(input("Enter the number of rows: "))
col = int(input("Enter the number of columns: "))

datafile = open(filename, "w")

data = np.random.rand(row, col)
data = data * 100
finaldata = ""
firstline = 1
for i in data:
    line = ""
    for j in range(col):
        if j == 0 and firstline == 1:
            line = line + str(i[j])
            firstline = 0
        elif j == 0:
            line = '\n' + line + str(i[j])
        else:
            line = line + ',' + str(i[j])
    finaldata = finaldata + line

datafile.write(finaldata)
datafile.close()
