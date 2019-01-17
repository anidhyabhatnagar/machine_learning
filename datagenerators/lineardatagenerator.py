# -*- coding: utf-8 -*-
"""
Created on: Tue Jan 15 00:41:14 2019
@author: Anidhya Bhatnagar
@description: Linear Data Generator
"""

import numpy as np

filename = str(input("Enter the file name: "))
valuerange = int(input("Enter the range of data points: "))
datapoints = int(input("Enter the number of data points: "))

datafile = open(filename, "w")

x = np.arange(valuerange)
delta = np.random.uniform(0, 10, size=(datapoints,))
y = 0.4 * x + 3 + delta

finaldata = ""
firstline = 1
for i in range(valuerange):
    line = ""
    if i == 0:
        line = line + str(x[i]) + ',' + str(y[i])
    else:
        line = '\n' + line + str(x[i]) + ',' + str(y[i])
    finaldata = finaldata + line

datafile.write(finaldata)
datafile.close()
