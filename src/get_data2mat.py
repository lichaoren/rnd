#!/usr/bin/python

import cv2
import numpy as np
import scipy.io

import os
from os.path import expanduser
HOME = expanduser("~")
vecshape = ([361, 1616], [361, 1386], [1386, 1616])

data_name = 'x800'
data_path = os.path.join(HOME, "rnd/data/"+data_name+".dat")
data = np.fromfile(data_path, dtype = np.float64)
data = data.reshape(vecshape[1])
data = data + np.amin(data) * -1.0

# # save the data as mat format
scipy.io.savemat(data_name+'.mat', mdict={'data' : data})
