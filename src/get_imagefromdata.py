#!/usr/bin/python

import cv2
import numpy as np

import os
from os.path import expanduser
HOME = expanduser("~")

data_path = os.path.join(HOME, "rnd/data/z100.dat")
data = np.fromfile(data_path, dtype = np.float64)
data = data.reshape((1386, 1616))
data = data + np.amin(data)

cv2.imwrite("data.ppm", data)