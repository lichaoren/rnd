#!/usr/bin/python
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np
import scipy.io

import os
from os.path import expanduser
HOME = expanduser("~")

fig = plt.gcf()
fig.set_size_inches(10, 6)

vecshape = ([361, 1616], [361, 1386], [1386, 1616])

data_name = 'x800'
data_path = os.path.join(HOME, "rnd/data/"+data_name+".dat")
data = np.fromfile(data_path, dtype = np.float64)

# initializa a window
cv2.namedWindow("image")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,
#     help = "Path to the image to be thresholded")
ap.add_argument("-t", "--threshold", type = int, default = 128,
    help = "Threshold value")
args = vars(ap.parse_args())

###-------------------------------------------------------------------------###

# convert data to 1d array
data = data.reshape((-1, 1))

Z = np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
K = 128
ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

print len(label), len(center)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(vecshape[1])
cv2.imshow("res2", res2)

cv2.waitKey(0)
cv2.destroyAllWindows()