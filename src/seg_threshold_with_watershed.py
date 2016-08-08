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
# precision loss
gray1u8 = np.uint8(data.reshape(vecshape[1]))

# initialize the list of threshold methods
methods = [
    # ("THRESH_BINARY", cv2.THRESH_BINARY),
    # ("THRESH_BINARY + THRESH_OTSU", cv2.THRESH_BINARY + cv2.THRESH_OTSU),
    ("THRESH_BINARY_INV", cv2.THRESH_BINARY),
    # ("THRESH_TRUNC", cv2.THRESH_TRUNC),
    # ("THRESH_TOZERO", cv2.THRESH_TOZERO),
    # ("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV),
    # ("THRESH_OTSU", cv2.THRESH_OTSU)
    ]
# thresh, dst = cv2.threshold(img,0,255,
#     cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# loop over the threshold methods
for (threshName, threshMethod) in methods:
    # threshold the image and show it
    (T, thresh) = cv2.threshold(gray1u8, args["threshold"], 255, threshMethod)
    cv2.imshow(threshName, thresh)
    cv2.waitKey(0)

# dst = cv2.adaptiveThreshold(gray1u8,255,
#     cv2.ADAPTIVE_THRESH_MEAN_C,
#     cv2.THRESH_BINARY_INV,
#     9, 0)
# cv2.imshow("ADAPTIVE_THRESH_MEAN_C", dst)
# cv2.waitKey(0)

# dst = cv2.adaptiveThreshold(gray1u8,255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#     cv2.THRESH_BINARY_INV,
#     9, 0)
# cv2.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", dst)
# cv2.waitKey(0)

# noise removal
kernel = np.ones((3, 3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
cv2.imshow("sure_bg", sure_bg)
cv2.waitKey(0)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
cv2.imshow("dist_transform", dist_transform)
cv2.waitKey(0)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
cv2.imshow("sure_fg", sure_fg)
cv2.waitKey(0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow("unknown", unknown)
cv2.waitKey(0)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
imgC = cv2.applyColorMap(gray1u8, cv2.COLORMAP_JET)
cv2.imshow("imgC", imgC)
cv2.waitKey(0)

markers = cv2.watershed(imgC, markers)
imgC[markers == -1] = [255, 0, 0]
cv2.imshow("imgC", imgC)
cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()

# best result is regular binary