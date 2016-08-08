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
img1f64 = data.reshape(vecshape[1])
img1f32 = np.float32(img1f64)

clone = np.copy(img1f64)
img1u8 = np.uint8(cv2.normalize(clone, clone, 0, 255, cv2.NORM_MINMAX))

# initializa a window
# cv2.startWindowThread()
cv2.namedWindow("image")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = False,
    help = "Path to the image to be thresholded")
ap.add_argument("-t", "--threshold", type = int, default = 128,
    help = "Threshold value")
args = vars(ap.parse_args())

###-------------------------------------------------------------------------###
# attemp to use low pass methods to filter out some high frequency events before
# pass data down

kernelbank = {
    "5x5meanf32"    : np.ones((5,5),np.float32)/25,
    "scharr" : [[-3, 0, 3],
                [-10,0,10],
                [-3, 0, 3]],
    "sobelx" : [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
}

filters = [
#high pass filters
# "scharr", "sobel", 
"sobely",
"laplacian",
# low pass filters
# "mean", "blur", "gaussian", 
"bilateral", 
]
params = {
    "mean"      : (img1f64, -1, np.ones((5,5), np.float32)/25),
    "blur"      : (img1f64, (5,5)),
    "gaussian"  : (img1f64, (5, 5), 0),
    "bilateral" : (img1f64,9,75,75)
}
funcdict = {
    "mean"      : cv2.filter2D(img1f64, -1, np.ones((5,5), np.float32)/25),
    "blur"      : cv2.blur(img1f64, (5,5)),
    "gaussian"  : cv2.GaussianBlur(img1f64, (5, 5), 0),
    "bilateral" : cv2.bilateralFilter(img1f32,9,35,35, cv2.BORDER_REFLECT),
    "scharry"    : cv2.Scharr(img1f64, cv2.CV_64F,0,1,0),
    "sobely"     : cv2.Sobel(img1f64, cv2.CV_64F,0,2,0),
    "laplacian" : cv2.Laplacian(img1f64, cv2.CV_64F)
}

# for i in xrange(len(filters)):
#     key = filters[i]
#     cv2.imshow(key, funcdict[key])
#     cv2.waitKey(0)

tmp = np.float32(img1f64)
tmp = cv2.bilateralFilter(tmp, 9,35,35, cv2.BORDER_REFLECT)
cv2.imshow("bil", tmp)
cv2.waitKey(0)

# tmp = np.float32(tmp)
tmp = cv2.Sobel(tmp, cv2.CV_32F,0,2,0)
cv2.imshow("sob", tmp)
cv2.waitKey(0)

tmp = cv2.bilateralFilter(tmp, 9,35,35, cv2.BORDER_REFLECT)
cv2.imshow("bil", tmp)
cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()