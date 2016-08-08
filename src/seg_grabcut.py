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
# NOT WORKING
# Not widely used, 

img = img1u8

mask = np.zeros(img.shape, dtype = np.uint8)
bgdModel = np.zeros((4, 4),np.float64)
fgdModel = np.zeros((230, 15),np.float64)

rect = (300,120,470,350)

# this modifies mask 
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

# If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

# adding additional dimension for rgb to the mask, by default it gets 1
# multiply it with input image to get the segmented image
img_cut = img*mask2[:,:,np.newaxis]

plt.subplot(211),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(img_cut)
plt.title('Grab cut'), plt.xticks([]), plt.yticks([])
plt.show()


img3u8 = cv2.applyColorMap(img1u8, cv2.COLORMAP_HSV)
cv2.imshow("test", img3u8)

if cv2.waitKey(0) & 0xff == 27:
    pass

cv2.destroyAllWindows()