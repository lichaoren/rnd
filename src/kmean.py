#!/usr/bin/python

import matplotlib.pyplot as plt

import cv2
import numpy as np

import os
from os.path import expanduser
HOME = expanduser("~")

fig = plt.gcf()
fig.set_size_inches(10, 6)

vecshape = ([361, 1616], [361, 1386], [1386, 1616])

data_name = 'x800'
data_path = os.path.join(HOME, "rnd/data/"+data_name+".dat")
data = np.fromfile(data_path, dtype = np.float64)
data = data.reshape((-1, 1))
# data = cv2.normalize(data, data)
# data = data.reshape((361, 1386))
# cv2.imshow('raw data', data)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

Z = np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
K = 64
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

print len(label), len(center)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(vecshape[1])
plt.subplot(2, 1, 1), plt.imshow(res2)

# Z = np.float32(data)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
# K = 64
# ret, label, center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)

# # Now convert back into uint8, and make original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape(vecshape[1])
# plt.subplot(2, 1, 2), plt.imshow(res2, cmap = 'gray')

plt.show(); plt.savefig("images/tmp.png", dpi = 200)
