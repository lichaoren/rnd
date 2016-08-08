#!/usr/bin/python
import Queue, time
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

clone = data.reshape(vecshape[1])

ratio = 3

xx = 230
yy = 15

val = clone[xx, yy]
i, j = clone.shape
ret = np.zeros(clone.shape, dtype = np.float32)
cv2.circle(ret, (yy, xx), 2, (255, 0, 0))
mask = np.zeros(clone.shape, dtype = bool)
ret[xx, yy] = 100
mask[xx, yy] = 1

# offset = ((-1, 0), (-1, -1), (0, -1), (1, -1), \
#           (1, 0), (1, 1), (0, 1), (-1, 1))
offset = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0))
# offset = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1),\
#           (0, -2), (1, -2), (0, 2), (1, 2))


# cv2.namedWindow("image")
# cv2.imshow("raw", clone)

cv2.namedWindow("image")
# cv2.imshow("ret", ret)
# cv2.waitKey(0)

q = Queue.Queue()
q.put((xx, yy))
while not q.empty():
    coord = q.get()
    for i in xrange(len(offset)):
        cc = tuple(np.add(coord, offset[i]))
        # print coord, " + ", offset[i], " = ", cc
        if (cc[0] >= 0 and cc[0] < vecshape[1][0] \
            and cc[1] >= 0 and cc[1] < vecshape[1][1] \
            # and mask[cc] == 0):
            and mask[cc] == 0 and clone[cc]*val > 0):
            mask[cc] = 1
            q.put(cc)
            ret[cc] = 100
            # print clone[cc]
            # time.sleep(0.001)

cv2.imshow("ret", ret)
cv2.waitKey(0)
# cv2.imwrite("images/tmp2.png", ret)
print "done"
cv2.destroyAllWindows()
