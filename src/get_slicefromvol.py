#!/usr/bin/python

from base import *
import numpy

import os
from os.path import expanduser
HOME = expanduser("~")

vol = Volume()
vol.load('/data2/devtest/tornado/horizons/Seismic2Hrz/rtmsef.vol_regrid_regrid.gz')

size = vol.getSize()
print size


# # --------------------- get inline slices--------------------------
# data = numpy.ndarray(shape=(size[2], size[1]), dtype=numpy.float64)
#     # print "copying" + str(i) + "slice"
# for i in range(600, 611):
#     for k in xrange(size[2]):       
#         for j in xrange(size[1]):
#             # val = vol.getValue(i, j, k)
#             data[k, j] = vol.getValue(i, j, k)

#     # data.tofile('/data2/devtest/tornado/chali/rnd/data/z100to110.')
#     # cv2.imwrite("data.ppm", data)
#     data.tofile(os.path.join(HOME, "rnd/data/i"+str(i)+".dat"))


# # # --------------------- get xline slices--------------------------
data = numpy.ndarray(shape=(size[2], size[0], size[1]), dtype=numpy.float64)
    # print "copying" + str(i) + "slice"
for i in xrange(size[2]):
    for j in xrange(size[0]):       
        for k in xrange(size[1]):
            # val = vol.getValue(i, j, k)
            data[i, j, k] = vol.getValue(j, k, i)

    # data.tofile('/data2/devtest/tornado/chali/rnd/data/z100to110.')
    # cv2.imwrite("data.ppm", data)
    # p = os.path.join(HOME, "rnd/data/x"+str(j)+".dat")
    # print "saving", p, data.shape
    # data.tofile(p)
p = os.path.join(HOME, "rnd/data/volumezxy.dat")
data.tofile(p)

print "done"