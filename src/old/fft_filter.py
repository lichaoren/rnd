#!/usr/bin/python

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

## ------------ HPF in numpy ---------------
# f = np.fft.fft2(data)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))

# plt.subplot(331),plt.imshow(data, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(332),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# # plt.show()


# rows, cols = data.shape
# crow,ccol = rows/2 , cols/2
# fshift[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
# f_ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)

# plt.subplot(334),plt.imshow(data, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(335),plt.imshow(img_back, cmap = 'gray')
# plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
# plt.subplot(336),plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])


# rows, cols = data.shape
# nrows = cv2.getOptimalDFTSize(rows)
# ncols = cv2.getOptimalDFTSize(cols)
# crow, ccol = rows/2 , cols/2

# right = ncols - cols
# bottom = nrows - rows
# bordertype = cv2.BORDER_CONSTANT #just to avoid line breakup in PDF file
# nimg = cv2.copyMakeBorder(data,0,bottom,0,right,bordertype, value = 0)

# dft = cv2.dft(np.float32(nimg),flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)

# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# plt.subplot(221),plt.imshow(data, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()


# # create a mask first, center square is 1, remaining all zeros
# mask = np.zeros((nrows,ncols,2),np.uint8)
# mask_size = 45
# mask[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 1

# # apply mask and inverse DFT
# fshift = dft_shift*mask
# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv2.idft(f_ishift)
# img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

# # plt.subplot(223),plt.imshow(mimi, cmap = 'gray')
# # plt.title('fshift'), plt.xticks([]), plt.yticks([])
# plt.subplot(224),plt.imshow(img_back, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

# plt.show()
# plt.tight_layout()
# plt.savefig("./images/tmp.png", dpi=200)