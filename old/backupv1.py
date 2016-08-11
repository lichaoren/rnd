#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np
import scipy.io

import argparse
import Queue
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

# initializa a graphics
cv2.namedWindow("image")
imgraw = np.copy(img1f64)
cv2.normalize(img1f64, imgraw)
cv2.imshow("Raw", imgraw * 255)
font = cv2.FONT_HERSHEY_SIMPLEX


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = False,
    help = "Path to the image to be thresholded")
ap.add_argument("-t", "--threshold", type = int, default = 128,
    help = "Threshold value")
args = vars(ap.parse_args())

###-------------------------------------------------------------------------###
# ## CONSTANTS

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

# construct the Scharr x-axis kernel
ScharrX = np.array((
    [-3, 0, 3],
    [-10,0,10],
    [-3, 0, 3]), dtype="int")

# construct a diagnal kernel
Diagnal_1 = np.array((
    [1, 2, 3, 4, 0],
       [2, 3, 4, 0, -4],
          [3, 4, 0, -4, -3],
             [4, 0, -4, -3, -2],
                [0, -4, -3, -2, -1]), dtype="int")

Diagnal_2 = np.array((
    [1, 2, 0],
       [2, 0, -2],
          [0, -2, -1]), dtype="int")

# construct the kernel bank, a list of kernels we're going
# to apply using both our custom `convole` function and
# OpenCV's `filter2D` function
kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY),
    ("diagnal", Diagnal_1)
)


# ## filter bank
# filterBank = {
#     # gaussian filter using a 5x5 kernel
#     "gaussian"      : cv2.GaussianBlur(img_to_filter, (5, 5), 0),
#     # global threshold filter
#     "global"        : cv2.threshold(img_to_filter, 127, 255, 
#                         cv2.THRESH_BINARY),
#     # adaptive method: ADAPTIVE_THRESH_MEAN_C, block size, constant
#     "adaptive"      : cv2.adaptiveThreshold(img_to_filter,127,
#                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21, 0),
#     # otsu filter provieds more localized result than constant threshold
#     "otsu"          : cv2.threshold(img_to_filter, 0, 255, 
#                         cv2.THRESH_BINARY + cv2.THRESH_OTSU),
#     # performs a mean blur
#     "blur"          : cv2.blur(img_to_filter, (5,5)),
#     # performs dilate then erode, essentially fills small holes
#     "bilateral"     : cv2.bilateralFilter(img_to_filter,9,35,35, cv2.BORDER_REFLECT),
#     # identical to global filter, but with customized kernel size
#     "mean"          : cv2.filter2D(img_to_filter, -1, np.ones((5,5), np.float32)/25),
#     # ##---------------HIGH pass filter_names---------------------##
#     "scharry"       : cv2.Scharr(img_to_filter, cv2.CV_64F,0,1,0),
#     "sobely"        : cv2.Sobel(img_to_filter, cv2.CV_64F,0,2,0),
#     "laplacian"     : cv2.Laplacian(img_to_filter, cv2.CV_64F)
# }

###-------------------------------------------------------------------------###

###-------------------------------------------------------------------------###
## customized convolution for diagnal edge detection
def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = (kW - 1) / 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum()

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k

    # rescale the output image to be in the range [0, 255]
    # output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    # return the output image
    return output


###-------------------------------------------------------------------------###
# ## FFT wrapper 
# def fft(inp, axis):
#     f = np.fft.fft2(data)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20*np.log(np.abs(fshift))

#     plt.subplot(331),plt.imshow(data, cmap = 'gray')
#     plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(332),plt.imshow(magnitude_spectrum, cmap = 'gray')
#     plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#     # plt.show()


#     rows, cols = data.shape
#     crow,ccol = rows/2 , cols/2
#     fshift[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
#     f_ishift = np.fft.ifftshift(fshift)
#     img_back = np.fft.ifft2(f_ishift)
#     img_back = np.abs(img_back)

#     plt.subplot(334),plt.imshow(data, cmap = 'gray')
#     plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(335),plt.imshow(img_back, cmap = 'gray')
#     plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
#     plt.subplot(336),plt.imshow(img_back)
#     plt.title('Result in JET'), plt.xticks([]), plt.yticks([])


#     rows, cols = data.shape
#     nrows = cv2.getOptimalDFTSize(rows)
#     ncols = cv2.getOptimalDFTSize(cols)
#     crow, ccol = rows/2 , cols/2

#     right = ncols - cols
#     bottom = nrows - rows
#     bordertype = cv2.BORDER_CONSTANT #just to avoid line breakup in PDF file
#     nimg = cv2.copyMakeBorder(data,0,bottom,0,right,bordertype, value = 0)

#     dft = cv2.dft(np.float32(nimg),flags = cv2.DFT_COMPLEX_OUTPUT)
#     dft_shift = np.fft.fftshift(dft)

#     magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

#     plt.subplot(221),plt.imshow(data, cmap = 'gray')
#     plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
#     plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#     plt.show()

#     return ret


###-------------------------------------------------------------------------###
# attemp to use low pass methods to filter out some high frequency events before
# pass data down
# def apply_filter(filter, img_to_filter = None):
    # kernelbank = {
    #     "5x5meanf32"    : np.ones((5,5),np.float32)/25,
    #     "scharr" : [[-3, 0, 3],
    #                 [-10,0,10],
    #                 [-3, 0, 3]],
    #     "sobelx" : [[-1, 0, 1],
    #                 [-2, 0, 2],
    #                 [-1, 0, 1]]
    # }

    # filter_names = [
    # "gaussian", 
    # # ##---------------LOW pass filter_names---------------------##
    # "global",
    # "adaptive",
    # "otsu",
    # "mean", 
    # "blur", 
    # "bilateral", 
    # # ##---------------HIGH pass filter_names---------------------##
    # "scharry", 
    # "sobely",
    # "laplacian",
    # ]
    
    # ## parameters for filters
    # params = {
    #     # ##---------------LOW pass filter_names---------------------##
    #     "mean"      : (img_to_filter, -1, np.ones((5,5), np.float32)/25),
    #     "blur"      : (img_to_filter, (5,5)),
    #     "gaussian"  : (img_to_filter, (5, 5), 0),
    #     "bilateral" : (img_to_filter,9,75,75)
    # }

    # funcdict = {
    #     "gaussian"  : cv2.GaussianBlur(img_to_filter, (5, 5), 0),
    #     "global"     : cv2.threshold(img_to_filter, 127, 255, 
    #         cv2.THRESH_BINARY),
    #     #  adaptive method: ADAPTIVE_THRESH_MEAN_C, block size, constant
    #     "adaptive"  : cv2.adaptiveThreshold(img_to_filter,127,
    #         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21, 0),
    #     "otsu"      : cv2.threshold(img_to_filter, 0, 255, 
    #         cv2.THRESH_BINARY + cv2.THRESH_OTSU),
    #     "mean"      : cv2.filter2D(img_to_filter, -1, np.ones((5,5), np.float32)/25),
    #     "blur"      : cv2.blur(img_to_filter, (5,5)),
    #     "bilateral" : cv2.bilateralFilter(img_to_filter,9,35,35, cv2.BORDER_REFLECT),
    #     # ##---------------HIGH pass filter_names---------------------##
    #     "scharry"   : cv2.Scharr(img_to_filter, cv2.CV_64F,0,1,0),
    #     "sobely"    : cv2.Sobel(img_to_filter, cv2.CV_64F,0,2,0),
    #     "laplacian" : cv2.Laplacian(img_to_filter, cv2.CV_64F)
    # }

    # for i in xrange(len(filter_names)):
    #     key = filter_names[i]
    #     cv2.imshow(key, funcdict[key])
    #     cv2.waitKey(0)

    # tmp = np.float32(np.copy(img1f64))
    # tmp = cv2.bilateralFilter(tmp, 9,35,35, cv2.BORDER_REFLECT)
    # cv2.imshow("bil", tmp)
    # cv2.waitKey(0)

    # tmp = np.float32(tmp)
    # tmp = cv2.Sobel(tmp, cv2.CV_32F,0,1,0)
    # cv2.imshow("sob", tmp)
    # cv2.waitKey(0)

    # tmp = cv2.bilateralFilter(tmp, 9,35,35, cv2.BORDER_REFLECT)
    # cv2.imshow("bil", tmp)
    # cv2.waitKey(0)
    # return funcdict[filter]


# ### -------------------------------------------------------------------------###
  ## FUNCS


## thresholding 
def Threshold():
    pass

## convert np array coordinates to opencv image coordinates
    # opencv coordinates        # numpy coordinates
    #       y                   #       x
    #       |                   #       |
    #       |                   #       |
    # x------------>            # y------------>
    #       |                   #       |
    #       |                   #       |
    #       v                   #       v


def np2cv(ox, oy, image = None):
    x, y = oy, ox
    return (x, y)

## running canny to find edges
def CannyThreshold(img = None):
    if (img == None):
        img = img1u8
    edges = cv2.Canny(img, params[0], params[1], params[2])
    cv2.putText(edges, str(params[idx]), (10, 30), 
        font, 0.7, float(255.0))
    cv2.imshow("canny", edges)
    f_count = 0
    return edges

## get angle from previously-found data
def GetAngle(inp, x, y, w, h):


    return angle

## finding connected pixels from propagating the given pixel
def FindConnected(inp, ref, xx, yy):
    ret = np.zeros(inp.shape, dtype = np.uint8)
    cv2.circle(ret, (yy, xx), 2, (255, 0, 0))
    mask = np.copy(ref)
    # test = np.zeros(mask.shape, dtype = np.uint8)
    # cv2.imshow("mask", 
    #     cv2.normalize(mask, test, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
    ret[xx, yy] = 255
    mask[xx, yy] = 1
    val = inp[xx, yy]

    # offset = ((-1, 0), (-1, -1), (0, -1), (1, -1), \
    #           (1, 0), (1, 1), (0, 1), (-1, 1))
    offset = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0))
    offset2 = ((-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0))

    q = Queue.Queue()
    q.put((xx, yy))
    coord = (xx, yy)
    while not q.empty():
        coord = q.get()
        found = 0
        # A loop over 5 connected neigbours
        for i in xrange(len(offset)):
            cc = tuple(np.add(coord, offset[i]))
            # print cc, mask[cc], inp[cc], val
            if (cc[0] >= 0 and cc[0] < vecshape[1][0] \
                and cc[1] >= 0 and cc[1] < vecshape[1][1] \
                and inp[cc]*val > 0):
                if (mask[cc] == 0):
                    found = found + 1
                else:
                    mask[cc] = 0
                    q.put(cc)
                    ret[cc] = 255
                    # cv2.imshow("processing...", ret)
                    # cv2.waitKey(1)

        # B if no neigbours were found from step A, expand search rediius +1
        if found == 0:
            for i in xrange(len(offset2)):
                cc = tuple(np.add(coord, offset2[i]))
                # print cc, mask[cc], inp[cc], val
                if (cc[0] >= 0 and cc[0] < vecshape[1][0] \
                    and cc[1] >= 0 and cc[1] < vecshape[1][1] \
                    and mask[cc] == 1 and inp[cc]*val > 0):
                    mask[cc] = 0
                    q.put(cc)
                    ret[cc] = 255
                    found = found + 1
                    # cv2.imshow("processing...", ret)
                    # cv2.waitKey(1)

        # C if no neigbours were found from step B, compute a range of 
        # slopes from previous data, then shoot beams to find a slope
        # with the strongest signal stack 
        # if found == 0:
        #     win_width = 5
        #     y0 = cc[1] - win_width
        #     if y0 < 0:
        #         y0 = 0;
        #     yax = np.zeros(0, dtype=np.uint8)
        #     xax = np.zeros(0, dtype=np.uint8)

        #     ## compute a projected slope by looking back samples in the given width
        #     for i in range(y0, y0+win_width):
        #         for j in xrange(ret.shape[0]):
        #             if ret[j, i] > 0:
        #                 xax = np.append(xax, i)
        #                 yax = np.append(yax, j)

        #     A = np.vstack([xax, np.ones(len(xax))]).T
        #     m, c = np.linalg.lstsq(A, yax)[0]
        #     print m, c

        #     ## examine the linear regression
        #     # plt.plot(xax, yax, 'o', label='Original data', markersize=10)
        #     # plt.plot(xax, m*xax + c, 'r', label='Fitted line')
        #     # plt.legend()
        #     # plt.show(); plt.savefig("images/tmp.png", dpi = 200)

        #     # img8uc3 = cv2.cvtColor(ret, cv2.COLOR_GRAY2RGB)
        #     # cv2.line(img8uc3, 
        #     #     np2cv((m*y0 + c).astype(int), y0),
        #     #     np2cv((m*(y0+win_width) + c).astype(int), y0+win_width),
        #     #     (0, 0, 255), 1, cv2.LINE_AA)
        #     # cv2.imshow("processing...", img8uc3)
        #     # cv2.waitKey(0)
            

    print "break at ", cc[0], cc[1], ", running linear regression from window 10 x 10"


    return ret

# ### -------------------------------------------------------------------------###
# # attemp to find edge on input
idx = 1
inc = 10
params = [20, 350, 3]
use_gaussian = False
pre_filter = False
f_count = 0

raw = np.copy(img1u8)
mask = np.zeros(img1u8.shape, dtype = bool)


CannyThreshold()

while True:
    keydown = cv2.waitKey(20)
    if keydown & 0xff == 27:
        break
    elif keydown & 0xff == ord('1'):
        idx = 0
        inc = 10
        CannyThreshold()
    elif keydown & 0xff == ord('2'):
        idx = 1
        inc = 10
        CannyThreshold()
    elif keydown & 0xff == ord('3'):
        pass
    elif keydown & 0xff == ord('c'):
        CannyThreshold(img1u8)
    elif keydown & 0xff == ord('w'):
        params[idx] = params[idx] + inc
        CannyThreshold()
    elif keydown & 0xff == ord('s'):
        params[idx] = params[idx] - inc
        CannyThreshold()
    elif keydown & 0xff == ord('g'):
        use_gaussian = not use_gaussian
        print use_gaussian
        if (use_gaussian):
            pre_filter = False
            img1u8 = np.uint8(cv2.GaussianBlur(img1u8, (5,5), 0))
            # img1u8 = convolve(img1u8, sobelY)
            cv2.imshow("filtered", img1u8)
            cv2.waitKey(0)
            ret, img1u8 = cv2.threshold(
                CannyThreshold(), 
                0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            cv2.imshow("filtered", img1u8)
            cv2.waitKey(0)
            img1u8 = cv2.dilate(
                img1u8, 
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations = 1)
        else : 
            img1u8 = raw
        # CannyThreshold()
        cv2.imshow("filtered", img1u8)
        cv2.waitKey(1)
    elif keydown & 0xff == ord('f'):
        pre_filter = not pre_filter
        # if (pre_filter):
            # use_gaussian = False
        if f_count == 0:
            f_count = f_count + 1

            h = vecshape[1][0]
            w= vecshape[1][1]
            eigen = cv2.cornerEigenValsAndVecs(img1u8, 15, 3)
            eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
            flow = eigen[:,:,2]
            vis = img1u8.copy()
            vis[:] = (192 + np.uint32(vis)) / 2
            d = 12
            points =  np.dstack( np.mgrid[d/2:w:d, d/2:h:d] ).reshape(-1, 2)
            for x, y in np.int32(points):
                vx, vy = np.int32(flow[y, x]*d)
                cv2.line(vis, (x-vx, y-vy), (x+vx, y+vy), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('input', img1u8)
            cv2.imshow('flow', vis)
            cv2.waitKey(0)
            


            ## compute segments using hough transformation
            # edges = CannyThreshold()
            # lines = cv2.HoughLinesP(img1u8,
            #     1,
            #     np.pi/180,
            #     20,
            #     minLineLength=30,
            #     maxLineGap=5)
            # color = cv2.cvtColor(img1u8, cv2.COLOR_GRAY2RGB)
            # print lines
            # for line in lines:
            #     x1,y1,x2,y2 = line[0]
            #     cv2.line(color,(x1,y1),(x2,y2),(0,255,0),2)
            # cv2.imshow("mask", color)
            

            # img1u8 = cv2.bilateralFilter(img1u8, 9,70, 70)
            # img1u8 = cv2.erode(
            #     img1u8, 
            #     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
            #     )

            mask = img1u8.astype(bool)
            # cv2.imshow("mask", img1u8)

        # else:
        #     img1u8 = raw
        elif f_count == 1:
            f_count = f_count + 1
            FindConnected(img1f64, mask, 230, 15)
        elif f_count == 2:
            f_count = f_count + 1
            pass
    else:
        pass


# ###------------------------------------tackerBar TEST--------------------------###
# def CannyThreshold(kernel_size):
#     detected_edges = cv2.GaussianBlur(gray,(3,3),0)
#     detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio, kernel_size)
#         # apertureSize = kernel_size, L2gradient  = True)
#     dst = cv2.bitwise_and(img1u8,img1u8,mask = detected_edges)  # just add some colours to edges from original image.
#     cv2.imshow('canny demo',dst)
  
# lowThreshold = 0
# max_lowThreshold = 100
# ratio = 2
# kernel_size = 3
# max_kernel_size = win_width
  
# gray = img1u8
  
# cv2.namedWindow('canny demo')
  
# cv2.createTrackbar('kernel_size','canny demo', kernel_size, max_kernel_size, CannyThreshold)
  
# CannyThreshold(kernel_size)  # initialization
# cv2.waitKey(0)

cv2.destroyAllWindows()