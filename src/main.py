#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np
from sklearn import preprocessing
from cv2 import imshow

min_max_scaler = preprocessing.MinMaxScaler()

import argparse
import math
import Queue
import os, sys
from os.path import expanduser
HOME = expanduser("~")

fig = plt.gcf()
fig.set_size_inches(10, 6)

vecshape = ([361, 1616], [361, 1386], [1386, 1616])

data_name = 'x800'
data_path = os.path.join(HOME, "rnd/data/" + data_name + ".dat")
data = np.fromfile(data_path, dtype=np.float64)
img_raw = data.reshape(vecshape[1])
#  img_raw = (img_raw - img_raw.min()) /img_raw.ptp()

font = cv2.FONT_HERSHEY_SIMPLEX

#  construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
    help="Path to the image to be thresholded")
ap.add_argument("-t", "--threshold", type=int, default=128,
    help="Threshold value")
args = vars(ap.parse_args())

###-------------------------------------------------------------------------###
#  ## CONSTANTS

#  construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

#  construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

#  construct the Laplacian kernel used to detect edge-like
#  regions of an image
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

#  construct the Sobel x-axis kernel
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

#  construct the Sobel y-axis kernel
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

#  construct the Scharr x-axis kernel
ScharrX = np.array((
    [-3, 0, 3],
    [-10, 0, 10],
    [-3, 0, 3]), dtype="int")

#  construct a diagnal kernel
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

#  construct the kernel bank, a list of kernels we're going
#  to apply using both our custom `convolve` function and
#  OpenCV's `filter2D` function
kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY),
    ("diagnal", Diagnal_1)
)

#  ## filter bank
#  filterBank = {
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
#  }


#  # neighbor bank
neighborBank = {
                "4" : ((-1, 0), (0, 1), (1, 0), (0, -1)),
                "8" : ((-1, 0), (0, 1), (1, 0), (0, -1),
                       (-1, 1), (1, 1), (1, -1), (-1, -1)),
                }

###-------------------------------------------------------------------------###

###-------------------------------------------------------------------------###
#  # customized convolution for diagnal edge detection
def convolve(image, kernel):
    #  grab the spatial dimensions of the image, along with
    #  the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    #  allocate memory for the output image, taking care to
    #  "pad" the borders of the input image so the spatial
    #  size (i.e., width and height) are not reduced
    pad = (kW - 1) / 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    #  loop over the input image, "sliding" the kernel across
    #  each (x, y)-coordinate from left-to-right and top to
    #  bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            #  extract the ROI of the image by extracting the
            #  *center* region of the current (x, y)-coordinates
            #  dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            #  perform the actual convolution by taking the
            #  element-wise multiplicate between the ROI and
            #  the kernel, then summing the matrix
            k = (roi * kernel).sum()

            #  store the convolved value in the output (x,y)-
            #  coordinate of the output image
            output[y - pad, x - pad] = k

    #  rescale the output image to be in the range [0, 255]
    #  output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    #  return the output image
    return output


###-------------------------------------------------------------------------###
#  ## FFT wrapper
#  def fft(inp, axis):
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
#  attemp to use low pass methods to filter out some high frequency events before
#  pass data down
#  def apply_filter(filter, img_to_filter = None):
    #  kernelbank = {
    #     "5x5meanf32"    : np.ones((5,5),np.float32)/25,
    #     "scharr" : [[-3, 0, 3],
    #                 [-10,0,10],
    #                 [-3, 0, 3]],
    #     "sobelx" : [[-1, 0, 1],
    #                 [-2, 0, 2],
    #                 [-1, 0, 1]]
    #  }

    #  filter_names = [
    #  "gaussian",
    #  # ##---------------LOW pass filter_names---------------------##
    #  "global",
    #  "adaptive",
    #  "otsu",
    #  "mean",
    #  "blur",
    #  "bilateral",
    #  # ##---------------HIGH pass filter_names---------------------##
    #  "scharry",
    #  "sobely",
    #  "laplacian",
    #  ]

    #  ## parameters for filters
    #  params = {
    #     # ##---------------LOW pass filter_names---------------------##
    #     "mean"      : (img_to_filter, -1, np.ones((5,5), np.float32)/25),
    #     "blur"      : (img_to_filter, (5,5)),
    #     "gaussian"  : (img_to_filter, (5, 5), 0),
    #     "bilateral" : (img_to_filter,9,75,75)
    #  }

    #  funcdict = {
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
    #  }

    #  for i in xrange(len(filter_names)):
    #     key = filter_names[i]
    #     cv2.imshow(key, funcdict[key])
    #     cv2.waitKey(0)

    #  tmp = np.float32(np.copy(img_f64c1))
    #  tmp = cv2.bilateralFilter(tmp, 9,35,35, cv2.BORDER_REFLECT)
    #  cv2.imshow("bil", tmp)
    #  cv2.waitKey(0)

    #  tmp = np.float32(tmp)
    #  tmp = cv2.Sobel(tmp, cv2.CV_32F,0,1,0)
    #  cv2.imshow("sob", tmp)
    #  cv2.waitKey(0)

    #  tmp = cv2.bilateralFilter(tmp, 9,35,35, cv2.BORDER_REFLECT)
    #  cv2.imshow("bil", tmp)
    #  cv2.waitKey(0)
    #  return funcdict[filter]


# ### -------------------------------------------------------------------------###
# ### FUNCS

def click_and_crop(event, x, y, flags, param):
    #  grab references to the global variables
    global refPt, cropping

    #  if the left mouse button was clicked, record the starting
    #  (x, y) coordinates and indicate that cropping is being
    #  performed
    if event == cv2.EVENT_LBUTTONDOWN:
        print "click at ", (x, y), " = ", mask1[y, x]
        refPt = [(x, y)]
        cropping = True

    #  check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        #  record the ending (x, y) coordinates and indicate that
        #  the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        #  draw a rectangle around the region of interest

#  # thresholding
def Threshold():
    pass

#  # convert np array coordinates to opencv image coordinates
    #  opencv coordinates        # numpy coordinates
    #       y                   #       x
    #       |                   #       |
    #       |                   #       |
    #  x------------>            # y------------>
    #       |                   #       |
    #       |                   #       |
    #       v                   #       v


def np2cv(ox, oy, image=None):
    x, y = oy, ox
    return (x, y)


#  # get angle from previously-found data
def GetAngle(inp, x, y, w, h):
    angle = 2
    return angle


#  offset1 = ((-1, 0), (-1, -1), (0, -1), (1, -1), \
#           (1, 0), (1, 1), (0, 1), (-1, 1))
offset1 = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0))
offset2 = ((-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0))
#  # finding connected pixels from propagating the given pixel
def FindConnected(inp, xx, yy):
    print inp.shape
    ret = np.zeros(inp.shape, dtype=np.uint8)
    visited = np.ones(inp.shape, dtype=np.bool)
    ret[xx][yy] = 255
    visited[xx][yy] = True
    start_val = inp[xx][yy]

    q = Queue.Queue()
    q.put((xx, yy))
    point = (xx, yy)
    prevdir = (-1, -1)
    while not q.empty():
        point = q.get()
        old = 0
        newfound = 0
        #  A loop over 5 connected neighbors
        for i in xrange(len(offset1)):
            x, y = tuple(np.add(point, offset1[i]))
            if (x >= 0 and x < vecshape[1][0] and 
                y >= 0 and y < vecshape[1][1] and 
                inp[x][y] * start_val > 0):
                print (x, y),  mask1[x][y], inp[x][y], start_val
                if (visited[x][y] == False):
                    old = old + 1
                else:
                    newfound = newfound + 1
                    visited[x][y] = False
                    ret[x][y] = 255
                    q.put((x, y))
#                      print "adding ", (x, y)
#              print ''

        #  B if no neigbours were old from step A, expand search rediius +1
#         if old == 0:
#             for i in xrange(len(offset2)):
#                 cc = tuple(np.add(point, offset2[i]))
#                 # print cc, mask[cc], inp[cc], start_val
#                 if (cc[0] >= 0 and cc[0] < vecshape[1][0] \
#                     and cc[1] >= 0 and cc[1] < vecshape[1][1] \
#                     and mask[cc] == 1 and inp[cc] * start_val > 0):
#                     mask[cc] = 0
#                     q.put(cc)
#                     ret[cc] = 255
#                     old = old + 1

        #  C if no neigbours were old from step B, compute a range of
        #  slopes from previous data, then shoot beams to find a slope
        #  with the strongest signal stack
        #  if old == 0:
        #     win_width = 5
        #     y0 = cc[1] - win_width
        #     if y0 < 0:
        #         y0 = 0;
        #     yax = np.zeros(0, dtype=np.uint8)
        #     xax = np.zeros(0, dtype=np.uint8)
        #
        #     ## compute a projected slope by looking back samples in the given width
        #     for i in range(y0, y0+win_width):
        #         for j in xrange(ret.shape[0]):
        #             if ret[j, i] > 0:
        #                 xax = np.append(xax, i)
        #                 yax = np.append(yax, j)
        #
        #     A = np.vstack([xax, np.ones(len(xax))]).T
        #     m, c = np.linalg.lstsq(A, yax)[0]
        #     print m, c
        #
        #     ## examine the linear regression
        #     # plt.plot(xax, yax, 'o', label='Original data', markersize=10)
        #     # plt.plot(xax, m*xax + c, 'r', label='Fitted line')
        #     # plt.legend()
        #     # plt.show(); plt.savefig("images/tmp.png", dpi = 200)
        #
        #     # img8uc3 = cv2.cvtColor(ret, cv2.COLOR_GRAY2RGB)
        #     # cv2.line(img8uc3,
        #     #     np2cv((m*y0 + c).astype(int), y0),
        #     #     np2cv((m*(y0+win_width) + c).astype(int), y0+win_width),
        #     #     (0, 0, 255), 1, cv2.LINE_AA)
        #     # cv2.imshow("processing...", img8uc3)
        #     # cv2.waitKey(0)

        #  D if no neighbor old from steps above, use flow matrix computed from
        #  the covariance matrix of derivatives over the neighborhood, where the
        #  derivatives are computed using the sobel() operator
        #
        if newfound == 0 and q.empty() == True:
            dx, dy = get_guide_vec(point, flow2, 0, 7)
            if (prevdir != (-1, -1)):
                if (np.dot(prevdir, (dx,dy)) < 0):
                    dx = prevdir[0]
                    dy = prevdir[1]
            newx = point[0] + dx
            newy = point[1] + dy
            nextp = tuple([newx, newy])
            print point, (dx, dy)

            # boundary check
            if (nextp[0] >= 0 and nextp[0] < vecshape[1][0] and
                nextp[1] >= 0 and nextp[1] < vecshape[1][1]):
                mask1[nextp] = True
                q.put(nextp)
                ret[nextp] = 255
                prevdir = (newx, newy)

                ## draw a new point and its flow direction
                bg = cv2.cvtColor(img_f32c1_norm, cv2.COLOR_GRAY2BGRA)
                fg = cv2.cvtColor(np.float32(ret), cv2.COLOR_GRAY2BGRA)
                fg[:,:,0] = 0
                fg[:,:,1] = 0
                fg[:,:,3] = fg[:,:,2]

                t = np.arctan(dx/dy)
                color = (0, 0.9, 0.9, 1)
                if (t > 0):
                    color = (0, 0.9, np.cos(t)*0.9, 1)
                elif (t < 0):
                    color = (0, np.cos(t)*0.9, 0.9, 1)
#                 print np.degrees(np.cos(t))
#                 color = (0, 0, 255, 1)
                cv2.line(bg,
                         (int(point[1]), int(point[0])),
                         (int(point[1]+ dy*10.), int(point[0] + dx*10.)),
                         color, 1, cv2.LINE_AA)
                img = cv2.add(bg, fg)

                cv2.imshow("processing...", img)
                cv2.moveWindow("processing...", 20, 780)
                key = cv2.waitKey(0)
                if key & 0xff == 27:
                    break

    print "break at ", (x, y), ", running linear regression from window 10 x 10"
#      cv2.imshow("processing...", ret)
#      cv2.waitKey(0)

    return ret


def rotate(vec, deg):
    x1 = vec[0] * math.cos(deg) - vec[1] * math.sin(deg)
    y1 = vec[1] * math.cos(deg) + vec[0] * math.sin(deg)
    return (x1, y1)

def mag(vec):
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])

def normalize(vec):
    m = mag(vec)
    return (vec[0] / m, vec[1] / m)

def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

#  # get a guide vector from flow matrix
#  @param w the width of window
#  @param halfw
#  @param mode 0: the sample area is an AA rectangle
#              1: the sample area is a FA rectangle
#              2: the sample area is a FA cone align
def get_guide_vec(p, mat, mode=0, w=5, deg=-999.):
    if (w % 2 != 1):
        print "window width for get_guide_vec must be an odd number"
        sys.exit()

    ret = (-1., -1.)
    #  if not to use a window, i.e., not interpolate vectors
    if (mode == 0):
        ret = mat[p][::-1]
    #  if to use a window, parameter mode determines which method to be used
    else:
        halfw = (w - 1) / 2
        if (halfw < 0):
            halfw = 0
        #  # axis-aligned rectangle
        if (mode == 1):
            topleft = np.float64(np.subtract(p, (w, halfw)))
            count = 0.0
            s = (0, 0)
            for i in xrange(w):
                for j in xrange(w):
                    x = topleft[0] + i
                    y = topleft[1] + j
                    if (x < 0 or y < 0 or
                        x >= mat.shape[0] or y >= mat.shape[1]):
                        continue

                    count = count + 1
                    weight = 1 - min(abs(i - halfw), abs(j - halfw)) / float(halfw)
                    tan = mat[x, y][::-1]
                    s = s + np.multiply(tan, weight)
            ret = normalize(s)

        #  # flow-aligned rectangle
        elif(mode == 2):
            count = 0.0
            for i in xrange(w):
                for j in xrange(w):
                    x = topleft[0] + i
                    y = topleft[1] + j
                    if (deg != -999.):
                        x, y = rotate((i, j), deg)

                    if (x < 0 or y < 0 or x >= mat.shape[0] or y >= mat.shape[1]):
                        continue
                    count = count + 1
                    weight = 0.5
                    if x == p[0] and y == p[1]:
                        weight = 1
                    tan = mat[x, y][::-1]
                    ret = ret + weight * tan
            ret = ret / count
            rad = math.atan2(ret[0], ret[1])
            deg = math.degrees(rad)
            print "average angle = ", rad, " = ", deg
            return np.multiply(normalize(ret), 2)
            pass
        elif(mode == 3):
            pass

    return ret

def mmnorm_to_u8(img):
    if img is None:
        pass
    else:
        return np.uint8((img - img.min()) / img.ptp() * 255)

def mmnorm_to_f32(img):
    if img is None:
        pass
    else:
        return np.float32((img - img.min()) / img.ptp())

def mmnorm(img):
    if img is None:
        pass
    else:
        return (img - img.min()) / img.ptp()

# ### -------------------------------------------------------------------------###
img_mask = np.zeros(img_raw.shape, dtype=np.bool)
img_f64c1 = np.ma.array(np.asarray(img_raw),
                        mask=img_mask,
                        fill_value=9999,
                        dtype=np.float64)
img_f32c1 = np.ma.array(np.asarray(np.float32(img_raw)),
                        mask=img_mask,
                        fill_value=9999,
                        dtype=np.float32)
img_f16c1 = np.ma.array(np.asarray(np.float16(img_raw)),
                        mask=img_mask,
                        fill_value=9999,
                        dtype=np.float16)
img_f32c1_norm = (img_f32c1 - img_f32c1.min()) / img_f32c1.ptp()
#  img_u8c1 = np.uint8(cv2.normalize(img_f64c1, img_f64c1, 0, 255, cv2.NORM_MINMAX))
img_u8c1 = np.ma.array(np.asarray(mmnorm_to_u8(img_raw)),
                        mask=img_mask,
                        fill_value=9999,
                        dtype=np.uint8)


#  initialize a graphics and display raw image in u8c1
#  cv2.namedWindow("image")
#  cv2.moveWindow("image", 20, 20)
cv2.imshow("Raw", img_f32c1_norm)
cv2.moveWindow("Raw", 20, 20)

#  # attemp to find edge on input
idx = 1
inc = 10
params = [70, 330, 5]
use_filter = False
g_count = 0

#  eigenvectors of flow
flow1 = []
flow2 = []
mask = np.zeros(img_raw.shape, dtype=bool)
mask1 = []
mask2 = []
mask3 = []


#  # running canny to find edges
def CannyThreshold(img=None):
    edges = cv2.Canny(img, params[0], params[1], params[2])
    if (use_filter == True):
        txt = "min = " + str(params[0]) + "   max = " + str(params[1])
        cv2.putText(edges, txt, (10, 30),
            font, 0.7, float(255.0))
    else :
        cv2.putText(edges, "raw", (10, 30),
            font, 0.7, float(255.0))
#      cv2.imshow("Canny", edges)
#      cv2.moveWindow("Canny", 20, 20)
    return edges



curr = np.ma.copy(img_f32c1)

while True:
    keydown = cv2.waitKey(20)
    if keydown & 0xff == 27:
        break
    elif keydown & 0xff == ord('1'):
        idx = 0
        inc = 10
        CannyThreshold(mmnorm_to_u8(curr))
#         CannyThreshold(mask)
    elif keydown & 0xff == ord('2'):
        idx = 1
        inc = 10
        CannyThreshold(mmnorm_to_u8(curr))
#         CannyThreshold(mask)
    elif keydown & 0xff == ord('3'):
        idx = 2
        inc = 1
#         CannyThreshold(mask)
        pass
    elif keydown & 0xff == ord('c'):
        pass
    elif keydown & 0xff == ord('w'):
        params[idx] = params[idx] + inc
        CannyThreshold(mmnorm_to_u8(curr))
#         CannyThreshold(mask)
    elif keydown & 0xff == ord('s'):
        params[idx] = params[idx] - inc
        CannyThreshold(mmnorm_to_u8(curr))
#         CannyThreshold(mask)
    elif keydown & 0xff == ord('f'):
        use_filter = not use_filter
        g_count = 0
        if (use_filter == True):
            pre_filter = False

            #  create a CLAHE object (Arguments are optional). It largely
#              increase the layer resolution, may be used as level 4 mask.
#              clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#              cl1 = clahe.apply(np.uint16(img_raw))

            curr = cv2.GaussianBlur(img_f32c1, (3, 3), 0)

            # sobel descretize too much !!!
#              curr = convolve(curr, sobelY)

            #  use canny for level 1 mask
            cannyEdges = CannyThreshold(mmnorm_to_u8(curr))
            #  # dilate the result of canny for rough mask
            mask1 = cv2.dilate(cannyEdges,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)),
                              iterations=1)
            mask1 = cv2.erode(mask1,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)),
                              iterations=1)

            #  construct level 2 mask
            #  otsu localize some, but also introduced dots
            ret, threshold = cv2.threshold(mmnorm_to_u8(curr),
                                      0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #  gives good edges, but cut off too much
            mask2 = cv2.bilateralFilter(threshold, 75, 15, 7)

            #  construct a level 3 mask from gaussian filtered result
            ret, mask3 = cv2.threshold(np.uint8(curr),
                                      0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cv2.imshow("mask1", mask1)
            cv2.moveWindow("mask1", 20, 20)
            cv2.setMouseCallback("mask1", click_and_crop)

            tmp = np.ndarray(mask1.shape, dtype=np.bool)
            for i in range(len(mask1)):
                for j in range(len(mask1[0])):
                    #  TODO : use values from clicked position
                    if (mask1[i][j] > 0):
                        tmp[i][j] = False
                    else:
                        tmp[i][j] = True
            mask1 = tmp
#
            ## add auxiliary computed mask
            curr = np.ma.masked_array(curr,
                                      mask1, fill_value = -999)

#              cv2.imshow("mask2", mask2)
#              cv2.imshow("mask3", mask3)

                
#              todraw = np.zeros(mask1.shape, np.uint8)
#              todraw = cv2.cvtColor(todraw, cv2.COLOR_GRAY2BGR)
#              # test contour on level 2 mask
#              im2, contours, hierarchy = cv2.findContours( mask1,
#                                                           cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
#              cnt1 = contours[0]
#              print cnt1
#
#              # centeroid of the first contour
#              M = cv2.moments(cnt1)
#              cy = int(M['m10']/M['m00'])
#              cx = int(M['m01']/M['m00'])
#              print cx, cy
#
#              # area of the first contour
#              area = cv2.contourArea(cnt1)
#              print area
#
#              todraw = cv2.drawContours(todraw, contours, 1, (0, 0, 255), -1)
#              cv2.imshow("pressed f", todraw)
#              cv2.moveWindow("pressed f", 20, 20)
        else :
            cv2.destroyWindow("mask")
            curr = np.ma.copy(img_f32c1)
#              CannyThreshold(mmnorm_to_u8(curr))


    elif keydown & 0xff == ord('g'):
        if g_count == 0:
            g_count = g_count + 1

            h = vecshape[1][0]
            w = vecshape[1][1]
            eigen = cv2.cornerEigenValsAndVecs(np.float32(curr), 15, 9)
            eigen = eigen.reshape(h, w, 3, 2)  #  [[e1, e2], v1, v2]

            eigenvalues = eigen[:, :, 0]
            flow1 = eigen[:, :, 1]
            flow2 = eigen[:, :, 2]

#              # regulate direction of the flow field
#              for i in xrange(len(flow2)):
#                  for j in xrange(len(flow2[0])):
#                      if (flow2[i][j][0] < 0 and flow2[i][j][1] < 0):
#                         flow2[i][j][0] = -flow2[i][j][0]
#                         flow2[i][j][1] = -flow2[i][j][1]
#                     elif (flow2[i][j][0] < 0 and flow2[i][j][1] > 0):
#                         flow2[i][j][0] = -flow2[i][j][0]
#                         flow2[i][j][1] = -flow2[i][j][1]


            vis = cv2.cvtColor(img_f32c1_norm, cv2.COLOR_GRAY2RGB)
            d = 8
            k = d - 2
            points = np.dstack(np.mgrid[d / 2:w:d, d / 2:h:d]).reshape(-1, 2)

            for x, y in np.int32(points):
                vx, vy = flow2[y, x]
                t = np.arctan(vy / vx)
                color = (0, 0.9, 0.9)
                if (t > 0):
                    color = (0, 0.9, np.cos(t) * 0.9)
                elif (t < 0):
                    color = (0, np.cos(t) * 0.9, 0.9)
                cv2.line(vis, (x, y), (x + int(vx * k), y + int(vy * k)), color, 1, cv2.LINE_AA)

            cv2.imshow('flow 2 : ', vis)
            cv2.moveWindow("flow 2 : ", 20, 400)

            #  # compute segments using hough transformation
            #  edges = CannyThreshold()
            #  lines = cv2.HoughLinesP(img_u8c1,
            #     1,
            #     np.pi/180,
            #     20,
            #     minLineLength=30,
            #     maxLineGap=5)
            #  color = cv2.cvtColor(img_u8c1, cv2.COLOR_GRAY2RGB)
            #  print lines
            #  for line in lines:
            #     x1,y1,x2,y2 = line[0]
            #     cv2.line(color,(x1,y1),(x2,y2),(0,255,0),2)
            #  cv2.imshow("mask", color)

        elif g_count == 1:
            g_count = g_count + 1
            FindConnected(curr, 225, 15)
#         elif g_count == 2:
#             g_count = g_count + 1
#             pass
    else:
        pass


# ###------------------------------------tackerBar TEST--------------------------###
#  def CannyThreshold(kernel_size):
#     detected_edges = cv2.GaussianBlur(gray,(3,3),0)
#     detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio, kernel_size)
#         # apertureSize = kernel_size, L2gradient  = True)
#     dst = cv2.bitwise_and(img_u8c1,img_u8c1,mask = detected_edges)  # just add some colours to edges from original image.
#     cv2.imshow('canny demo',dst)

#  lowThreshold = 0
#  max_lowThreshold = 100
#  ratio = 2
#  kernel_size = 3
#  max_kernel_size = win_width

#  gray = img_u8c1

#  cv2.namedWindow('canny demo')

#  cv2.createTrackbar('kernel_size','canny demo', kernel_size, max_kernel_size, CannyThreshold)

#  CannyThreshold(kernel_size)  # initialization
#  cv2.waitKey(0)

cv2.destroyAllWindows()
