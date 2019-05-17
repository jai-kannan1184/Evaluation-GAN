from __future__ import division
import numpy as np
from skimage.util.dtype import dtype_range
from skimage._shared.utils import skimage_deprecation, warn
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import cv2

#check the numpy array
def _assert_compatiable(imageA, imageB):
    if not imageA.dtype == imageB.dtype:
        raise ValueError('Images must be same dtype')
    if not imageA.shape == imageB.shape:
        raise ValueError('Input images must be same dimension')
    return
#check and conver the array to float
def _as_floats(imageA,imageB):
    float_type = np.result_type(imageA.dtype,imageB.dtype,np.float32)
    if imageA.dtype != float_type:
        imageA = imageA.astype(float_type)
    if imageB.dtype != float_type:
        imageB = imageB.astype(float_type)
    return imageA, imageB
#Define the mean square function
def mse(imageA, imageB):
    err= np.sum((imageA.astype('float') - imageB.astype('float')) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err
#compare the mse of the two images
def compare_mse(imageA,imageB):
    _assert_compatiable(imageA,imageB)
    imageA,imageB = _as_floats(imageA,imageB)
    return np.mean(np.square(imageA - imageB), dtype=np.float64)
#compare the normalised mean squared error
def nrmse(imageA, imageB, norm_type='Euclidean'):
    _assert_compatiable(imageA,imageB)
    imageA, imageB = _as_floats(imageA, imageB)
    norm_type = norm_type.lower()
    if norm_type == 'euclidean':
        denom = np.sqrt(np.mean((imageA*imageB), dtype=np.float64))
    elif norm_type == 'min-max':
        denom = imageA.max() - imageA.min()
    elif norm_type == 'mean':
        denom = imageA.mean()
    else:
        raise ValueError('image unsupported norm_type')
    return np.sqrt(compare_mse(imageA, imageB)) / denom
#compare the PSNR of the two images
def compare_psnr(imageA, imageB, data_range=None, dynamic_range=None):
    _assert_compatiable(imageA, imageB)
    if dynamic_range is not None:
        warn('dynamic_range has been deprecated in favor of data_range. The dynamic_range argument will be removed',skimage_deprecation)
        data_range= dynamic_range
    if data_range is None:
        dmin, dmax = dtype_range[imageA.dtype.type]
        true_min, true_max = np.min(imageA), np.max(imageA)
        if true_max > dmax or true_min < dmin:
            raise ValueError('intensity value out of range for the data type')
        if true_min >= 0:
            data_range = dmax
        else:
            data_range = dmax - dmin
    imageA, imageB = _as_floats(imageA, imageB)
    err = compare_mse(imageA, imageB)
    return 10 * np.log10((data_range ** 2) / err)
def dssim (imageA, imageB):
    _assert_compatiable(imageA, imageB)
    _as_floats(imageA, imageB)
    R = ssim(imageA, imageB)
    return 10 * np.log10(R)

#Define the compare image function using ssim
def compare_image(imageA, imageB, title):
    m= mse(imageA, imageB)
    s= ssim(imageA, imageB)
    n= nrmse(imageA,imageB)
    p= compare_psnr(imageA,imageB)
    d= dssim(imageA, imageB)
    fig= plt.figure('MSE: %.2f, SSIM: %.2f, NRMSE: %.2f, PSNR: %.2f, DSSIM: %.2f' % (m, s, n, p, d))

    #Display image A
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap= plt.cm.gray)
    plt.axis('on')

    #Display image B
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap= plt.cm.gray)
    plt.axis('on')

    plt.show()
#Load the images
map_1= cv2.imread('images/509.jpg')
map_2= cv2.imread('images/509actual.png')
map_3= cv2.imread('images/508-outputs.png')
map_4= cv2.imread('images/508-targets.png')
map_5= cv2.imread('images/1.png')
map_6= cv2.imread('images/1t.png')
map_7= cv2.imread('images/test1.png')
map_8= cv2.imread('images/test2.png')

#Resize into zero patting
map_1= cv2.resize(map_1, (10000, 10000))
map_2= cv2.resize(map_2, (10000, 10000))
map_3= cv2.resize(map_3, (10000, 10000))
map_4= cv2.resize(map_4, (10000, 10000))
#map_5= cv2.resize(map_5, (10000, 10000))
#map_6= cv2.resize(map_6, (10000, 10000))
map_7= cv2.resize(map_7, (1000, 1000))
map_8= cv2.resize(map_8, (1000, 1000))

#convert to gray
map_1= cv2.cvtColor(map_1, cv2.COLOR_BGR2GRAY)
map_2= cv2.cvtColor(map_2, cv2.COLOR_BGR2GRAY)
map_3= cv2.cvtColor(map_3, cv2.COLOR_BGR2GRAY)
map_4= cv2.cvtColor(map_4, cv2.COLOR_BGR2GRAY)
map_5= cv2.cvtColor(map_5, cv2.COLOR_BGR2GRAY)
map_6= cv2.cvtColor(map_6, cv2.COLOR_BGR2GRAY)
#map_7= cv2.cvtColor(map_7, cv2.COLOR_BGR2GRAY)
#map_8= cv2.cvtColor(map_8, cv2.COLOR_BGR2GRAY)
#initilize the figure
fig = plt.figure('maps')
images = ('First map', map_1), ('Second map', map_2), ('Third map', map_3), ('Fourth map', map_4), ('Fifth map', map_5), ('Sixith map', map_6), ('Seventh map', map_7), ('Eights map', map_8)

#loop the images
for(i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 9, i+1)
    #plt.imshow(images, cmap= plt.cm.gray)
    plt.axis('off')

plt.show()
compare_image(map_1, map_1,'same image')
compare_image(map_1, map_2, 'First map vs. Second map')
compare_image(map_3, map_4, 'Third map vs. Fourth map')
compare_image(map_5, map_6, 'Fifth map vs Sixth map')
compare_image(map_7, map_8, 'Seventh map vs. Eighth map')