import matplotlib
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import cv2 as cv

#load the images
map_1 = cv.imread('images/509.jpg')

#Resizing the images
map_1 = cv.resize(map_1, (10000,10000))

#Convert to green blue
map_1 = cv.cvtColor(map_1, cv.COLOR_BGR2GRAY)

plt.imshow(map_1)
plt.show()