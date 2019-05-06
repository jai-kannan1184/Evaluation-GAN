import numpy as np

import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import cv2

#Define the mean square function
def mse(imageA, imageB):
    err= np.sum((imageA.astype('float') - imageB.astype('float')) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

#Define the compare image function using ssim
def compare_image(imageA, imageB, title):
    m= mse(imageA, imageB)
    s= ssim(imageA, imageB)
    fig= plt.figure('MSE: %.2f, SSIM: %.2f' % (m, s))

    #Display image A
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap= plt.cm.gray)
    plt.axis('off')

    #Display image B
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap= plt.cm.gray)
    plt.axis('off')

    plt.show()
#Load the images
map_1= cv2.imread('images/509.jpg')
map_2= cv2.imread('images/509actual.png')
map_3= cv2.imread('images/508-outputs.png')
map_4= cv2.imread('images/508-targets.png')
map_5= cv2.imread('images/1.png')
map_6= cv2.imread('images/1t.png')

#Resize into zero patting
map_1= cv2.resize(map_1, (10000, 10000))
map_2= cv2.resize(map_2, (10000, 10000))
map_3= cv2.resize(map_3, (10000, 10000))
map_4= cv2.resize(map_4, (10000, 10000))
#map_5= cv2.resize(map_5, (10000, 10000))
#map_6= cv2.resize(map_6, (10000, 10000))

#convert to gray
map_1= cv2.cvtColor(map_1, cv2.COLOR_BGR2GRAY)
map_2= cv2.cvtColor(map_2, cv2.COLOR_BGR2GRAY)
map_3= cv2.cvtColor(map_3, cv2.COLOR_BGR2GRAY)
map_4=cv2.cvtColor(map_4, cv2.COLOR_BGR2GRAY)
map_5=cv2.cvtColor(map_5, cv2.COLOR_BGR2GRAY)
map_6=cv2.cvtColor(map_6, cv2.COLOR_BGR2GRAY)

#initilize the figure
fig = plt.figure('maps')
images = ('First map', map_1), ('Second map', map_2), ('Third map', map_3), ('Fourth map', map_4), ('Fifth map', map_5), ('Sixith map', map_6)

#loop the images
for(i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 6, i+1)
    #plt.imshow(images, cmap= plt.cm.gray)
    plt.axis('off')

plt.show()
compare_image(map_1,map_1,'same image')
compare_image(map_1, map_2, 'First map vs. Second map')
compare_image(map_3, map_4, 'Third map vs. Fourth map')
compare_image(map_5, map_6, 'Fifth map vs Sixth map')