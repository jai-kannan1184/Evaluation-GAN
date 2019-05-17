from __future__ import print_function
from _elementtree import Element, SubElement
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def get_coordinates(map_image):
   #f_number = getFileNumber(map_image)
   original = cv2.imread('Test.png')
   trimmed = cv2.imread('Test.png')

   img = cv2.imread(map_image)
   points = img.copy()

   #img = cv2.medianBlur(img,1)
   #img = cv2.GaussianBlur(img, (9,9), 0)
   #img = cv2.bilateralFilter(img, 5, 50, 50)
   _, threshold = cv2.threshold(img, 248, 255, cv2.THRESH_BINARY)

   #threshold = cv2.medianBlur(threshold, 5)
   kernel = np.ones((4,4),np.uint8)
   #threshold = cv2.morphologyEx(threshold,cv2.MORPH_OPEN,kernel, iterations = 1)
   threshold = cv2.dilate(threshold,kernel,iterations=1)
   threshold = cv2.bilateralFilter(threshold, 9, 75, 75)

   _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   #Create root node for xml file
   root = Element('map')

   for cnt in contours:
       approx = cv2.approxPolyDP(cnt, 0.005*cv2.arcLength(cnt, True), True)
       cv2.drawContours(points, [approx], 0, (0), 1)
       x = approx.ravel()[0]
       y = approx.ravel()[1]


       entry = SubElement(root, 'block')

       for ap in approx:
           ap_x = ap[0][0]
           ap_y = ap[0][1]
           cv2.circle(points, (ap_x, ap_y), 2, (0, 255, 0), -1)
           #add point as node in xml
           node = SubElement(entry, "node")
           node.set('x', str(ap_x))
           node.set('y', str(ap_y))


   #Save xml file
  # # output_file = open(f_number+'.xml', 'w')
  #  output_file.write('<?xml version="1.0"?>')
  #  output_file.write(prettify(root))
  #  output_file.close()

   #Save image with points in output directory
   cv2.imwrite("points", points)

   titles = ['Original', 'Trimmed', 'Map', 'Binary', 'Points']
   images = [original, trimmed, img, threshold, points]
   rows=2
   for i in range(len(images)):
       plt.subplot(rows, 3, i+1), plt.imshow(images[i], 'gray')
       plt.title(titles[i])
       plt.xticks([]), plt.yticks([])

plt.show()