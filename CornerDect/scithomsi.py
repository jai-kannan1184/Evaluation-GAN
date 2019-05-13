from __future__ import print_function
import cv2 as cv
import numpy as np
import  argparse
import random as rng



src_win = 'image'
maxTrackbar = 100
rng.seed(12345)
max_thresh = 255
thresh = 100

def featureTrack(val):
    maxCorners = max(val, 1)

    #parameters for Shi-Thomsi
    qualityLevel = 0.01
    minDst = 10
    blockSize = 3
    gradSize = 3
    useHarrisDect = False
    k = 0.04

    #copy source img
    copy = np.copy(src)

    #Apply Detection
    corners = cv.goodFeaturesToTrack(src_gray, maxCorners, qualityLevel, minDst, None, blockSize=blockSize, \
                                     gradientSize=gradSize, useHarrisDetector=useHarrisDect, k=k)
    #drawcrner detected
    print('** No. of corner detected:', corners.shape[0])
    radius = 4
    for i in range(corners.shape[0]):
        cv.circle(copy, (corners[i,0,0], corners[i,0,1]), radius, (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv.FILLED)

    #Draw the result
    cv.namedWindow(src_win)
    cv.imshow(src_win, copy)
def thresh_callback(val):
    threshold = val
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    _, contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1],3), dtype = np.uint8)
    for i in range(len(contours)):
        color = (255,250,255)
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
        cv.imshow('Contours',drawing)


#Load image and convert it to grayscale
parser = argparse.ArgumentParser(description='shi tomsi')
parser = argparse.ArgumentParser(description= 'Corners')
parser.add_argument('--input', help='Path to input img', default='Test.png')
args = parser.parse_args()

src = cv.imread(args.input)
if src is None:
    print('No Image in dir', args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray,(3 ,3))


#create a window and track bar
cv.namedWindow(src_win)
maxCorners = 23
cv.createTrackbar('Threshold:', src_win, maxCorners, maxTrackbar, featureTrack)
cv.createTrackbar('Threshold:', src_win, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.imshow(src_win,src)
featureTrack(maxCorners)

cv.waitKey()