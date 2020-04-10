#! /usr/bin/env python

import cv2 as cv
import numpy as np

plate_img_directory = "/home/fizzer/enph353_ws/src/competition/scripts/plate_images/g.png"

img = cv.imread(plate_img_directory, 0) #Read as grayscale

# Thresholding the image
(thresh, img_bin) = cv.threshold(img, 128, 255,cv.THRESH_BINARY|cv.THRESH_OTSU)

cv.imshow("black and white", img_bin)


