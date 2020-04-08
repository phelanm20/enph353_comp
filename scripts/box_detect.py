#! /usr/bin/env python

import cv2 as cv
import numpy as np

plate_img_directory = "/home/maria/enph353_ws/src/competition/scripts/plate_images/g.png"

img = cv.imread(plate_img_directory, 0) #Read as grayscale
img = cv.bilateralFilter(img, 11, 17, 17)

kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 2)
kernel = np.ones((4,4),np.uint8)
dilation = cv.dilate(erosion,kernel,iterations = 2)
edged = cv.Canny(dilation, 30, 200)
cv.imshow("post magic formula", edged)
edges = cv.Canny(img, 200, 200) #Make high contrast image
cv.imshow("EDgED", edges)
#WHICH EDGES ONE WORKS BETTER?????
cv.waitKey(0)

_, contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #find contours

for cont in range(0, len(contours)):
    imcont = img
    cv.drawContours(imcont, contours, cont, (0,255,0), 3)
    label = str(cont)
    #cv.imshow(label, imcont)
    #cv.waitKey(0)


for cnt in range(0, len(contours)):
    bloop = contours[cnt]
    approx = cv.approxPolyDP(bloop, 0.01*cv.arcLength(bloop, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) == 4 and cv.contourArea(bloop) > 2000:
        #print("FOUND A RECTANGLE at: ")
        #print(cnt)

        rect = cv.boundingRect(bloop)

        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]

        plate = img[x:(x+w), y:(y+h)]  

