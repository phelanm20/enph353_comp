#! /usr/bin/env python

import sys
import rospy
from sensor_msgs.msg import Image
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, backend
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import time
from os import listdir
from os.path import isfile, join

test_dir = "/home/maria/enph353_ws/src/competition/scripts/images/plate_training/"
paths = [ join(test_dir, f) for f in listdir(test_dir) if isfile(join(test_dir, f)) ]

#Load Model
sess = tf.Session(target='', graph=None, config=None)
graph = tf.get_default_graph()
backend.set_session(sess)
new_model = models.load_model('/home/maria/enph353_ws/src/competition/scripts/plate_model.h5')

for n in range (0, len(listdir(test_dir)) - 22) :
    if n%3 == 0 :
        plate = cv.imread(paths[n], 0)

        delta = 5
        width, height = plate.shape
        plate = plate[delta:width-delta, delta: height-delta]   #Remove border
        ret, thresh = cv.threshold(plate, 0, 255, cv.THRESH_OTSU) #Threshold
        _, ctrs, _ = cv.findContours(thresh.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) #find contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv.boundingRect(ctr)[0])

        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv.boundingRect(ctr)
            area = w*h
            if area < 1200 and area > 200:
                charThresh = thresh[y:y + h, x:x + w] #Isolate each character

                dim = (40, 27)
                char = cv.resize(charThresh, dim, interpolation = cv.INTER_AREA)
                char = char.reshape(char.shape[0], char.shape[1], 1) #ignore this error
                char = np.swapaxes(char, 0, 1)
                img_aug = np.expand_dims(char, axis=0)
                with graph.as_default():
                    backend.set_session(sess)
                    y_predict = new_model.predict(img_aug)[0] #Put through CNN
                    spot = int(np.where(y_predict == np.amax(y_predict))[0])
                    if spot < 10:
                      name = str(spot)
                    else:
                      name = chr(spot + 55)
                
                cv.imshow(name, charThresh)
                cv.waitKey(0)
