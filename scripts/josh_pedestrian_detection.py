#! /usr/bin/env python

import sys
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, backend
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import time
import os

path = os.path.dirname(os.path.realpath(__file__)) + "/"
plate_model_dir = path + 'plate_model.h5'
person_model_dir = path + 'person_model.h5'
dump_dir = path + 'images/fresh_images/'

class plate_reader:

    def __init__(self):
	#Load Model
        self.sess = tf.Session(target='', graph=None, config=None)
        self.graph = tf.get_default_graph()
        backend.set_session(self.sess)
        self.person_model = models.load_model(person_model_dir)

        #Initialize Variables
        self.counter = 0
        self.startTime = int(time.time())
        self.isCrosswalk = 0
	    
        #Subscribe to the robot's camera, crosswalk boolean
        self.bridge = CvBridge()
        self.pedestrian_pub = rospy.Publisher('pedestrian', Bool, queue_size=1)
        self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.picCallback)
        self.crosswalk_sub = rospy.Subscriber('crosswalk', Bool, self.crosswalkCheck)

  	#Read camera image, and display license plate text if detected
    def picCallback(self, image_sub):
        self.counter += 1

        #IF CROSSWALK, Check for pedestrian
        if self.isCrosswalk == True:
            try:
                feed_img = self.bridge.imgmsg_to_cv2(image_sub, "mono8")
            except CvBridgeError as e:
                print(e)

            self.pedestrian_pub.publish(self.isPedestrian(feed_img))
            self.isCrosswalk = 0

    #Callback if crosswalk is flagged
    def crosswalkCheck(self, crosswalk_sub) :
        self.isCrosswalk = crosswalk_sub

    #Takes feed image, returns true if a pedestrian is on crosswalk
    def isPedestrian(self, img) :
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img_aug = np.expand_dims(img, axis=0)
        with self.graph.as_default():
            backend.set_session(self.sess)
            y_predict = self.person_model.predict(img_aug)[0]
        isPerson = int(np.where(y_predict == np.amax(y_predict))[0])
        print("isPerson: " + str(isPerson))
        return isPerson

def main(args):
    reader = plate_reader()
    rospy.init_node('plate_reader', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()   

if __name__ == '__main__':
    main(sys.argv)
