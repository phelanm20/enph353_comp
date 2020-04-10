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

plate_dir = "/home/maria/enph353_ws/src/competition/scripts/images/fresh_images/"

class plate_reader:

  def __init__(self):
    #Load Model
    self.sess = tf.Session(target='', graph=None, config=None)
    self.graph = tf.get_default_graph()
    backend.set_session(self.sess)
    self.new_model = models.load_model('/home/maria/enph353_ws/src/competition/scripts/car_model.h5')
    self.gumbo = 1
    
    #Load Image
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.getSpaceCar)

  #Read camera image
  #Find number of plate pairs
  def getSpaceCar(self, image_sub):
    if time.time() % 15.0 < .5: #PLEAASEE GET RID OF THIS should be called 
      try:
          feed_img = self.bridge.imgmsg_to_cv2(image_sub, "mono8")
      except CvBridgeError as e:
          print(e)

      numpla = self.howManyPlates(feed_img)
      print("PLATES DETECTED: " + str(2*numpla)) #could also pring number of plate pairs if needed
      if numpla > 0 :
        self.chop(feed_img)

  def chop(self, img):
    img = cv.bilateralFilter(img, 11, 17, 17) #high contrast image
    edges = cv.Canny(img, 200, 200)
    _, contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #find contours

    for cnt in range(0, len(contours)):
      contour = contours[cnt]
      approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
      if len(approx) == 4 and cv.contourArea(contour) > 2000: #MORE IF STATEMENTS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          rect = cv.boundingRect(contour)
          x = rect[0]
          y = rect[1]
          w = rect[2]
          h = rect[3]
          plate = img[y:(y+h), x:(x+w)]
          cv.imwrite(plate_dir + str(self.gumbo) + "_" + str(cnt) + ".png", plate)

    self.gumbo = self.gumbo + 1

  #Takes feed image, returns number of plate pairs in image
  def howManyPlates(self, img):
    img = img.reshape(img.shape[0], img.shape[1], 1) #ignore this error
    img = np.swapaxes(img, 0, 1)
    img_aug = np.expand_dims(img, axis=0)
    with self.graph.as_default():
      backend.set_session(self.sess)
      y_predict = self.new_model.predict(img_aug)[0]
    numPlates = int(np.where(y_predict == np.amax(y_predict))[0])
    return numPlates

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
