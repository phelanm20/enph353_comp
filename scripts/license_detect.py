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
import os

path = os.path.dirname(os.path.realpath(__file__)) + "/"
plate_model_dir = path + 'plate_model.h5'
car_model_dir = 'home/maria/enph353_ws/src/car_model.h5'
dump_dir = path + 'images/fresh_images/'

class plate_reader:

  def __init__(self):
    #Load Model
    self.sess = tf.Session(target='', graph=None, config=None)
    self.graph = tf.get_default_graph()
    backend.set_session(self.sess)
    self.car_model = models.load_model(car_model_dir)
    self.counter = 0
    
    #Load Image
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.picCallback)

  #Read camera image, and display license plate text if detected
  def picCallback(self, image_sub):
    self.counter += 1

    if self.counter % 6 == 0 :
      try:
          feed_img = self.bridge.imgmsg_to_cv2(image_sub, "mono8")
      except CvBridgeError as e:
          print(e)

      if self.isPlate(feed_img) :
        realPlates = self.chop(feed_img)
        print("\nPlates found: " + str(len(realPlates)))

        if len(realPlates) == 0 :
          print("\tFALSE DETECTION")
        else :
          for realPlate in realPlates :
            self.show_plate_val(realPlate)

  #Takes feed image, returns true if readable license plate in image
  def isPlate(self, img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img_aug = np.expand_dims(img, axis=0)
    with self.graph.as_default():
      backend.set_session(self.sess)
      y_predict = self.car_model.predict(img_aug)[0]
    numPlates = int(np.where(y_predict == np.amax(y_predict))[0])
    return numPlates

  #If license plate in image, crops image, returns true, cropped image
  #If not, returns false, original environment image
  def chop(self, img):
    img = cv.bilateralFilter(img, 11, 17, 17) #high contrast image
    edges = cv.Canny(img, 200, 200)
    _, contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #find contours

    platesSeen = []
    for contour in contours:
      approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
      if len(approx) == 4 and cv.contourArea(contour) > 2000: #if certain sized rectangle #MORE IF STATEMENTS HERE!!!!!!
          x, y, w, h = cv.boundingRect(contour)
          plate = img[y:(y+h), x:(x+w)]
          cv.imwrite(dump_dir + str(y) + '.png', plate)
          platesSeen.append(plate)

    if len(platesSeen) > 1 :     
      platesSeen[0], platesSeen[1] = platesSeen[1], platesSeen[0]

    return platesSeen

  #Takes image of license plate, reads it, displays text
  def show_plate_val(self, plate):
    print("\tReading a plate")

    characters = ""

    #Load Model
    sess = tf.Session(target='', graph=None, config=None)
    graph = tf.get_default_graph()
    backend.set_session(sess)
    plate_model = models.load_model(plate_model_dir)

    #Pre Process Image
    delta = 1
    height, width = plate.shape
    plate = plate[delta:height-delta, delta: width-delta]   #Remove border
    ret, thresh = cv.threshold(plate, 0, 255, cv.THRESH_OTSU) #Threshold
    _, ctrs, _ = cv.findContours(thresh.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) #find contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv.boundingRect(ctr)[0])

    #Identify and display characters
    for i, ctr in enumerate(sorted_ctrs):
      x, y, w, h = cv.boundingRect(ctr)
      area = w*h
      # cv.circle(thresh, (x, y), 4, 0)
      # cv.imshow("contour " + str(i) + " bounding rect, showing top left", thresh)
      # cv.waitKey(0)

      if area < 1500 and area > 200:
        charThresh = thresh[y:y + h, x:x + w] #Isolate each character
        # cv.imshow("area passed check:" + str(area), charThresh)
        # cv.waitKey(0)
        dim = (14, 20)
        char = cv.resize(charThresh, dim, interpolation = cv.INTER_AREA)
        char = char.reshape(20, 14, 1)
        # cv.imshow("aafter reshaping", char)
        # cv.waitKey(0)
        img_aug = np.expand_dims(char, axis=0)
        with graph.as_default():
          backend.set_session(sess)
          y_predict = plate_model.predict(img_aug)[0] #Put through CNN
          spot = int(np.where(y_predict == np.amax(y_predict))[0])
          if spot < 10:
            name = str(spot)
          else:
            name = chr(spot + 55)
          
          characters += name

    numchar = len(characters)  
    print("\t\tcharacters found:" + characters)
    if numchar == 2 or numchar == 3:
      if 'B' in characters or 'I' in characters or 'S' in characters or 'Z' in characters:
        characters = characters.replace('B', '8')
        characters = characters.replace('I', '1')
        characters = characters.replace('S', '5')  
        characters = characters.replace('Z', '2')
    elif numchar == 4:
      correctChar = characters
      correctNum = characters
      if 'B' in characters[2:] or 'I' in characters[2:] or 'S' in characters[2:] or 'Z' in characters[2:]: 
        characters = characters.replace('B', '8')
        characters = characters.replace('I', '1')
        characters = characters.replace('S', '5')
        characters = characters.replace('Z', '2')
        correctNum = characters
      if '8' in characters[:2] or '1' in characters[:2] or '5' in characters[:2] or '2' in characters[:2]: 
        characters = characters.replace('8', 'B')
        characters = characters.replace('1', 'I')
        characters = characters.replace('5', 'S')
        characters = characters.replace('2', 'Z')
        correctChar = characters
      characters = correctChar[:2] + correctNum[2:]
    else : 
      print("\tFALSE DETECTION")

    print("\t\tPlate read as:\t" + characters) 


def main(args):
  reader = plate_reader()
  #Debug plate mode
  # imgimg = cv.imread(dump_dir + '340.png', 0)
  # reader.show_plate_val(imgimg)
  #Run Sim Mode
  rospy.init_node('plate_reader', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv.destroyAllWindows()   

if __name__ == '__main__':
    main(sys.argv)
