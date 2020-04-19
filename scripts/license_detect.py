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
car_model_dir = '/home/maria/enph353_ws/src/car_model.h5'
person_model_dir = '/home/maria/enph353_ws/src/person_model.h5'
dump_dir = path + 'images/fresh_images/'

class plate_reader:

  def __init__(self):
    #Load Model
    self.sess = tf.Session(target='', graph=None, config=None)
    self.graph = tf.get_default_graph()
    backend.set_session(self.sess)
    self.car_model = models.load_model(car_model_dir)
    self.person_model = models.load_model(person_model_dir)

    #Initialize Variables
    self.counter = 0
    self.plateDict = {}
    self.keyList = []
    self.startTime = int(time.time())
    self.isCrosswalk = 0
    
    #Subscribe to the robot's camera, crosswalk boolean
    self.bridge = CvBridge()
    self.pedestrian_pub = rospy.Publisher('pedestrian', Bool, queue_size=1)
    self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.picCallback)
    self.crosswalk_sub = rospy.Subscriber('crosswalk', Bool, self.crosswalkCheck)

  #Read camera image, and display license plate text if detected
  def picCallback(self, image_sub):
    #IF TIME IS UP
    if int(time.time()) - self.startTime > 230 :
      print("FOUR MINUTES" + "\n\nPLATES READ:\n")
      sorted(self.keyList)
      
      for key in self.keyList :
        print("P" + str(key) + "\t" + str(self.plateDict[key]))

      while True:
        pass

    self.counter += 1

    #IF CROSSWALK, Check for pedestrian
    if self.isCrosswalk :
      try:
          feed_img = self.bridge.imgmsg_to_cv2(image_sub, "mono8")
      except CvBridgeError as e:
          print(e)

      self.pedestrian_pub.publish(self.isPedestrian(feed_img))
      self.isCrosswalk = 0

    #Every SIXTH IMAGE check for a plate, and read if present    
    if self.counter % 6 == 0 :
      try:
          feed_img = self.bridge.imgmsg_to_cv2(image_sub, "mono8")
      except CvBridgeError as e:
          print(e)

      if self.isPlate(feed_img) :
        realPlates = self.chop(feed_img)

        #Add the plates read to the plate dictionary
        if len(realPlates) == 2 :
          newKey = self.get_plate_val(1, realPlates[0])
          newVal = self.get_plate_val(2, realPlates[1])

          for char in "0123456789" :
            if char in newVal[:2] :
              return
          if newKey in self.plateDict:
            if 'B' in self.plateDict[newKey] and 'B'  not in newVal :
              self.plateDict[newKey] = newVal
            if 'U' in self.plateDict[newKey] and 'U'  not in newVal and 'L' in newVal :
              self.plateDict[newKey] = newVal
          else :
            self.plateDict[newKey] = newVal
            self.keyList.append(newKey)

  #Callback if crosswalk is flagged
  def crosswalkCheck(self, Bool) :
    self.isCrosswalk = Bool
    if self.isCrosswalk :
      print("CROSSWALK DETECTED!")

  #Takes feed image, returns true if a pedestrian is on crosswalk
  def isPedestrian(self, img) :
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img_aug = np.expand_dims(img, axis=0)
    with self.graph.as_default():
      backend.set_session(self.sess)
      y_predict = self.person_model.predict(img_aug)[0]
    isPerson = int(np.where(y_predict == np.amax(y_predict))[0])
    return isPerson

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
          platesSeen.append(plate)

    if len(platesSeen) > 1 :     
      platesSeen[0], platesSeen[1] = platesSeen[1], platesSeen[0]

    return platesSeen

  #Takes image of license plate, reads it, displays text
  def get_plate_val(self, num, plate):
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
    #print("\t\tcharacters found:" + characters)

    if num == 1 :
      if 'B' in characters or 'I' in characters or 'S' in characters or 'Z' in characters:
        characters = characters.replace('B', '8')
        characters = characters.replace('I', '1')
        characters = characters.replace('S', '5')  
        characters = characters.replace('Z', '2')
      
      characters = characters[1:]

      print("plate read as: " + characters)
      cv.imwrite(dump_dir + characters + ".png", plate)
      return characters

    if num == 2 :
      if numchar == 6 :
        characters = characters[1:]
        numchar = numchar - 1

      if numchar == 5 :
        characters = characters[0] + characters[2:]
        numchar = numchar - 1

      if numchar == 4 :
        correctChar = characters
        correctNum = characters
        if 'B' in characters[2:] or 'I' in characters[2:] or 'S' in characters[2:] or 'Z' in characters[2:]: 
          characters = characters.replace('B', '8')
          characters = characters.replace('I', '1')
          characters = characters.replace('S', '5')
          characters = characters.replace('Z', '2')
          correctNum = characters
        if '8' in characters[:2] or '1' in characters[:2] or '5' in characters[:2] or '2' in characters[:2] or '9' in characters[:2]: 
          characters = characters.replace('8', 'B')
          characters = characters.replace('1', 'I')
          characters = characters.replace('5', 'S')
          characters = characters.replace('2', 'Z')
          characters = characters.replace('9', 'Q')
          correctChar = characters
        characters = correctChar[:2] + correctNum[2:]

    print("plate read as: " + characters)
    cv.imwrite(dump_dir + characters + ".png", plate)
    return characters


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
