#! /usr/bin/env python

#Takes an image of surroundings and prints PLATE if there is a pair of plates in sight

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv


class plate_reader:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.readFeed)

  #Read camera image
  #Return grayscale cv2 image
  def readFeed(self, image_sub):
      try:
          feed_img = self.bridge.imgmsg_to_cv2(data, "mono8")
      except CvBridgeError as e:
          print(e)
      if isPlate(feed_img):
          print("PLATE DETECTED:")
          #print(plateText)

  def isPlate(img): 
      bool = neuralNEEEET(image) #Obviously, replace neuralneet with something that works
      return bool

def main(args):
  reader = plate_reader()
  rospy.init_node('plate_reader', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()   

if __name__ == '__main__':
    main(sys.argv)

