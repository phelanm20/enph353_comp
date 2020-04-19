#! /usr/bin/env python

print("PICTURES WILL BE COLLECTED HEADS UP")
#Takes images of license plates and surrounding environment for training purposes

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

readRate = 5


path = os.path.dirname(os.path.realpath(__file__)) + "/"
img_dump = path + "images/fresh_images/"

class image_writer:
	
  timeSinceLastRead = 0
  timeOfLastRead = 0
  def __init__(self):
    print("img_init")
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.callback)

  def callback(self,data):
    time = int(rospy.get_time())
    image_writer.timeSinceLastRead = time - image_writer.timeOfLastRead
    if time > 20 and (time%readRate == 0 or image_writer.timeSinceLastRead > readRate) :
      image_writer.timeOfLastRead = time
      print("Collected image at time: " + str(time))
      try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
      except CvBridgeError as e:
          print(e)
      cv2.imwrite(img_dump + str(time) + ".png", cv_image)

def main(args):
  ic = image_writer()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()   

if __name__ == '__main__':
    main(sys.argv)

