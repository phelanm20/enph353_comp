#! /usr/bin/env python

print("GOOD JOB PHELAN")
#Takes images of license plates and surrounding environment for training purposes

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

img_dump = "/home/maria/enph353_ws/src/competition/scripts/images/plate_detection_training/"

class image_writer:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.callback)

  def callback(self,data):
    time = int(10*rospy.get_time())

    if time%40 == 0:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imwrite(img_dump + str(time) + ".png", cv_image)
        print(str(time) + ": new image loaded")

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

