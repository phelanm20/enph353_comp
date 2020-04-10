#! /usr/bin/env python

print("GOOD JOB PHELAN")
#Takes images of license plates and surrounding environment for training purposes

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

img_dump = "/home/fizzer/enph353_ws/src/competition/scripts/env_images/"
readRate = 20
class image_writer:
	
  timeSinceLastRead = 0
  timeOfLastRead = 0
  def __init__(self):
    print("img_init")
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.callback)

  def callback(self,data):
    time = int(10*rospy.get_time())
    image_writer.timeSinceLastRead = time - image_writer.timeOfLastRead
    if time%readRate == 0 or image_writer.timeSinceLastRead > readRate:
	image_writer.timeOfLastRead = time
	print("Collected image at time: " + str(time))
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
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

