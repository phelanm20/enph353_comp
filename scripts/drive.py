#! /usr/bin/env python

#Drives the robot with PID

import sys
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
import math
from std_msgs.msg import String
import glob
import time

img_dump = "/home/fizzer/enph353_ws/src/competition/scripts/env_images/"
BW_img_dump = "/home/fizzer/enph353_ws/src/competition/scripts/BW_images/"
COM_img_dump = "/home/fizzer/enph353_ws/src/competition/scripts/COM_images/"
readRate = 20

#IMAGE/MAPPING
width = 640
height = 480
imageCentreY = height/2
imageCentreX = width/2
TY = 455
BY = 480
X1 = 100
X2 = width-X1
YCentre = (TY + BY)/2
XCentre = 320
roadRatioL = []
roadRatioR = []
bBound = height 
tBound = height/2

#MOTION/CORNERING
speed = 0.25
dT = 1.25
TDConvert = speed*dT*1.1
turnTime = 1.7
turnSpeed = 2.75
creepTime = 2.1
creepSpeed = 2
P_straight = 0.25
P_turn = 0.25
Tolerance = 15
AngTolerance = 0.1

#ANGLES
angleAdj = 25.0001
dPixdTheta = 0.00685
camHeight = 1

#LEFT TRACK
lBL = 0, height-30
lBR = 135, height
lTL = imageCentreX-5, imageCentreY
lTR = imageCentreX+5, imageCentreY   
  
#RIGHT TRACK
rBL = width-135, height
rBR = width, height-30
rTL = imageCentreX-5, imageCentreY
rTR = imageCentreX+5, imageCentreY   

#TRACKING
LY = height-10
UY = height-100
LYTheta = float(imageCentreY+angleAdj-LY)*dPixdTheta
UYTheta = float(imageCentreY+angleAdj-UY)*dPixdTheta
deltaTheta = LYTheta-UYTheta

class drive:
        nextDistL = 0
	nextDistR = 0
	timeSinceLastRead = 0
        timeOfLastRead = 0
        turnedRecently = False
	def __init__(self):
		print("drive_init")
                #populate arrays
                for y in range(0, bBound):      
                        roadRatioL.append(0.0)
                        roadRatioR.append(0.0)
		self.move_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.callback)
        


	def processImage(self, image, time):
	        print("\n")
	        print("Processing Image at time: " + str(time))
		grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                (thresh, BWImage) = cv.threshold(grayImage, 200, 255, cv.THRESH_BINARY)
                cv.imwrite(BW_img_dump + str(time) + ".png", BWImage) ##debug
		
		#initialization
		rightLineX = []
		leftLineX = []
		nextRightCorner = 0
		nextLeftCorner = 0
		for i in range(0, width):
			rightLineX.append(width)
			leftLineX.append(0)
		#RIGHT
		for y in range(LY, imageCentreY, -1):
			findLine = False
			rightLineX[y] = width
			for x in range(imageCentreX, width):
				pixR = int(BWImage[y,x])
				if pixR == 255:
					rightLineX[y] = x
					findLine = True
					break
			if findLine == False:
				rightLineX[y] = width
				nextRightCorner = y
				break
				
					
		#LEFT
		for y in range(LY, imageCentreY, -1):
			findLine = False 
			leftLineX[y] = 0
			for x in range(imageCentreX, 0, -1):
				pixL = int(BWImage[y,x])
				if pixL == 255:
					leftLineX[y] = x
					findLine = True
					break
			if findLine == False:
				leftLineX[y] = 0
				nextLeftCorner = y
				break

		UpperAv = (rightLineX[UY] + leftLineX[UY])/2
		LowerAv = (rightLineX[LY] + leftLineX[LY])/2
		roadAngle = float(math.atan(float(UpperAv-LowerAv)/(LY-UY)))
		driveAngle = float(math.atan(float(UpperAv-(LowerAv+imageCentreX)/2)/(LY-UY)))
		print("Road Angle: " + str(roadAngle))
		print("Drive Angle: " + str(driveAngle))
		cornerAngleR = float(nextRightCorner-imageCentreY+angleAdj)*dPixdTheta
		cornerAngleL = float(nextLeftCorner-imageCentreY+angleAdj)*dPixdTheta
		print("Right Corner Angle: " + str(cornerAngleR))
		print("Left Corner Angle: " + str(cornerAngleL))
 
                RDist = camHeight/(float(math.tan(cornerAngleR)))
                LDist = camHeight/(float(math.tan(cornerAngleL)))
		print("Right Corner Distance: " + str(RDist))
		print("Left Corner Distance: " + str(LDist))

                drive.nextDistR = RDist - speed*dT
                drive.nextDistL = LDist - speed*dT
		print("RDist Prediction: " + str(drive.nextDistR))
		print("LDist Prediction: " + str(drive.nextDistL))
		print("AngTolerance: " + str(AngTolerance))
		
		cv.line(image,(UpperAv,UY),(LowerAv,LY),(0,0,255))
    		cv.circle(image,(imageCentreX, LY), 3, (0,0,255), -1) ##debug
    		cv.line(image,(UpperAv,UY),((LowerAv+imageCentreX)/2,LY),(0,0,255))
    		cv.imwrite(COM_img_dump + str(time) + ".png", image) ##debug
    	        return driveAngle, LDist, RDist

  	def callback(self,data):
    		currentTime = int(10*rospy.get_time())
                drive.timeSinceLastRead = currentTime - drive.timeOfLastRead
    		if currentTime%readRate == 0 or drive.timeSinceLastRead > readRate:
    		        drive.timeOfLastRead = currentTime
    			try:
        			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    				driveAngle, LDist, RDist = self.processImage(cv_image, currentTime)
    				timeL = float(LDist)/TDConvert
    				timeR = float(RDist)/TDConvert
		  	    	move = Twist()
				#LEFT CORNER
    				if drive.nextDistL < 0 and drive.turnedRecently == False:
					print("Turning Left")
					print("Driving Forwards " + str((creepTime+timeL)*speed/creepSpeed) + " seconds")
					move.linear.x = creepSpeed
					self.move_pub.publish(move)
					rospy.sleep((creepTime+timeL)*speed/creepSpeed) #keep moving forward
					print("Turning for " + str(turnTime) + " seconds")
					move.linear.x = 0.0
					move.angular.z = turnSpeed + 0.5
					rospy.sleep(0.2)
					self.move_pub.publish(move)
					rospy.sleep(turnTime)
					move.angular.z = 0
					drive.turnedRecently = True
				#RIGHT CORNER
  	    			elif drive.nextDistR < 0 and drive.turnedRecently == False:
					print("Turning Right")
					print("Driving Forwards " + str((creepTime+timeR)*speed/creepSpeed) + " seconds")
					move.linear.x = creepSpeed
					self.move_pub.publish(move)
					rospy.sleep((creepTime+timeR)*speed/creepSpeed) #keep moving forward
					print("Turning for " + str(turnTime) + " seconds")
					move.linear.x = 0.0
					move.angular.z = -turnSpeed
					rospy.sleep(0.2)
					self.move_pub.publish(move)
					rospy.sleep(turnTime)
					move.angular.z = 0
					drive.turnedRecently = True
				#STRAIGHTAWAY
       	 			else:
					if abs(driveAngle) < AngTolerance:
						print("Goin' Straight")
						move.linear.x = speed
						move.angular.z = -(driveAngle)*P_straight 
						drive.turnedRecently = False
						print("X vel: " + str(speed))
						print("Angular vel: " + str(-driveAngle*P_straight)) 
		                        else:
						print("Stopped and Recentering...")
						print("Angular vel: " + str(-driveAngle*P_turn))
						move.angular.z = -(driveAngle)*P_turn
						move.linear.x = 0
				
				self.move_pub.publish(move)
      			except CvBridgeError as e:
	    			print(e)


def main(args):
  	driver = drive()
  	rospy.init_node('drive', anonymous=True)
  	try:
  		rospy.spin()
  	except KeyboardInterrupt:
    		print("Shutting down")
  	cv.destroyAllWindows()   

if __name__ == '__main__':
    main(sys.argv)
 
