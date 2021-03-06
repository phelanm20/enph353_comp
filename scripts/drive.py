#! /usr/bin/env python

#Drives the robot

import sys
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
import math
from std_msgs.msg import String
import glob
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

path = os.path.dirname(os.path.realpath(__file__)) + "/"
BW_img_dump = path + 'BW_images/'
COM_img_dump = path + 'COM_images/'
readRate = 20

#IMAGE/MAPPING
width = 640
height = 480
imageCentreY = height/2
imageCentreX = width/2

#MOTION/CORNERING
speed = 0.25
dT = 1.25
TDConvert = speed*dT*1.1
turnTime = 1.7
turnSpeed = 2.75
turnAroundSpeed = 4.25
creepTime = 2.5
creepSpeed = 2.1
P_straight = 0.3
P_turnX = 0.5
P_turnAng = 0.35
XTolerance = 0.1
AngTolerance = 0.05
scaleFactor = 0.001

#ANGLES
angleAdj = 25.0001
dPixdTheta = 0.00685
camHeight = 1

#TRACKING
LY = height-20
UY = height-90

class drive:
        nextDistL = 0
	nextDistR = 0
	timeSinceLastRead = 0
        timeOfLastRead = 0
        turnedRecently = False
	Recentered = False
	crosswalkDetect = Bool()
	reachedTSection = False
	pedAlert = False
	def __init__(self):
		print("drive_init")
                #populate arrays
		self.move_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
		self.crosswalk_pub = rospy.Publisher('crosswalk', Bool, queue_size=1)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.callback)
		self.pedestrian_sub = rospy.Subscriber('pedestrian', Bool, queue_size=1)
		crosswalkDetect = False
    
	def processImage(self, image, time):
	        #print("\n")
	        #print("Processing Image at time: " + str(time))
		grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                (thresh, BWImage) = cv.threshold(grayImage, 200, 255, cv.THRESH_BINARY)
                #cv.imwrite(BW_img_dump + str(time) + ".png", BWImage) ##debug
		
		#initialization
		self.pedAlert = False
		rightLineX = []
		leftLineX = []
		nextRightCorner = UY
		nextLeftCorner = UY
		self.crosswalkDetect = False
		for i in range(0, width):
			rightLineX.append(width)
			leftLineX.append(0)
		#RIGHT
		for y in range(LY, UY-1, -1):
			findLine = False
			rightLineX[y] = width
			for x in range(imageCentreX, width):
				pixR = int(BWImage[y,x])
				if pixR == 255:
					rightLineX[y] = x
					findLine = True
					break
				RGBPix = image[y,x]
				compare = RGBPix == (0,0,255)
				if self.crosswalkDetect == False and compare.all():
					self.crosswalkDetect = True 
					#print("Detected Crosswalk!")
			if findLine == False:
				rightLineX[y] = width
				nextRightCorner = y
				#print("Found RCorner at y=" + str(nextRightCorner))
				break
				
					
		#LEFT
		for y in range(LY, UY-1, -1):
			findLine = False 
			leftLineX[y] = 0
			for x in range(imageCentreX, 0, -1):
				pixL = int(BWImage[y,x])
				if pixL == 255:
					leftLineX[y] = x
					findLine = True
					break
				RGBPix = image[y,x]
				compare = RGBPix == (0,0,255)
				if self.crosswalkDetect == False and compare.all():
					self.crosswalkDetect = True 
					print("Detected Crosswalk!")
			if findLine == False:
				leftLineX[y] = 0
				nextLeftCorner = y
				#print("Found LCorner at y=" + str(nextLeftCorner))
				break

		UpperAv = (rightLineX[UY] + leftLineX[UY])/2
		LowerAv = (rightLineX[LY] + leftLineX[LY])/2
		#print("UpperAv: " + str(UpperAv))
		#print("LowerAv: " + str(LowerAv))
		roadAngle = float(math.atan(float(UpperAv-LowerAv)/(LY-UY)))
		driveAngle = float(math.atan(float(UpperAv-(LowerAv+imageCentreX)*0.5)/(LY-UY)))
		#print("Road Angle: " + str(roadAngle))
		#print("Drive Angle: " + str(driveAngle))
		cornerAngleR = float(nextRightCorner-imageCentreY+angleAdj)*dPixdTheta
		cornerAngleL = float(nextLeftCorner-imageCentreY+angleAdj)*dPixdTheta
		#print("Right Corner Angle: " + str(cornerAngleR))
		#print("Left Corner Angle: " + str(cornerAngleL))
 
                RDist = camHeight/(float(math.tan(cornerAngleR)))
                LDist = camHeight/(float(math.tan(cornerAngleL)))
		#print("Right Corner Distance: " + str(RDist))
		#print("Left Corner Distance: " + str(LDist))

                self.nextDistR = RDist - speed*dT
                self.nextDistL = LDist - speed*dT
		#print("RDist Prediction: " + str(self.nextDistR))
		#print("LDist Prediction: " + str(self.nextDistL))
		#print("AngTolerance: " + str(AngTolerance))
		
			
		cv.line(image,(UpperAv,UY),(LowerAv,LY),(0,0,255))
    		cv.circle(image,(imageCentreX, LY), 3, (0,0,255), -1) ##debug
    		cv.line(image,(UpperAv,UY),((LowerAv+imageCentreX)/2,LY),(0,0,255))
    		#cv.imwrite(COM_img_dump + str(time) + ".png", image) ##debug
    		XOffset = LowerAv - imageCentreX
		
    	        return driveAngle, XOffset, LDist, RDist

  	def callback(self,data):
    		currentTime = int(10*rospy.get_time())
                self.timeSinceLastRead = currentTime - self.timeOfLastRead
    		if currentTime%readRate == 0 or self.timeSinceLastRead > readRate:
    		        drive.timeOfLastRead = currentTime
    			try:
        			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    				driveAngle, XOffset, LDist, RDist = self.processImage(cv_image, currentTime)			
    				Tolerance = 0
    				if self.crosswalkDetect == True:
    					error = XOffset*scaleFactor
    					Tolerance = XTolerance
					P_turn = P_turnX
    					
    				else:
    					error = driveAngle
    					Tolerance = AngTolerance
					P_turn = P_turnAng
    					
    				timeL = float(LDist)/TDConvert
    				timeR = float(RDist)/TDConvert
    				self.crosswalk_pub.publish(self.crosswalkDetect)
    				
				if self.pedestrian_sub == True:
					print("Pedestrian Alert!")
					self.pedAlert = True
		  	    	move = Twist()
				#T-SECTION
				if self.nextDistR < 0 and self.nextDistL < 0 and self.turnedRecently == False and self.reachedTSection == False:
					print("Tunring Around")
					print("Turning for " + str(turnTime) + " seconds")
					move.linear.x = -0.005
					move.angular.z = -turnAroundSpeed
					rospy.sleep(0.2)
					self.move_pub.publish(move)
					rospy.sleep(turnTime)
					move.angular.z = 0
					self.turnedRecently = True
					self.Recentered = False
					self.reachedTSection = True
				#RIGHT CORNER
  	    			elif self.nextDistR < 0 and self.turnedRecently == False:
					#print("Turning Right")
					#print("Driving Forwards " + str((creepTime+timeR)*speed/creepSpeed) + " seconds")
					move.linear.x = creepSpeed
					self.move_pub.publish(move)
					rospy.sleep((creepTime+timeR)*speed/creepSpeed) #keep moving forward
					#print("Turning for " + str(turnTime) + " seconds")
					move.linear.x = -0.005
					move.angular.z = -turnSpeed
					rospy.sleep(0.2)
					self.move_pub.publish(move)
					rospy.sleep(turnTime)
					move.angular.z = 0
					self.turnedRecently = True
					self.Recentered = False
				#LEFT CORNER
    				elif self.nextDistL < 0 and self.turnedRecently == False:
					#print("Turning Left")
					#print("Driving Forwards " + str((creepTime+timeL)*speed/creepSpeed) + " seconds")
					move.linear.x = creepSpeed
					self.move_pub.publish(move)
					rospy.sleep((creepTime+timeL)*speed/creepSpeed) #keep moving forward
					#print("Turning for " + str(turnTime) + " seconds")
					move.linear.x = -0.005
					move.angular.z = turnSpeed + 0.2
					rospy.sleep(0.2)
					self.move_pub.publish(move)
					rospy.sleep(turnTime)
					move.angular.z = 0
					self.turnedRecently = True
					self.Recentered = False
				#STRAIGHTAWAY
       	 			else:
					if abs(error) < Tolerance and self.Recentered == True:
						#print("Goin' Straight")
						#print("Ped Alert: " + str(self.pedAlert))
						if self.pedAlert == False:
							move.linear.x = speed
							move.angular.z = -(error)*P_straight 
						else:
							move.linear.x = 0
							move.angular.z = 0
						self.turnedRecently = False
						#print("X vel: " + str(speed))
						#print("Angular vel: " + str(-error*P_straight)) 
		                        else:
						#print("Stopped and Recentering...")			 
						#print("Angular vel: " + str(-error*P_turn))
						move.angular.z = -(error)*P_turn
						move.linear.x = -0.005
						self.Recentered = True
				
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
 
