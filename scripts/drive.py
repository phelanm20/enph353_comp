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
dT = 1.5
TDConvert = speed*dT*1.1
turnTime = 1.8
turnSpeed = 2.75
creepTime = 2
creepSpeed = 2
P_straight = 0.0045
P_turn = 0.0055
Tolerance = 3

#ANGLES
angleAdj = 25.0001
dPixdTheta = 0.00685
camHeight = 1

#LEFT TRACK
lBL = 0, height-30
lBR = 135, height
lTL = width/2-5, height/2
lTR = width/2+5, height/2   
  
#RIGHT TRACK
rBL = width-135, height
rBR = width, height-30
rTL = width/2-5, height/2
rTR = width/2+5, height/2   

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
		
		cv.line(image,(X1,BY),(X1,TY),(0,0,255))
		cv.line(image,(X2,BY),(X2,TY),(0,0,255))
		
                cv.line(image, lBL, lTL, (0,0,255))
		cv.line(image, lBR, lTR, (0,0,255))
		cv.line(image, rBL, rTL, (0,0,255))
		cv.line(image, rBR, rTR, (0,0,255))
		
                cv.imwrite(BW_img_dump + str(time) + ".png", BWImage) ##debug
                
                #line slopes
                slopeL1 = (float(lBL[1]-lTL[1])/float(lTL[0]-lBL[0]))
                slopeL2 = (float(lBR[1]-lTR[1])/float(lTR[0]-lBR[0]))
                slopeR1 = (float(rBL[1]-rTL[1])/float(rBL[0]-rTL[0]))
                slopeR2 = (float(rBR[1]-rTR[1])/float(rBR[0]-rTR[0]))
                
                #Left
                bBound = lBL[1]
                tBound = lTR[1]
                distToCornerL = 0
                LRatio = 0
                LDist = 0
                lBound = lBL[0]
                rBound = lBR[0]
                lBoundFloat = float(lBound)
                rBoundFloat = float(rBound)  
                for y in range(bBound-1,tBound,-1):
                        roadRatioL[y] = 0          
                        for x in range(lBound, rBound):
                                roadRatioL[y] += int(BWImage[y,x])/255
                        LRatio = float(roadRatioL[y])/float(rBound - lBound)
                        if LRatio < 0.001:
                                distToCornerL = height-y
                                break
                        lBoundFloat += slopeL1
                        rBoundFloat += slopeL2
                        lBound = int(round(lBoundFloat))
                        rBound = int(round(rBoundFloat))
                #print("pixDistL: " + str(distToCornerL))
                theta = float(bBound-distToCornerL-tBound+angleAdj)*dPixdTheta
                print("LTheta: " +str(theta))
                LDist = camHeight/(float(math.tan(theta)))
                print("LDist: " + str(LDist))
		print("NextDist: " + str(LDist-speed*dT))
		#print("Prediction error: " + str(drive.nextDistL - LDist))
                drive.nextDistL = LDist - speed*dT
                
                #Right
                bBound = rBR[1]
                tBound = rTL[1]
                distToCornerR = 0
                RRatio = 0
                RDist = 0
                lBound = rBL[0]
                rBound = rBR[0]
                lBoundFloat = float(lBound)
                rBoundFloat = float(rBound) 
                for y in range(bBound-1,tBound, -1):
                        roadRatioR[y] = 0
                        for x in range(lBound, rBound):
                                roadRatioR[y] += int(BWImage[y,x])/255
                        RRatio = float(roadRatioR[y])/float(rBound - lBound)
                        if RRatio < 0.001:
                                distToCornerR = height-y 
                                break
                        lBoundFloat -= slopeR1
                        rBoundFloat -= slopeR2
                        lBound = int(round(lBoundFloat))
                        rBound = int(round(rBoundFloat))
                theta = float(bBound-distToCornerR-tBound+angleAdj)*dPixdTheta
                print("RTheta: " + str(theta))
                RDist = camHeight/(float(math.tan(theta)))
                print("RDist: " + str(RDist))
		print("NextDist: " + str(RDist - speed*dT))
		#print("Prediction error: " + str(drive.nextDistR - RDist))
		drive.nextDistR = RDist - speed*dT
		
		#Line Follow COM
		XAv = 0
		YAv = 0
		roadX = 0
		roadY = 0
		numRoadPixels = 0
		for i in range(TY, BY):
			for j in range(X1, X2):
				pix = BWImage[i, j]
				if pix < 120:
					roadX += j
					roadY += i
					numRoadPixels += 1
		if numRoadPixels > 0:
			XAv = (int)(roadX/numRoadPixels)
			YAv = (int)(roadY/numRoadPixels)
			cv.circle(image,(XAv, YAv), 5, (0,0,255), -1) ##debug
		
    		cv.imwrite(COM_img_dump + str(time) + ".png", image) ##debug
    	        return XAv, YAv, LDist, RDist

  	def callback(self,data):
    		currentTime = int(10*rospy.get_time())
                drive.timeSinceLastRead = currentTime - drive.timeOfLastRead
    		if currentTime%readRate == 0 or drive.timeSinceLastRead > readRate:
    		        drive.timeOfLastRead = currentTime
    			try:
        			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    				XAv, YAv, LDist, RDist = self.processImage(cv_image, currentTime)
    				timeL = float(LDist)/TDConvert
    				timeR = float(RDist)/TDConvert
		  	    	move = Twist()
		    		XCentre = width/2
		    		XOffset = XAv-XCentre
		    		YOffset = YAv-YCentre
		    		#debug
		    		#print("XOffset: " + str(XOffset))
		    		#print("yOffset: " + str(YOffset))
		    		#print("Tolerance: " + str(Tolerance))
		    		print(drive.turnedRecently)
    				if drive.nextDistL < 0 and drive.turnedRecently == False:
					print("Turning Left")
					print("Driving Forwards " + str((creepTime+timeL)*speed/creepSpeed) + " seconds")
					move.linear.x = creepSpeed
					self.move_pub.publish(move)
					rospy.sleep((creepTime+timeL)*speed/creepSpeed) #keep moving forward
					print("Turning for " + str(turnTime) + " seconds")
					move.linear.x = 0.0
					move.angular.z = turnSpeed + 0.1
					rospy.sleep(0.2)
					self.move_pub.publish(move)
					rospy.sleep(turnTime)
					move.angular.z = 0
					drive.turnedRecently = True
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
       	 			else:
					print("Goin' Straight")
					print(XOffset)
					if abs(XOffset) < Tolerance:
					        move.linear.x = speed
					        move.angular.z = -(XOffset)*P_straight
					        drive.turnedRecently = False
                                        else:
					        move.angular.z = -(XOffset)*P_turn
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
 
