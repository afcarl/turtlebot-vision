#!/usr/bin/env python

'''
consluted for 
- https://github.com/markwsilliman/turtlebot/blob/master/goforward.py
- https://github.com/markwsilliman/turtlebot/blob/master/take_photo.py
'''
from __future__ import print_function
import rospy
from geometry_msgs.msg import Twist
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class GoForward():
    def __init__(self):

        # initiliaze nodes
        rospy.init_node('GoForward', anonymous=False)
        # tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")
        # What function to call when you ctrl + c    
        rospy.on_shutdown(self.shutdown)
        # Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=100)
        #TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        self.r = rospy.Rate(1);
        # Twist is a datatype for velocity
        self.move_cmd = Twist()


        #set up cameras
        self.bridge = CvBridge()
        self.image_received = False
        # Connect image topic
        img_topic = "/camera/color/image_raw"
        
        # Allow up to one second to connection
        rospy.sleep(1)

        self.frame=0

        self.image_sub = rospy.Subscriber(img_topic, Image, self.callback)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    def callback(self, data):

        # Convert image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # let's go forward at 0.2 m/s
        self.move_cmd.linear.x = 0.02
        # let's turn at 0 radians/s
        self.move_cmd.angular.z = 0

        #cv2.imwrite("%d.jpg"%self.frame, cv_image)
        # publish the velocity
        self.cmd_vel.publish(self.move_cmd)
        # # wait for 0.1 seconds (10 HZ)
        # self.r.sleep()
        self.frame+=1
                        
    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
        # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)
 
if __name__ == '__main__':
    try:
        GoForward()
    except:
        rospy.loginfo("GoForward node terminated.")