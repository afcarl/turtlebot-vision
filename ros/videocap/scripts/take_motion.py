#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

class TakeMotion():
    def __init__(self):
    	self.cmd_vel=Twist()
        rospy.Subscriber('/cmd_vel', Twist, self.callback)
        # Allow up to one second to connection
        rospy.sleep(1)

    def callback(self,data):
        self.cmd_vel=data

    def take_motion(self):
        return {"vel":self.cmd_vel.linear.x,"ang":self.cmd_vel.angular.z}

if __name__ == '__main__':
    # Initialize
    rospy.init_node('take_motion', anonymous=False)

    
