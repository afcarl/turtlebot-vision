#!/usr/bin/env python

import rospy
from take_photo import TakePhoto
from take_motion import TakeMotion 
import time
import os
import argparse

def parse_args():
    #Parses the arguments.
    parser = argparse.ArgumentParser(description="capture")
    parser.add_argument('--save',type=str,default="~/capture",help='save dir. use absolute path!')
    parser.add_argument('--rate',type=int,default=30,help='save frequency. Maximum should be 30 due to camera frequency.')
    return parser.parse_args()



if __name__ == '__main__':
    #get argument
    args = parse_args()
    if os.path.isdir(args.save):
        print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort."%args.save)
        time.sleep(5)
        os.system('rm -rf %s/'%args.save)
    else:
        os.makedirs(args.save)
        print("made the save directory",args.save)

    #initialize
    rospy.init_node('capture', anonymous=False)
    motion = TakeMotion()
    camera = TakePhoto()
    r = rospy.Rate(args.rate);#default rate is 30Hz

    print("rate:%d Hz"%args.rate)

    raw_input("Press Enter to continue...")

    i=0
    while not rospy.is_shutdown():
        current_motion=motion.take_motion()
        # print(current_motion)
        vel=current_motion["vel"]
        ang=current_motion["ang"]

        img_title = time.strftime("%d-vel%0.3f-ang%0.3f-"%(i,vel,ang)+"%Y.%m.%d.%H.%M.%S.png")
        if camera.take_picture(os.path.join(args.save,img_title)):
            rospy.loginfo("Saved image " + img_title)
        else:
            rospy.loginfo("No images received")

        i+=1
        r.sleep()