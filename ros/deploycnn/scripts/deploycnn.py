#!/usr/bin/env python

# part of the script is taken from https://github.com/markwsilliman/turtlebot/blob/3863eb23f459f6fda65ce07cbcc19dd9c87fff20/goforward_and_avoid_obstacle.py

import rospy
from geometry_msgs.msg import Twist
from take_photo import TakePhoto

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import os
import sys
import json

import numpy as np

from models import CNN1
import time

from PIL import Image

import argparse 

def parse_args():
    #Parses the arguments.
    parser = argparse.ArgumentParser(description="deploy")
    parser.add_argument('--speed',type=float,default=0.02,help='velocity. default is 0.02')
    parser.add_argument('--model',type=str,default="/home/stsutsui/catkin_ws/src/deploycnn/scripts/models/model_classify_equalsample",help='model path')
    parser.add_argument('--rate',type=int,default=10,help='save frequency. Maximum should be 30 due to camera frequency.')
    return parser.parse_args()

def shutdown():
    # stop turtlebot
    rospy.loginfo("Stop TurtleBot")
    # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
    cmd_vel.publish(Twist())
    # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
    rospy.sleep(1)

args = parse_args()

# initiliaze
rospy.init_node('GoForward', anonymous=False)
camera = TakePhoto()

# tell user how to stop TurtleBot
rospy.loginfo("To stop TurtleBot CTRL + C")

# What function to call when you ctrl + c    
rospy.on_shutdown(shutdown)

# Create a publisher which can "talk" to TurtleBot and tell it to move
# Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

#TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
r = rospy.Rate(args.rate);

# Twist is a datatype for velocity
move_cmd = Twist()
# let's go forward at 0.02 m/s
move_cmd.linear.x = args.speed
# let's turn at 0 radians/s
move_cmd.angular.z = 0

#
class_mapping = {0:0.0,1:0.1,2:-0.1}

#pytorch
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Scale(256),
    transforms.Lambda(lambda x : x.resize([224,224])),
    transforms.ToTensor(),
    normalize,
])

model = CNN1(output_dim=3,regression=False)
model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
model.eval()

rospy.loginfo("finish loading")

# as long as you haven't ctrl + c keeping doing...
while not rospy.is_shutdown():

    image = camera.image[::-1, :, ::-1].copy()
    image = Image.fromarray(np.uint8(image))
    image = transform(image)
    input_var = Variable(image[None, :, :], volatile=True)
    output = model(input_var)
    pred_class = np.argmax(output.data[0])
    ang_vel = class_mapping[pred_class]
    rospy.loginfo(ang_vel)
    move_cmd.angular.z = ang_vel

    # publish the velocity
    cmd_vel.publish(move_cmd)
    # wait for 0.1 seconds (10 HZ) and publish again
    r.sleep()