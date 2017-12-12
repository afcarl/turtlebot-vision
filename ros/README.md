# ros packages

```
cd ~/catkin_ws/src/
ln -s ~/turtlebot-vision/ros/videocap #change to the right location 
ln -s ~/turtlebot-vision/ros/deploycnn #change to the right location

deploycnn
README.md
videocap

```

# Notes
```
SSID: Buffalo-A-D-380
Password: hidden
Robot-IP: hidden

#log into robot
ssh iurobotics-106@192.168.13.4

Turtle bot lab: https://docs.google.com/document/d/1xKH0H6Cj20W7mOZhRlCnm2I4gEBnA0ru5hJCvmMV6_g/edit

#start turtlebot in turtlebot
roslaunch turtlebot3_bringup turtlebot3_robot.launch

#start camera in turtle bot
roslaunch realsense_camera r200_nodelet_default.launch

#keyboard
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch

#run the vision based navitaiton
rosrun deploycnn deploycnn.py

https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032
```

# Acknowlegement
we use some codes from https://github.com/markwsilliman/turtlebot/tree/3863eb23f459f6fda65ce07cbcc19dd9c87fff20

