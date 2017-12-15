import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import cv2
import random

import logging
logger = logging.getLogger(__name__)
FRAME_DIR = "/N/u/chen478/gym/gym/envs/turtlebot/dataset"
train_set = ["capture_train1", "capture_train2", "capture_train3", "capture_train4"]
test_set = ["capture_test"]
#frame_number = 0
class TurtlebotEnv(gym.Env, utils.EzPickle):

    def __init__(self):
        # Load all the frames
        self.frame_list={}
        self.frame_number=0
        offset=[0, 1302, 2628, 4592]
        i=0
        for partition in train_set:
          paths=os.path.join(FRAME_DIR, partition)
          base=offset[i]
          for x in os.listdir(paths):
            ID=str(int(x.split('-')[0]) + base)
            path=os.path.join(paths, x)
            angle=float(x.split('-ang')[1].split('-2017')[0]) * 10
            self.frame_list[ID]=[path, angle]
          i += 1
 
        #self._action_set=[-0.025, -0.05, -0.1, 0, 0.025, 0.05, 0.1]
        self._action_set=[-0.25, -0.5, -1, 0, 0.25, 0.5, 1]

    def _step(self, a):
        true_action_last_time = self.frame_list[str(self.frame_number)][1]
        reward = -abs(a-true_action_last_time)
   
        tmp=[abs(x - a) for x in self._action_set]
        index=np.argmin(tmp)
        action = self._action_set[index]
        #print("predicted: " + str(action) + ", true: " +str(true_action_last_time))
        #print(true_action_last_time)
        #print('************************')

     
        #while(action != float(self.frame_list[str(self.frame_number)][1])):
        #  if(self.frame_number <= 6500):
        #    self.frame_number += 1 
        #  else:
        #    self.frame_number = 0       
        terminal=False
        if self.frame_number==1301 or self.frame_number==2627 or self.frame_number==4591 or self.frame_number==6501:
          terminal=True

        self.frame_number += 1
        if true_action_last_time != 0 and random.uniform(0, 1)>0.5:
          self.frame_number -= 1
          
        #if true_action_last_time == 0 and random.uniform(0, 1)>0.5:
        #  self.frame_number += 1

        if self.frame_number > 6501: 
          self.frame_number = 0
        ob = cv2.imread(self.frame_list[str(self.frame_number)][0],0)

        return ob, reward, terminal, {}


    def _reset(self):
        self.frame_number = 0
        ob = cv2.imread(self.frame_list[str(self.frame_number)][0],0)
        return ob
        



