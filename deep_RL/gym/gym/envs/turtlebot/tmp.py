import numpy as np
import os
import cv2
FRAME_DIR = "/N/u/chen478/gym/gym/envs/turtlebot/dataset"
train_set = ["capture_train1", "capture_train2", "capture_train3", "capture_train4"]
test_set = ["capture_test"]

frame_list={}
#for partition in test_set:
i=0
for partition in train_set:
  paths=os.path.join(FRAME_DIR, partition)
  for x in os.listdir(paths):
    ID=str(i)
    #ID=x.split('-')[0]
    path=os.path.join(paths, x)
    angle=float(x.split('-ang')[1].split('-2017')[0]) * 10
  #tmp_list=[os.path.join(paths, x) for x in os.listdir(paths)]
    frame_list[ID]=[path, angle]
    i=i+1

print(frame_list['6500'])
print(i)
#ob = cv2.imread(frame_list[str(3)][0],0)
#print(ob.shape)
#for log in frame_list:
  #print(log.split('/')[10].split('-')[0])
  #print(log.split('/')[10].split('-ang')[1].split('-2017')[0])
