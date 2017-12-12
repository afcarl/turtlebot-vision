from dataset import NavigationDataset

dataset = NavigationDataset(image_dirs=["../data/turtlebot1/capture_train1/","../data/turtlebot1/capture_train2/","../data/turtlebot1/capture_train3/","../data/turtlebot1/capture_train4/"])
print(len(dataset))
print(dataset[0])
print(dataset[1])
print(dataset[-1])

print(set([i[1] for i in dataset.all_data]))
# {-0.1,
#  -0.075000003,
#  -0.050000001,
#  -0.025,
#  0.0,
#  0.025,
#  0.050000001,
#  0.075000003,
#  0.1}