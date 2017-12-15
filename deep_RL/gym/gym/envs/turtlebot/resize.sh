PATH=/N/u/chen478/gym/gym/envs/turtlebot/dataset
PATH_DES=/N/u/chen478/gym/gym/envs/turtlebot/dataset_small

for entry in $PATH/*
do
  IFS='/' read -r -a array <<< "$entry"
  echo ${entry}
  echo ${array[-1]}
  break
  for img in $entry/*
  do
    echo "$img"
    #convert $img -colorspace Gray destination.jpg
    #convert $img -resize 200x100 myfigure.jpg
  done
  break
done
