#!/bin/bash

source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/catkin_ws
catkin config --blacklist $BLACKLIST
catkin build

source ~/scripts/aliases.sh
exec "/bin/bash"
