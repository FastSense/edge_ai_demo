# general
alias ra="ranger"
alias nv="nvim"

# ros
alias r_follow="roslaunch rosbot2 rosbot_slam_follower.launch"
alias r_slam="roslaunch rosbot2 rosbot_slam.launch"
alias r_navigation="roslaunch rosbot2 rosbot_navigation.launch"

alias cb="roscd && cd .. && catkin build"

default_fp="-fp ~/robot_configs/rosbot2/fplans/hw-tests/following_enable.txt"
default_cfg="~/robot_configs/rosbot2/ros-launch.json"
robot_name="rosbot2"
path_to_catkin_ws="~/catkin_ws/src"

alias launcher="cd ${path_to_catkin_ws}/fs_common/ros_launcher/scripts && ./ros_launcher.py -cfg ${default_cfg}"
alias robot-client="cd ${path_to_catkin_ws}/webio/src/ && ./robot-client.py -name ${robot_name}"
