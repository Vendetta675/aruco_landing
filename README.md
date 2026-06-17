# aruco_landing
Aruco Marker Detection and Landing which switches to expanding spiral search when the marker is lost from frame . It converts the camera coordinate to body FLU frame ,which later converts to ENU frame ,used for ROS.
Clone the repository in your ros2 workspace and build it using :
### colcon build 
source the installed dependencies using
### source install/setup.bash
Run the node using the command :
### ros2 run takeoff auto_takeoff
#Dependencies used :
opencv
mavros_msgs
geometry_msgs
rclpy
### Prerequisites
Ardupilot SITL
Gazebo Harmonic
ROS_GZ_BRIDGE
MAVROS
Note : Include the models and worlds file in your ardupilot_gazebo file which you can clone from this repository
### git clone https://github.com/ArduPilot/ardupilot_gazebo.git

