# Overview

This demo shows the simultaneous processing of five neural networks running on hardware accelerators for processing input video streams from two cameras. For each video stream, image segmentation is performed using a [DeepLabV3](https://github.com/tensorflow/models/tree/master/research/deeplab), as well as object detection using a [SSD_Mobilenet_v2](https://aihub.cloud.google.com/p/products%2F79cd5d9c-e8f3-4883-bf59-31566fa99e49), and for each detected person, its identifier is determined using [ReID OsNet](https://github.com/KaiyangZhou/deep-person-reid).

*Detection node*

*Segmentation node*

# Installation

## Check your edge AI devices

For this demo, 5 devices are used: three myriads and two corals.

The presence of myriads in the system can be checked using the command:
```
lspci -d 1B73:1100
```

The presence of corals in the system can be checked using the command:
```
lspci -d 1AC1:089A
```

## Clone and install all dependencies

It is highly recommended to run everything inside a docker container(*TODO LINK!!!*).

```
sudo apt install -y python-rosdep
cd <your_ws>/src
git clone --recursive https://github.com/FastSense/edge_ai_demo.git
cd ..
rosdep install --from-paths src --ignore-src -r -y
pip3 install nnio
```

## Build your workspace

```
cd <your_ws>
catkin_make
source <your_ws/>/devel/setup.bash
```

# Quick start

## Setting up cameras
### RealSense setup

Open `desktop_demo/launch/demo.launch`. Check the serial numbers of your cameras and write them down to the launch file:
```
<arg name="serial_no_camera1" default="XXXXXXXXXXXX"/>
<arg name="serial_no_camera2" default="XXXXXXXXXXXX"/>
```
### Other cameras setup

If you use any other cameras, you will need to independently launch within ROS and specify the names of the topics where they publish the images in the launch file `desktop_demo/launch/demo.launch`.

```
<arg name="camera1_topic" default="/set_your_topic_name_here"/>
<arg name="camera2_topic" default="/set_your_topic_name_here"/>
```

And then set flag `launch_realsense` to `false`:

```
<arg name="launch_realsense" default="false"/>
```

## Run the demo
All that's left now is to launch the launch file `demo.launch`:
```
roslaunch desktop_demo demo.launch
```

And then open [http://localhost:8888/](http://localhost:8888/) on your device. You will see the result of the networks in four frames, as in the overview.

# Interface

## Detection Node

### Parameters

### Subscribed topics

### Published topics

## Segmentation Node

### Parameters

### Subscribed topics

### Published topics