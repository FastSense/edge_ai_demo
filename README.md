# Overview

![](https://habrastorage.org/webt/ig/yi/3f/igyi3fznkfbrz7gunyifdywrcxc.gif)

This demo shows the simultaneous processing of five neural networks running on hardware accelerators for processing 
input video streams from two cameras.    

Demo was launched on [**Fast Sense X**](https://fastsense.readthedocs.io/en/latest/) ([website](https://www.fastsense.tech/robotics_ai))
which is a powerfull x86 on-board computer with easy to plug in
edge ai accelerators. 
For this demo, five such devices are used:
  * two **Myriad X** devices;
  * three **Coral** devices.

Neural networks model creation on this devices are greatly simplified by [**nnio**](https://github.com/FastSense/nnio) python package which we are providing.  

We also are using **ROS** framework as a middleware for image transport and other message exchange, 
so you can easily get access to the models inference results from within ROS ecosystem.

For each video stream, image segmentation is performed using a 
[DeepLabV3](https://github.com/tensorflow/models/tree/master/research/deeplab), 
as well as object detection using a 
[SSD_Mobilenet_v2](https://aihub.cloud.google.com/p/products%2F79cd5d9c-e8f3-4883-bf59-31566fa99e49), 
and for each detected person, its identifier is determined using 
[ReID OsNet](https://github.com/KaiyangZhou/deep-person-reid).

![](https://habrastorage.org/webt/9_/yc/c5/9_ycc56st8dtywl52rg_xkcgbrk.png)

*Detection node*  runs SSD MobileNet inference for object detection and double OsNet for person reidentification.

*Segmentation node* runs DeepLabV3 for segmentation. This node runs in duplicate.

## Table of contents

<!-- vim-markdown-toc GitLab -->

* [Installation](#installation)
  * [Check your edge AI devices](#check-your-edge-ai-devices)
  * [Clone and install all dependencies](#clone-and-install-all-dependencies)
  * [Build your workspace](#build-your-workspace)
* [Quick start](#quick-start)
  * [Setting up cameras](#setting-up-cameras)
    * [RealSense setup](#realsense-setup)
    * [Other cameras setup](#other-cameras-setup)
  * [Run the demo](#run-the-demo)
* [Interface](#interface)
  * [Detection Node](#detection-node)
    * [Description](#description)
    * [Topics](#topics)
    * [Parameters](#parameters)
  * [Segmentation Node](#segmentation-node)
    * [Description](#description-1)
    * [Topics](#topics-1)
    * [Parameters](#parameters-1)

<!-- vim-markdown-toc -->

# Installation

## Check your edge AI devices

The presence of myriads in the system can be checked using the command:
```
lspci -d 1B73:1100
```

The presence of corals in the system can be checked using the command:
```
lspci -d 1AC1:089A
```

## Clone and install all dependencies

It is highly recommended to run everything inside a [docker container](./docker/README.md).

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

### Description

This node gets input video streams from 
several sources,  detects objects in them and
tries to reidentificate already detected humans using common database. 
For all incoming sources only one detection model is provided, which is shared between callback threads.
Each source is processed by its own reidentification model in parallel.

### Topics

| Topic                | I\O    | Message Type                                                                               | Description                                                                                                                 |
|----------------------|--------|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| /<in_img_topics[i]>  | Input  | ([sensor_msgs/Image](https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html)) | Topic with input image frame. Name of the topics spicified in `/<name>/in_img_topics`.                                      |
| /<out_img_topics[i]> | Output | ([sensor_msgs/Image](https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html)) | Topic where images are published with found objects marked on it. Name of the topics spicified in `/<name>/out_img_topics`. |

### Parameters

| Parameter name                | Type   | Default                                                    | Description                                                                                                                                                                                                                                                                      |
|-------------------------------|--------|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| in_img_topics                 | list   | [/camera1/color/image_raw, <br/> /camera2/color/image_raw] | Input image topics                                                                                                                                                                                                                                                               |
| out_img_topics                | list   | [detections_1, <br/> detections_2]                         | Output image topics                                                                                                                                                                                                                                                              |
| detection_inference_framework | string | EDGETPU                                                    | Framework for detection inference: *`"ONNX"`*, *`"OPENVINO"`* or *`"EDGE_TPU"`*.                                                                                                                                                                                                 |
| detection_inference_device    | string | TPU:0                                                      | Device name for SSD Mobilenet inference. Device for the inference. For OpenVINO: *`"CPU"`, `"GPU"` or `"MYRIAD"`*. <br/>EdgeTPU: *`"TPU:0"`* to use the first EdgeTPU device, *`"TPU:1"`* for the second etc...                                                                  |
| reid_inference_frameworks     | list   | [OPENVINO, OPENVINO]                                       | Frameworks for inference: *`"ONNX"`*, *`"OPENVINO"`* or *`"EDGE_TPU"`*.                                                                                                                                                                                                          |
| reid_inference_devices        | list   | [MYRIAD, MYRIAD]                                           | Device name for ReID inference. Device for the inference. For OpenVINO: *`"CPU"`, `"GPU"` or `"MYRIAD"`*. EdgeTPU: *`"TPU:0"`* to use the first EdgeTPU device, *`"TPU:1"`* for the second etc...                                                                                |
| reid_device_nums              | list   | [0, 0]                                                     | Value will be concated with `reid_inference_devices`. For example if the first device MYRIAD and the first num 0, result device name will be  "MYRIAD:0". <br /> Note that after using first device, second one become first, thats why [0, 0] will use first and second devices |
| threshold                     | float  | 0.25                                                       | Theshold for person reidentification. Higher value entails behavior in which, more likely two different persons will be considered as unique.                                                                                                                                    |
| inference_rate                | float  | 40.0                                                       | Rate for reidentification inference on each reid model, doubles up for detection node coz its only one                                                                                                                                                                           |



## Segmentation Node

### Description

This node subscribes to input Image topic from camera (Realsense D455 in demo).
Image segmentation is performed using DeepLabV3.
Segmented image is publishing into output image topic.

### Topics

| Topic                | I\O   | Message Type                                                                               | Description                                                                              |
|----------------------|-------|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| \<input_topic_name>  | Input | ([sensor_msgs/Image](https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html)) | Topic with input image frame. Name of the topic spicified in `/<name>/input_topic_name.` |
| \<output_topic_name> | Input | ([sensor_msgs/Image](https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html)) | Topic with input image frame. Name of the topic spicified in `/<name>/input_topic_name.` |

### Parameters

| Parameter name     | Type   | Default                  | Description                                                                                                                                                       |
|--------------------|--------|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_topic_name   | string | /camera1/color/image_raw | The name of the input topic.                                                                                                                                      |
| output_topic_name  | string | /segmentation            | The name of the output topic.                                                                                                                                     |
| inference_device   | string | TPU                      | Device for the inference. For OpenVINO: *`"CPU"`, `"GPU"` or `"MYRIAD"`*. EdgeTPU: *`"TPU:0"`* to use the first EdgeTPU device, *`"TPU:1"`* for the second etc... |
| max_inference_rate | float  | 10.0                     | Max Inference rate.                                                                                                                                               |
