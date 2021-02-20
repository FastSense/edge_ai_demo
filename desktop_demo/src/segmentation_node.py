#!/usr/bin/env python3
"""Segmetation node for ROS AI Demo.

This node subscribes to input Image topic from camera (Realsense D455 in demo).
Image segmentation is performed using DeepLabV3.
Segmented image is publishing into output image topic.
"""

import sys

import cv2

import nnio

import numpy as np

import rospy

from sensor_msgs.msg import Image


class SegmentationNode:
    """Segmentation node.

    TODO desription.

    Attributes:
        _name (str): name of the node.
        _input_image (:obj:`numpy.ndarray`): current input cv2 image.
        _input_image_raw (:obj:`sensor_msgs.msg.Image`): current input image as Ros message.
        _model (:obj:`nnio.model.Model`): segmentation neural net model.
        _timer (:obj:`rospy.Timer`): ROS timer that calls the `inference()` method with
                                     `~max_inference_rate` rate.
        _image_pub (:obj:`rospy.Publisher`): ROS Image publisher for the image after segmentation.
        _image_sub (:obj:`rospy.Subscriber`): ROS Image subscriber for the input video stream.

    """

    def __init__(self, name='segmentation_node'):
        """Create instance of `SegmentationNode` class.

        Args:
            name (str): name of the node.

        """
        self._name = name

        self.accept_params()

        rospy.init_node(name)

        self._input_image = None
        self._input_image_raw = None

        self._init_pub_sub()

        self._model = nnio.zoo.edgetpu.segmentation.DeepLabV3(device=self._inference_device)

        self._timer = rospy.Timer(rospy.Duration(1.0 / self._max_inference_rate),
                                  self.inference)

    def accept_params(self):
        """Read params from ROS parameter server.

        `/<name>/input_topic_name (string)`: the name of the input topic.
        `/<name>/output_topic_name (string)`: the name of the output topic.
        `/<name>/inference_device (string)`: the name of the inference device
                                             ("TPU:0", "TPU:1" etc...).
        `/<name>/max_inference_rate (float)`: max rate of the inference (Hz).

        """
        self._input_topic_name = rospy.get_param('/%s/input_topic_name' % self._name,
                                                 '/camera/color/image_raw')
        self._output_topic_name = rospy.get_param('/%s/output_topic_name' % self._name,
                                                  '/segmentation')
        self._inference_device = rospy.get_param('/%s/inference_device' % self._name, 'TPU:1')
        self._max_inference_rate = rospy.get_param('/%s/max_inference_rate' % self._name, 10)

    def _init_pub_sub(self):
        """Init ROS Publisher and ROS subsriber for the image topics."""
        self._image_pub = rospy.Publisher(self._output_topic_name, Image, queue_size=1)

        self._input_image_sub = rospy.Subscriber(self._input_topic_name, Image,
                                                 self._input_image_cb)

    def _input_image_cb(self, msg):
        """Handle callback for the input image topic.

        It only saves the input Image message to buffer? whic is used in the `inference` method.

        """
        self._input_image_raw = msg

    def _postprocess(self, result):
        """Make postrocess of the inference result.

        It applies color map to the inference result and overlays it on the original image.
        """
        im = (self._model.labels - result.astype('uint8')) * (255 // self._model.labels)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        imc = cv2.applyColorMap(im, cv2.COLORMAP_JET)

        output_image = cv2.resize(self._input_image, (513, 513))
        output_image = cv2.addWeighted(output_image[:, :, ::-1], 0.5, imc, 0.5, 0)
        output_image = cv2.resize((self._input_image_raw.width,
                                   self._input_image_raw.height))
        return output_image

    def _msg_to_nparray(self, msg):
        """Convert ROS Image message to numpy array."""
        img = np.frombuffer(msg.data, dtype='uint8')
        img = img.reshape((self._input_image_raw.height,
                           self._input_image_raw.width, 3))
        return img

    def _publish_img(self, img, width, height, encoding='rgb8'):
        data = img.tobytes()
        output = Image(width=width, height=height, data=data,
                       encoding=encoding, step=len(data) // height)

        self._image_pub.publish(output)

    def inference(self, **kwargs):
        """Run inference of the segmentation net."""
        if self._input_image_raw is not None:
            self._input_image = self._msg_to_nparray(self._input_image_raw)

            result = self._model(self._model.get_preprocessing()(self._input_image))

            output_image = self._postprocess(result)

            self._publish_img(output_image,
                              self._input_image_raw.width,
                              self._input_image_raw.height)


if __name__ == '__main__':
    node = SegmentationNode(sys.argv[1].replace('__name:=', ''))
    rospy.spin()
