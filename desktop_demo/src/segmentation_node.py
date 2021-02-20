#!/usr/bin/env python3
"""TODO description: segmentation node."""

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
        pass

    Methods:
        pass

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
        """Read params from ROS parameter server."""
        self._input_topic_name = rospy.get_param('/%s/input_topic_name' % self._name,
                                                 '/camera/color/image_raw')
        self._output_topic_name = rospy.get_param('/%s/output_topic_name' % self._name,
                                                  '/segmentation')
        self._inference_framework = rospy.get_param('/%s/inference_framework' % self._name,
                                                    'EDGETPU')
        self._inference_device = rospy.get_param('/%s/inference_device' % self._name, 'TPU:1')
        self._max_inference_rate = rospy.get_param('/%s/max_inference_rate' % self._name, 10)

    def _init_pub_sub(self):
        self._image_pub = rospy.Publisher(self._output_topic_name, Image, queue_size=1)

        self._input_image_sub = rospy.Subscriber(self._input_topic_name, Image,
                                                 self._input_image_cb)

    def _input_image_cb(self, msg):
        self._input_image_raw = msg

    def _postprocess(self, result):
        im = cv2.cvtColor((20 - result.astype('uint8')) * (255 // 20), cv2.COLOR_GRAY2BGR)
        imc = cv2.applyColorMap(im, cv2.COLORMAP_JET)

        output_image = cv2.resize(self._input_image, (513, 513))

        output_image = cv2.addWeighted(output_image[:, :, ::-1], 0.5, imc, 0.5, 0)

        output_image = cv2.resize((self._input_image_raw.width,
                                   self._input_image_raw.height))
        return output_image

    def _msg_to_nparray(self, msg):
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
