#!/usr/bin/env python3

import rospy
import argparse
import numpy as np
import cv2
import nnio

from sensor_msgs.msg import Image

class SegmentationNode:

    def __init__(self, name='segmentation_node'):
        self._name = name

        self.accept_params()

        rospy.init_node(self._name)

        self._input_image = None
        self._input_image_raw = None

        self._init_pub_sub()

        self._model = nnio.zoo.edgetpu.segmentation.DeepLabV3()

        self._palette = {0: (17, 17, 17),       # background
                         5: (46, 204, 113),     # bottle
                         9: (52, 152, 219),     # chair
                         15: (231, 76, 60),     # person
                         20: (245, 176, 65)}    # tvmonitor

        self._timer = rospy.Timer(rospy.Duration(1.0 / self._max_inference_rate),
                                  self.inference)

    def accept_params(self):
        self._input_topic_name = rospy.get_param('/%s/input_topic_name' % self._name,
                                                 '/camera/color/image_raw')
        self._output_topic_name = rospy.get_param('/%s/output_topic_name' % self._name,
                                                  '/segmentation')
        self._inference_framework = rospy.get_param('/%s/inference_framework' % self._name,
                                                    'EDGETPU')
        self._inference_device = rospy.get_param('/%s/inference_device' % self._name, 'CPU')
        self._max_inference_rate = rospy.get_param('/%s/max_inference_rate' % self._name, 2)

    def _init_pub_sub(self):
        self._image_pub = rospy.Publisher(self._output_topic_name, Image, queue_size=1)

        self._input_image_sub = rospy.Subscriber(self._input_topic_name, Image,
                                                 self._input_image_cb)

    def _input_image_cb(self, msg):
        self._input_image_raw = msg

    def _postprocess(self, result):
        result = self._model(self._model.get_preprocessing()(self._input_image))

        segmented_image = np.ndarray((513, 513, 3), dtype='uint8')

        labels = set()

        for y in range(len(result)):
            for x in range(len(result[0])):
                labels.add(self._model.labels[result[y, x]])
                if result[y, x] in self._palette:
                    segmented_image[y, x] = self._palette[result[y, x]]

        rospy.loginfo('"%s": %r' % (self._name, labels))

        output_image = cv2.resize(cv2.addWeighted(cv2.resize(
                                  self._input_image, (513, 513))[:, :, ::-1],
                                  0.5, segmented_image, 0.5, 0),
                                  (self._input_image_raw.width,
                                   self._input_image_raw.height))
        return output_image

    def _msg_to_nparray(self, msg):
        img = np.frombuffer(msg.data, dtype='uint8')
        img = img.reshape((self._input_image_raw.height,
                           self._input_image_raw.width, 3))
        return img

    def _publish_img(self, img, width, height, encoding='rgb8'):
        data = tuple(img.reshape(1, -1)[0])

        output = Image(width=width, height=height, data=data,
                       encoding=encoding, step=len(data) // height)

        self._image_pub.publish(output)

    def inference(self, event=None):
         if self._input_image_raw is not None:
            self._input_image = self._msg_to_nparray(self._input_image_raw)

            result = self._model(self._model.get_preprocessing()(self._input_image))

            output_image = self._postprocess(result)

            self._publish_img(output_image,
                              self._input_image_raw.width,
                              self._input_image_raw.height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Node for the neural network segmentation')
    parser.add_argument('--name', type=str, help='name of the node', default='segmentation_node')
    args = parser.parse_args()
    node = SegmentationNode(args.name)
    rospy.spin()
