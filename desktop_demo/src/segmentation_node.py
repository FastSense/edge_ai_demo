#!/usr/bin/env python3

import rospy
import time
import numpy as np
import nnio
import cv2

from sensor_msgs.msg import Image

class SegmentationNode:

    def __init__(self):
        self.accept_params()

        rospy.init_node('segmentation_node')

        self._input_image = None
        self._input_image_raw = None

        self._init_pub_sub()

        self._model = nnio.zoo.edgetpu.segmentation.DeepLabV3()

        self._palette = {0: (17, 17, 17),
                         5: (46, 204, 113),
                         9: (52, 152, 219),
                         15: (231, 76, 60),
                         20: (245, 176, 65)}

        self._timer = rospy.Timer(rospy.Duration(1.0 / 2),
                                  self.inference)

    def accept_params(self):
        pass

    def _init_pub_sub(self):
        self._image_pub = rospy.Publisher('/segmentation1', Image, queue_size=1)

        self._input_image_sub = rospy.Subscriber('/camera/color/image_raw',
                                                 Image, self._input_image_cb)

    def _input_image_cb(self, msg):
        self._input_image_raw = msg


    def inference(self, event=None):
         if self._input_image_raw is not None:
            self._input_image = np.frombuffer(self._input_image_raw.data,
                                              dtype='uint8').reshape((self._input_image_raw.height,
                                                                      self._input_image_raw.width, 3))

            result = self._model(self._model.get_preprocessing()(self._input_image))

            segmented_image = np.ndarray((513, 513, 3), dtype='uint8')

            labels = set()

            for y in range(len(result)):
                for x in range(len(result[0])):
                    labels.add('%s-%d' % (self._model.labels[result[y, x]], result[y, x]))
                    if result[y, x] in self._palette:
                        segmented_image[y, x] = self._palette[result[y, x]]

            rospy.loginfo(labels)

            output_image = cv2.resize(cv2.addWeighted(cv2.resize(
                                      self._input_image, (513, 513))[:, :, ::-1],
                                      0.5, segmented_image, 0.5, 0),
                                      (self._input_image_raw.width,
                                       self._input_image_raw.height))

            output = Image()
            output.width = self._input_image_raw.width
            output.height = self._input_image_raw.height
            output.data = tuple(output_image.reshape(1, -1)[0])
            output.encoding = 'rgb8'
            output.step = len(output.data) // output.height
            self._image_pub.publish(output)


if __name__ == '__main__':
    node = SegmentationNode()
    rospy.spin()
