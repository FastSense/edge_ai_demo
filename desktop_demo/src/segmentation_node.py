#!/usr/bin/env python3

import rospy
import argparse
import numpy as np
import cv2
import nnio
import time

from sensor_msgs.msg import Image

class SegmentationNode:

    def __init__(self, name='segmentation_node'):
        self._name = name

        self.accept_params()

        rospy.init_node(self._name)

        self._input_image = None
        self._input_image_raw = None

        self._init_pub_sub()

        self._model = nnio.zoo.edgetpu.segmentation.DeepLabV3(device=self._inference_device)

        self._timer = rospy.Timer(rospy.Duration(1.0 / self._max_inference_rate),
                                  self.inference)

    def accept_params(self):
        self._input_topic_name = rospy.get_param('/%s/input_topic_name' % self._name,
                                                 '/camera/color/image_raw')
        self._output_topic_name = rospy.get_param('/%s/output_topic_name' % self._name,
                                                  '/segmentation')
        self._inference_framework = rospy.get_param('/%s/inference_framework' % self._name,
                                                    'EDGETPU')
        self._inference_device = rospy.get_param('/%s/inference_device' % self._name, 'TPU')
        self._max_inference_rate = rospy.get_param('/%s/max_inference_rate' % self._name, 10)

    def _init_pub_sub(self):
        self._image_pub = rospy.Publisher(self._output_topic_name, Image, queue_size=1)

        self._input_image_sub = rospy.Subscriber(self._input_topic_name, Image,
                                                 self._input_image_cb)

    def _input_image_cb(self, msg):
        self._input_image_raw = msg

    def _postprocess(self, result, last_time=[0]):
        im = cv2.cvtColor((20 - result.astype('uint8')) * (255 // 20), cv2.COLOR_GRAY2BGR)
        imc = cv2.applyColorMap(im, cv2.COLORMAP_JET)



        output_image = cv2.resize(cv2.addWeighted(cv2.resize(
                                  self._input_image, (513, 513))[:, :, ::-1],
                                  0.5, imc, 0.5, 0),
                                  (self._input_image_raw.width,
                                   self._input_image_raw.height))

        output_image = cv2.putText(
                            output_image,
                            'FPS: %.3f' % (1.0 / (time.time() - last_time[0])) ,
                            (0, 13),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA,
                        )

        last_time[0] = time.time()

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
    parser.add_argument('--name', type=str, help='name of the node', default='segmentation_node1')
    args = parser.parse_args()
    node = SegmentationNode(args.name)
    rospy.spin()
