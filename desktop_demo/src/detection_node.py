#!/usr/bin/env python3

from enum import Enum

import rospy

import numpy as np
import nnio

from threading import Lock, Thread, get_ident
import time
import functools

from sensor_msgs.msg import Image

EDGE_TPU_NAMES = [
    'EDGE', 'EDGETPU', 'EDGE_TPU'
]

OPENVINO_NAMES = [
    'OPEN_VINO', 'OPENVINO', 'MYRIAD'
]


class ModelAdapter():
    """ Adapter Class """

    def __init__(self, model, preprocess):
        self.preprocess = preprocess
        self.model = model

    def inference(self, img):
        return self.model(img)


class ImageSource:
    """ Class which permanently updates stored input image from topic in thread (using rospy.Subscriber),
        storing as well reshaped image, and currently detected boxes, can also reindeficate box with
        given neural network model and image sequence """

    def __init__(self, node_name, in_img_topic, out_img_topic):
        self.mutex = Lock()

        self._image_pub = rospy.Publisher(out_img_topic, Image, queue_size=1)
        self._img_topic_name = in_img_topic
        self._node_name = node_name
        self._in_img_raw = None
        self._in_img_np_array = np.array([])
        self._camera_img_topic = None

        self._boxes = []
        self._get_params()

        self._in_img_sub = rospy.Subscriber(
            self._camera_img_topic, Image, self._input_image_cb)

    def _input_image_cb(self, msg):
        self.mutex.acquire()
        self._in_img_raw = msg
        self.mutex.release()

    def _get_params(self):
        self._camera_img_topic = rospy.get_param('/%s/camera_image_topic' % self._node_name,
                                                 self._img_topic_name)

    def _get_cropped(self, image, box):
        h, w, _ = image.shape
        return image[
            int(h * box.x_1): int(h * box.x_2),
            int(w * box.y_1): int(w * box.y_2)
        ]

    def _raw_to_np_array(self):
        return np.frombuffer(self._in_img_raw.data, dtype='uint8').reshape(
            (self._in_img_raw.height, self._in_img_raw.width, 3))


class ObjectDetector:
    def __init__(self):
        self._get_params()
        self.mutex = Lock()

        rospy.init_node(self._name)

        self._sources = []
        self._models = {}
        self._database = {}

        self._timers = []
        self._threads = []

        self._fill_models()
        self._fill_sources()

        self._inference_timeout_val = 1.0 / self._max_inference_rate
        self._inference_timeout = rospy.Duration(self._inference_timeout_val)

        self._timers.append(rospy.Timer(
            self._inference_timeout, self._get_boxes))

        id = 1
        for source in self._sources:
            self._threads.append(Thread(target=self._inference_source, args=(
                source, "TOPIC_NAME", self._inference_timeout_val)))
            self._threads[-1].start()
            id += 1

    def _get_models(self):
        return None

    def input_image_cb(self, msg):
        self._input_image_raw = msg

    def find_closest(self, vec, threshold=0.7):
        keys = list(self._database.keys())
        vec = vec / np.sqrt((vec**2).mean())
        distances = [
            np.sqrt(((vec - self._database[key])**2).mean())
            for key in keys
        ]
        if len(distances) == 0:
            id_min = None
        else:
            id_min = np.argmin(distances)
        if id_min is None or distances[id_min] > threshold:
            new_key = str(len(self._database))
            rospy.loginfo('adding %s from thread %d', new_key, get_ident())
            if id_min is not None:
                rospy.loginfo('min distance: %s', distances[id_min])
            self._database[new_key] = vec
            return new_key

        return keys[id_min]

    def _get_boxes(self, event=None):
        for source in self._sources:
            self._get_boxes_from_source(source)

    def _get_boxes_from_source(self, source):
        if source._in_img_raw is not None:

            source.mutex.acquire()

            source._in_img_np_array = source._raw_to_np_array()
            img = self._models['DETECTION'].preprocess(source._in_img_np_array)
            source._boxes = self._models['DETECTION'].inference(img)

            source.mutex.release()

    def _inference_source(self, source, timeout):
        while not rospy.is_shutdown():
            if source._in_img_raw is not None and source._in_img_np_array is not None and source._boxes is not None:
                source.mutex.acquire()

                output_image = np.copy(source._in_img_np_array)

                for box in source._boxes:
                    if box.label == 'person':
                        img_cropped = source._get_cropped(output_image, box)

                        img_prepared = self._models['REID'].preprocess(
                            img_cropped)
                        vec = self._models['REID'].inference(img_prepared)[
                            0][0]

                        self.mutex.acquire()
                        key = self.find_closest(vec, self._threshold)
                        self.mutex.release()

                        box.label = 'person ' + key

                for box in source._boxes:
                    box.draw(output_image)

                self._publish_output(source, output_image)

                source.mutex.release()
            time.sleep(timeout)

    def _publish_output(self, source, output_image):
        output = Image()
        output.width = source._in_img_raw.width
        output.height = source._in_img_raw.height
        output.data = tuple(output_image.reshape(1, -1)[0])
        output.encoding = 'rgb8'
        output.step = len(output.data) // output.height

        source._image_pub.publish(output)

    def _fill_sources(self):
        self._sources.append(ImageSource(
            self._name, "/camera1/color/image_raw", "image_overlay_1"))

        self._sources.append(ImageSource(
            self._name, "/camera2/color/image_raw", "image_overlay_2"))

    def _fill_models(self):
        detection_model = self._create_detection_model()
        reid_model = self._create_reid_model()

        reid_preproc = nnio.Preprocessing(resize=(128, 256),
                                          dtype='float32',
                                          divide_by_255=True,
                                          means=[0.485, 0.456, 0.406],
                                          scales=1 /
                                          np.array([0.229, 0.224, 0.225]),
                                          channels_first=True,
                                          batch_dimension=True,
                                          padding=True)

        detection_prepoc = detection_model.get_preprocessing()

        self._models['DETECTION'] = ModelAdapter(
            detection_model, detection_prepoc)
        self._models['REID'] = ModelAdapter(reid_model, reid_preproc)

    def _create_detection_model(self):
        if self._detection_inference_framework in EDGE_TPU_NAMES:
            model = nnio.zoo.edgetpu.detection.SSDMobileNet(
                device=self._detection_inference_device)
        if self._detection_inference_framework in OPENVINO_NAMES:
            model = nnio.zoo.openvino.detection.SSDMobileNetV2(
                device=self._detection_inference_device)
        else:
            model = nnio.zoo.onnx.detection.SSDMobileNetV1()

        return model

    def _create_reid_model(self):
        if self._reid_inference_framework in EDGE_TPU_NAMES:
            model = nnio.EdgeTPUModel(
                device=self._reid_inference_device, model_path=self._reid_model_path)
        if self._reid_inference_framework in OPENVINO_NAMES:
            model = nnio.OpenVINOModel(
                device=self._reid_inference_device, model_xml=self._reid_model_path, model_bin=self._reid_bin_path)
        else:
            model = nnio.ONNXModel(self._reid_model_path)

        return model

    def _get_params(self):

        self._name = rospy.get_param('node_name', 'object_detector')

        self._image_overlay = rospy.get_param('/%s/image_overlay' % self._name,
                                              'image_overlay')

        self._detection_inference_framework = rospy.get_param('/%s/detection_inference_framework' % self._name,
                                                              'ONNX').upper()

        self._detection_inference_device = rospy.get_param('/%s/detection_inference_device' % self._name,
                                                           'CPU').upper()

        self._reid_inference_framework = rospy.get_param('/%s/reid_inference_framework' % self._name,
                                                         'ONNX').upper()

        self._reid_inference_device = rospy.get_param('/%s/reid_inference_device' % self._name,
                                                      'CPU').upper()

        self._max_inference_rate = rospy.get_param('/%s/max_inference_rate' % self._name,
                                                   20)

        self._reid_model_path = rospy.get_param('/%s/reid_model_path' % self._name,
                                                'http://192.168.123.4:8345/onnx/reid/osnet_x1_0_op10.onnx')

        self._reid_bin_path = rospy.get_param(
            '/%s/reid_bin_path' % self._name, '')  # TODO defailt bin path

        self._threshold = rospy.get_param('/%s/threshold' % self._name, 0.7)


if __name__ == '__main__':
    detector = ObjectDetector()

    rospy.spin()
