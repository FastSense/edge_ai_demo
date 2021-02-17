#!/usr/bin/env python3

from enum import Enum

import rospy

import numpy as np
import nnio

from threading import Lock, Thread, get_ident

import time
import itertools

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

        rospy.init_node(self._name)

        self.mutex = Lock()
        self._database = {}

        self._inference_timeout_val = 1.0 / self._max_inference_rate
        self._inference_timeout = rospy.Duration(self._inference_timeout_val)

        self._models = {}
        self._fill_models()
        self._sources = []
        self._fill_sources()

        self._timers = []
        self._timers.append(rospy.Timer(self._inference_timeout, self._detection_inference))

        self._threads = []
        i = 0
        for source in self._sources:
            self._threads.append(Thread(target=self._reid_inference_src, args=(
                source, self._models['REID'][i], self._inference_timeout_val)))
            self._threads[-1].start()
            i += 1

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

    def _detection_inference(self, event=None):
        for source in self._sources:
            self._detection_inference_src(source)

    def _detection_inference_src(self, source):
        if source._in_img_raw is not None:

            source.mutex.acquire()

            source._in_img_np_array = source._raw_to_np_array()
            img = self._models['DETECTION'].preprocess(source._in_img_np_array)
            source._boxes = self._models['DETECTION'].inference(img)

            source.mutex.release()

    def _reid_inference_src(self, source, model, timeout):
        while not rospy.is_shutdown():
            if source._in_img_raw is not None and source._in_img_np_array is not None and source._boxes is not None:
                source.mutex.acquire()

                for box in source._boxes:
                    if box.label == 'person':
                        img_cropped = source._get_cropped(
                            source._in_img_np_array, box)

                        img_prepared = model.preprocess(
                            img_cropped)
                        vec = model.inference(img_prepared)[0][0]

                        self.mutex.acquire()
                        key = self.find_closest(vec, self._reid_threshold)
                        self.mutex.release()

                        box.label = 'person ' + key

                for box in source._boxes:
                    box.draw(source._in_img_np_array)

                self._publish_src_output(source, source._in_img_np_array)

                source.mutex.release()
            time.sleep(timeout)

    def _publish_src_output(self, source, output_image):
        output = Image()
        output.width = source._in_img_raw.width
        output.height = source._in_img_raw.height
        output.data = output_image.tobytes()
        output.encoding = 'rgb8'
        output.step = len(output.data) // output.height

        source._image_pub.publish(output)

    def _fill_sources(self):
        if (len(self._in_img_topics) != len(self._out_img_topics)):
            rospy.signal_shutdown(
                "input topic number not equal to out topic number")

        for i in range(len(self._in_img_topics)):
            self._sources.append(ImageSource(
                self._name, self._in_img_topics[i], self._out_img_topics[i]))

    def _fill_models(self):
        detection_model = self._create_detection_model()
        reid_models = self._create_reid_models()

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
        self._models['REID'] = [ModelAdapter(reid_models[i], reid_preproc)
                                for i in range(len(reid_models))]

    def _create_detection_model(self):
        if self._detection_inference_framework in EDGE_TPU_NAMES:
            model = nnio.zoo.edgetpu.detection.SSDMobileNet(
                device=self._detection_inference_device)
        elif self._detection_inference_framework in OPENVINO_NAMES:
            model = nnio.zoo.openvino.detection.SSDMobileNetV2(
                device=self._detection_inference_device)
        else:
            model = nnio.zoo.onnx.detection.SSDMobileNetV1()

        return model

    def _create_reid_models(self):
        models = []
        for (dev, num, framework, path, bin_path) in zip(self._reid_inference_devices,
                                                         self._reid_device_nums,
                                                         self._reid_inference_frameworks,
                                                         self._reid_model_paths,
                                                         self._reid_bin_paths):

            models.append(self._create_reid_model(
                dev, num, framework, path, bin_path))

        return models

    def _create_reid_model(self, in_device, in_num, in_framework, in_model_path, model_bin_path):
        device_name = (in_device + ':' + str(in_num)) if str(in_num) else in_device

        rospy.logwarn('Creating model with params: \n%s\t %s\n %s\t %s\t', device_name, in_framework, in_model_path, model_bin_path)

        if in_framework in EDGE_TPU_NAMES:
            model = nnio.EdgeTPUModel(
                device=device_name, model_path=in_model_path)
        elif in_framework in OPENVINO_NAMES:
            model = nnio.OpenVINOModel(
                device=device_name, model_xml=in_model_path, model_bin=model_bin_path)
        else:
            model = nnio.ONNXModel(in_model_path)

        rospy.logwarn('Model created\n')

        return model

    def _get_params(self):

        # Node name
        self._name = rospy.get_param('node_name', 'object_detector')

        # Lists
        self._in_img_topics = rospy.get_param(
            '/%s/in_img_topics' % self._name, '')
        self._out_img_topics = rospy.get_param(
            '/%s/out_img_topics' % self._name, '')

        self._reid_inference_frameworks = rospy.get_param(
            '/%s/reid_inference_frameworks' % self._name, '')

        self._reid_inference_devices = rospy.get_param(
            '/%s/reid_inference_devices' % self._name, '')

        self._reid_device_nums = rospy.get_param(
            '/%s/reid_device_nums' % self._name, '')

        self._reid_model_paths = rospy.get_param(
            '/%s/reid_model_paths' % self._name, '')

        self._reid_bin_paths = rospy.get_param(
            '/%s/reid_bin_paths' % self._name, '')

        # Single params
        self._detection_inference_framework = rospy.get_param(
            '/%s/detection_inference_framework' % self._name, 'ONNX').upper()

        self._detection_inference_device = rospy.get_param(
            '/%s/detection_inference_device' % self._name, 'CPU').upper()

        self._reid_threshold = rospy.get_param(
            '/%s/threshold' % self._name, 0.7)
        self._max_inference_rate = rospy.get_param(
            '/%s/max_inference_rate' % self._name, 20)


if __name__ == '__main__':
    detector = ObjectDetector()

    rospy.spin()
