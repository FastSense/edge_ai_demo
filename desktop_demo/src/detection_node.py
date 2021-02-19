#!/usr/bin/env python3

from enum import Enum
from profiler import profile

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

database_mutex = Lock()
detection_mutex = Lock()
img_mutex = Lock()


class ModelAdapter():
    """ Adapter Class """

    def __init__(self, model, preprocess):
        self.preprocess = preprocess
        self.model = model

    @profile
    def inference(self, img):
        return self.model(img)


class ImageSourceProcesser:
    def __init__(self, node_name, in_img_topic, out_img_topic, detection_model, reid_model, reid_threshold, database, inference_rate):
        self._reid_threshold = reid_threshold
        self._node_name = node_name
        self._img_topic_name = in_img_topic
        self._image_pub = rospy.Publisher(out_img_topic, Image, queue_size=1)
        self._detection_model = detection_model
        self._reid_model = reid_model
        self._reid_threshold = reid_threshold

        self._reid_boxes = []
        self._has_reid_boxes = False

        self._database = database

        self._camera_img_topic = None
        self._get_params()

        self._img_np = np.array([])
        self._boxes = []

        self._detection_rate = rospy.Rate(inference_rate * 2.0)
        self._reid_rate = rospy.Rate(inference_rate)

        self._in_img_sub = rospy.Subscriber(
            self._camera_img_topic, Image, self._input_image_cb, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        self._m_reid_thread_ = Thread(target=self._reid_thread)
        self._m_reid_thread_.start()

    def _get_params(self):
        self._camera_img_topic = rospy.get_param('/%s/camera_image_topic' % self._node_name,
                                                 self._img_topic_name)

    def _input_image_cb(self, img):
        if img is None:
            return

        self._img_np = self._to_np_arr(img)

        self._boxes = self._detect(self._img_np)
        out_boxes = self._boxes

        if self._has_reid_boxes:
            out_boxes = self._reid_boxes
            self._has_reid_boxes = False
            self._draw_boxes(self._img_np, out_boxes, [])
        else:
            self._draw_boxes(self._img_np, out_boxes, ['person'])

        self._publish_img(self._img_np)
        self._detection_rate.sleep()

    def _reid_thread(self):
        while not rospy.is_shutdown():
            if self._boxes:
                img_mutex.acquire()
                img_output = np.copy(self._img_np)
                boxes = self._boxes.copy()
                img_mutex.release()

                self._reid_boxes = self._reid(img_output, boxes, 'person')
                self._has_reid_boxes = True
            else:
                pass
            self._reid_rate.sleep()

    def _to_np_arr(self, img_raw):
        return np.frombuffer(img_raw.data, dtype='uint8').reshape(
            (img_raw.height, img_raw.width, 3))

    @profile
    def _detect(self, img_np):
        img_preprocessed = self._detection_model.preprocess(img_np)
        detection_mutex.acquire()
        boxes = self._detection_model.inference(img_preprocessed)
        detection_mutex.release()

        return boxes

    @profile
    def _reid(self, img_np, boxes, label):
        for box in boxes:
            if box.label == label:
                img_cropped = self._crop(img_np, box)
                img_prepared = self._reid_model.preprocess(img_cropped)
                vec = self._reid_model.inference(img_prepared)[0]
                database_mutex.acquire()
                key = self._find_closest(vec, self._reid_threshold)
                database_mutex.release()
                box.label = label + ' ' + key
        return boxes

    def _crop(self, image, box):
        h, w, _ = image.shape
        return image[
            int(h * max(0, box.x_1)): int(h * min(1, box.x_2)),
            int(w * max(0, box.y_1)): int(w * min(1, box.y_2))
        ]

    def _find_closest(self, vec, threshold=0.7):
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
            rospy.loginfo('adding %s', new_key)
            if id_min is not None:
                rospy.loginfo('min distance: %s', distances[id_min])
            self._database[new_key] = vec
            return new_key

        return keys[id_min]

    def _draw_boxes(self, img_np, boxes, ignore):
        for box in boxes:
            if box not in ignore:
                box.draw(img_np)
        return img_np

    def _publish_img(self, img_np):
        output = Image()
        output.width = img_np.shape[1]
        output.height = img_np.shape[0]
        output.data = img_np.tobytes()
        output.encoding = 'rgb8'
        output.step = len(output.data) // output.height

        self._image_pub.publish(output)


class ObjectDetector:
    def __init__(self):
        self._get_params()
        rospy.init_node(self._name)
        self._database = {}

        self._models = self._get_models()
        self._sources = self._get_sources()

    def _get_sources(self):
        sources = []

        if (len(self._in_img_topics) != len(self._out_img_topics)):
            rospy.signal_shutdown(
                "input topic number not equal to out topic number")

        for i in range(len(self._in_img_topics)):
            sources.append(
                ImageSourceProcesser(self._name, self._in_img_topics[i], self._out_img_topics[i],
                                     self._models['DETECTION'], self._models['REID'][i],
                                     self._reid_threshold, self._database, self._inference_rate))
            rospy.logwarn(self._inference_rate)

        return sources

    def _get_models(self):
        models = {}

        detection_model = self._create_detection_model()
        reid_models = self._create_reid_models()

        detection_prepoc = detection_model.get_preprocessing()

        models['DETECTION'] = ModelAdapter(
            detection_model, detection_prepoc)
        models['REID'] = [ModelAdapter(
            reid_models[i], reid_models[i].get_preprocessing()) for i in range(len(reid_models))]

        return models

    def _create_detection_model(self):
        rospy.logwarn('Creating detection model with params: \n%s\t %s\n',
                      self._detection_inference_device, self._detection_inference_framework)

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

            time.sleep(4.0)

        return models

    def _create_reid_model(self, in_device, in_num, in_framework, in_model_path, model_bin_path):
        device_name = (in_device + ':' + str(in_num)
                       ) if str(in_num) else in_device

        rospy.logwarn('Creating reid model with params: \n%s\t %s\n %s\n %s\t',
                      device_name, in_framework, in_model_path, model_bin_path)

        if in_framework in EDGE_TPU_NAMES:
            model = nnio.zoo.edgetpu.reid.OSNet(device=device_name)
        elif in_framework in OPENVINO_NAMES:
            model = nnio.zoo.openvino.reid.OSNet(device=device_name)
        else:
            model = nnio.zoo.onnx.reid.OSNet(device=device_name)

        rospy.logwarn('Model created\n')

        return model

    def _get_params(self):
        self._name = rospy.get_param('node_name', 'object_detector')

        # Multiple params
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

        self._inference_rate = rospy.get_param(
            '/%s/inference_rate' % self._name, 20)


if __name__ == '__main__':
    detector = ObjectDetector()

    rospy.spin()
