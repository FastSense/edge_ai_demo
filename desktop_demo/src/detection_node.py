#!/usr/bin/env python3

from enum import Enum

import rospy

import numpy as np
import nnio

from sensor_msgs.msg import Image


class Model(Enum):
    DETECTION = 1
    REIDENTIFICATION = 2


class Operation(Enum):
    CREATE = 1
    PREPROCESS = 2
    INFERENCE = 3


EDGE_TPU_NAMES = [
    'EDGE', 'EDGETPU', 'EDGE_TPU'
]

OPENVINO_NAMES = [
    'OPEN_VINO', 'OPENVINO', 'MYRIAD'
]


class ObjectDetector:

    def __init__(self):
        self._accept_params()

        rospy.init_node(self._name)
        self.models = self._create_models()

        self._input_image = None
        self._input_image_raw = None
        self._depth_image = None
        self._depth_image_raw = None
        self._intrinsics = None
        self._boxes = []

        self._database = {}

        self._input_image_sub = rospy.Subscriber(self._camera_image_topic,
                                                 Image, self._input_image_cb)
        self._image_pub = rospy.Publisher(
            self._image_overlay, Image, queue_size=1)

        self._timer = rospy.Timer(rospy.Duration(1.0 / self._max_inference_rate),
                                  self.inference)

    def _input_image_cb(self, msg):
        self._input_image_raw = msg

    def _get_models(self):
        return None

    def input_image_cb(self, msg):
        self._input_image_raw = msg

    def _create_detection(self, box):
        d = {'bbox': {'x1': float(box.x_1), 'y1': float(box.y_1),
                      'x2': float(box.x_2), 'y2': float(box.y_2)}}
        return d

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
            rospy.loginfo('adding %s', new_key)
            if id_min is not None:
                rospy.loginfo('min distance: %s', distances[id_min])
            self._database[new_key] = vec
            return new_key
        return keys[id_min]

    def reshape(self):
        return np.frombuffer(self._input_image_raw.data, dtype='uint8').reshape((self._input_image_raw.height,
                                                                                 self._input_image_raw.width, 3))


    def inference(self, event=None):
        if self._input_image_raw is not None:
            self._input_image = self.reshape()

            img = self.models[Model.DETECTION][Operation.PREPROCESS](self._input_image)
            self._boxes = self.models[Model.DETECTION][Operation.INFERENCE](img)

            output_image = np.copy(self._input_image)
            for box in self._boxes:
                if box.label == 'person':
                    h, w, _ = self._input_image.shape
                    crop = self._input_image[
                        int(h * box.x_1): int(h * box.x_2),
                        int(w * box.y_1): int(w * box.y_2)
                    ]

                    crop_prepared = self.models[Model.REIDENTIFICATION][Operation.PREPROCESS](
                        crop)
                    vec = self.models[Model.REIDENTIFICATION][Operation.INFERENCE](crop_prepared)[
                        0][0]

                    key = self.find_closest(vec, self._treshold)
                    box.label = 'person ' + key

                box.draw(output_image)

            if self._input_image_raw is not None:
                output = Image()
                output.width = self._input_image_raw.width
                output.height = self._input_image_raw.height
                output.data = tuple(output_image.reshape(1, -1)[0])
                output.encoding = 'rgb8'
                output.step = len(output.data) // output.height
                self._image_pub.publish(output)

    def _create_models(self):
        models = {}
        detection_model = self._create_detection_model()
        reid_model = self._create_reidentification_model()

        reid_preproc = nnio.Preprocessing(resize=(128, 256),
                                          dtype='float32',
                                          divide_by_255=True,
                                          means=[0.485, 0.456, 0.406],
                                          scales=1/np.array([0.229, 0.224, 0.225]),
                                          channels_first=True,
                                          batch_dimension=True,
                                          padding=True)

        models[Model.DETECTION] = {Operation.INFERENCE: detection_model,
                                   Operation.PREPROCESS: detection_model.get_preprocessing()}

        models[Model.REIDENTIFICATION] = {Operation.INFERENCE: reid_model,
                                          Operation.PREPROCESS: reid_preproc}

        return models

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

    def _create_reidentification_model(self):
        if self._reid_inference_framework in EDGE_TPU_NAMES:
            model = nnio.EdgeTPUModel(
                device=self._reid_inference_device, model_path=self._reid_model_path)
        if self._reid_inference_framework in OPENVINO_NAMES:
            model = nnio.OpenVINOModel(
                device=self._reid_inference_device, model_xml=self._reid_model_path, model_bin=self._reid_bin_path)
        else:
            model = nnio.ONNXModel(self._reid_model_path)

        return model

    def _accept_params(self):

        self._name = rospy.get_param('node_name', 'object_detector')

        self._camera_image_topic = rospy.get_param('/%s/camera_image_topic' % self._name,
                                                   '/camera/color/image_raw')

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

        self._camera_frame = rospy.get_param(
            '/%s/camera_frame' % self._name, 'camera_link')

        self._max_inference_rate = rospy.get_param('/%s/max_inference_rate' % self._name,
                                                   20)

        self._reid_model_path = rospy.get_param('/%s/reid_model_path' % self._name,
                                                'http://192.168.123.4:8345/onnx/reid/osnet_x1_0_op10.onnx')

        self._reid_bin_path = rospy.get_param(
            '/%s/reid_bin_path' % self._name, '')  # TODO defailt bin path

        self._treshold = rospy.get_param('/%s/treshold' % self._name, 0.7)


if __name__ == '__main__':
    detecor = ObjectDetector()

    rospy.spin()
