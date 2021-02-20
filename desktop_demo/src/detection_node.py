#!/usr/bin/env python3

import time
from threading import Lock, Thread

import nnio
import numpy as np

import rospy
from sensor_msgs.msg import Image

EDGE_TPU_FRAMEWORK_NAMES = [
    'EDGE', 'EDGETPU', 'EDGE_TPU'
]

OPENVINO_FRAMEWORK_NAMES = [
    'OPEN_VINO', 'OPENVINO', 'MYRIAD'
]

detection_mutex = Lock()


class Detector:
    """ The class that detects objects from the input image in callback.

        And in parallel thread tries to reindeficate humans,
        who was already detected

    """

    def __init__(self, node_name, in_img_topic, out_img_topic,
                 detection_model, reid_model,
                 reid_threshold, detections_database, inference_rate):
        self._node_name = node_name

        self._img_topic_name = in_img_topic

        self._detection_model = detection_model
        self._detections_database = detections_database

        self._reid_model = reid_model
        self._reid_threshold = reid_threshold

        self._boxes = []
        self._reid_boxes = []

        self._img_np = np.array([])

        self._detection_rate = rospy.Rate(inference_rate * 2.0)
        self._reid_rate = rospy.Rate(inference_rate)

        self._image_pub = rospy.Publisher(out_img_topic, Image, queue_size=1)
        self._in_img_sub = rospy.Subscriber(
            in_img_topic, Image,
            self._input_image_cb, queue_size=1,
            buff_size=2**24, tcp_nodelay=True)

        self._m_reid_thread_ = Thread(target=self._reid_thread)
        self._m_reid_thread_.start()

    def _input_image_cb(self, img):
        """ Input image callback.

            Detecting objects, saving bouding boxes to buffer which
            is processing by reid model in another thread,
            then publishing reindeficated boxes

        """

        self._img_np = self._msg_to_np_arr(img)
        self._boxes = self._detect(self._img_np)
        out_boxes = self._boxes

        if self._reid_boxes:
            out_boxes = self._reid_boxes
            self._draw_boxes(self._img_np, out_boxes, ignore=[])
        else:
            self._draw_boxes(self._img_np, out_boxes, ignore=['person'])

        self._publish_img(self._img_np)
        self._detection_rate.sleep()

    def _reid_thread(self):
        """ Lookup for buffer with bouding boxes.
            Then starts reindefication, saving new persons to database

        """
        while not rospy.is_shutdown():
            if self._boxes:
                img_output = np.copy(self._img_np)
                boxes = self._boxes.copy()

                self._reid_boxes = self._reid(img_output, boxes, 'person')
            else:
                pass
            self._reid_rate.sleep()

    def _reid(self, img_np, boxes, label):
        for box in boxes:
            if box.label == label:
                img_cropped = self._crop(img_np, box)
                img_prepared = self._reid_model.get_preprocessing()(img_cropped)

                vec = self._reid_model(img_prepared)
                key = self._detections_database.find_closest(vec)
                box.label = label + ' ' + key
        return boxes

    def _detect(self, img_np):
        """Inference of the detection model

        Returns
            boxes (list): bouding boxes of detected objects

        """
        img_preprocessed = self._detection_model.get_preprocessing()(img_np)
        detection_mutex.acquire()
        boxes = self._detection_model(img_preprocessed)
        detection_mutex.release()

        return boxes

    def _msg_to_np_arr(self, img_raw):
        return np.frombuffer(img_raw.data, dtype='uint8').reshape(
            (img_raw.height, img_raw.width, 3))

    def _crop(self, image, box):
        h, w, _ = image.shape
        return image[
            int(h * max(0, box.x_1)): int(h * min(1, box.x_2)),
            int(w * max(0, box.y_1)): int(w * min(1, box.y_2))
        ]

    def _draw_boxes(self, img_np, boxes, ignore):
        for box in boxes:
            if box.label not in ignore:
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


class DetectionNode:
    """Class that creates one detection model and several threads for processing input images

        Also reindeficate detected persons storing them in common database

    """

    def __init__(self):
        self._receive_params()
        rospy.init_node(self._name)
        self._detections_database = nnio.utils.HumanDataBase(
            new_entity_threshold=self._reid_threshold, merging_threshold=0.20)

        self._models = self._make_models()
        self._sources = self._make_sources()

    def _make_sources(self):
        sources = []
        self._check_in_out_equality()

        for i in range(len(self._in_img_topics)):
            # pass single detection model on each source
            detection_model = self._models['DETECTION']

            # pass unique reid model on each source
            reid_model = self._models['REID'][i]
            sources.append(
                Detector(self._name, self._in_img_topics[i], self._out_img_topics[i],
                         detection_model, reid_model,
                         self._reid_threshold, self._detections_database, self._inference_rate))

        return sources

    def _check_in_out_equality(self):
        if len(self._in_img_topics) != len(self._out_img_topics):
            rospy.signal_shutdown(
                'input topic number not equal to output topic number')

    # Using nnio package for model creation

    def _make_models(self):
        models = {}

        models['DETECTION'] = self._make_detection_model()
        models['REID'] = self._make_reid_models()

        return models

    def _make_detection_model(self):
        rospy.logwarn('Creating detection model with params: \n%s\t %s\n',
                      self._detection_inference_device, self._detection_inference_framework)

        if self._detection_inference_framework in EDGE_TPU_FRAMEWORK_NAMES:
            model = nnio.zoo.edgetpu.detection.SSDMobileNet(
                device=self._detection_inference_device)
        elif self._detection_inference_framework in OPENVINO_FRAMEWORK_NAMES:
            model = nnio.zoo.openvino.detection.SSDMobileNetV2(
                device=self._detection_inference_device)
        else:
            print('DETECTION: ONNX!!!!')
            model = nnio.zoo.onnx.detection.SSDMobileNetV1()

        rospy.logwarn('Detection model created\n')

        return model

    def _make_reid_models(self):
        models = []
        for (dev, num, framework) in zip(self._reid_inference_devices,
                                         self._reid_device_nums,
                                         self._reid_inference_frameworks):

            models.append(self._make_reid_model(dev, num, framework))

            time.sleep(1.0)

        return models

    def _make_reid_model(self, in_device, in_num, in_framework):
        device_name = (in_device + ':' + str(in_num)
                       ) if str(in_num) else in_device

        rospy.logwarn('Creating reid model with params: \n%s\t %s\n',
                      device_name, in_framework)

        if in_framework in EDGE_TPU_FRAMEWORK_NAMES:
            model = nnio.zoo.edgetpu.reid.OSNet(device=device_name)
        elif in_framework in OPENVINO_FRAMEWORK_NAMES:
            model = nnio.zoo.openvino.reid.OSNet(device=device_name)
        else:
            model = nnio.zoo.onnx.reid.OSNet(device=device_name)

        rospy.logwarn('Reid model created\n')

        return model

    def _receive_params(self):
        self._name = rospy.get_param('node_name', 'object_detector')

        # Multiple params
        self._in_img_topics = rospy.get_param(
            '/%s/in_img_topics' % self._name, '[/camera1/color/image_raw, /camera2/color/image_raw]')
        self._out_img_topics = rospy.get_param(
            '/%s/out_img_topics' % self._name, '[detections_1, detections_2]')

        # Single params
        self._detection_inference_framework = rospy.get_param(
            '/%s/detection_inference_framework' % self._name, 'EDGETPU').upper()

        self._detection_inference_device = rospy.get_param(
            '/%s/detection_inference_device' % self._name, 'TPU:0').upper()

        self._reid_device_nums = rospy.get_param(
            '/%s/reid_device_nums' % self._name, '["0", "0"]')

        self._reid_inference_frameworks = rospy.get_param(
            '/%s/reid_inference_frameworks' % self._name, '[OPENVINO, OPENVINO]')
        self._reid_inference_devices = rospy.get_param(
            '/%s/reid_inference_devices' % self._name, '[MYRIAD, MYRIAD]')
        self._reid_threshold = rospy.get_param(
            '/%s/threshold' % self._name, 0.25)

        self._inference_rate = rospy.get_param(
            '/%s/inference_rate' % self._name, 40)


if __name__ == '__main__':
    detector = DetectionNode()

    rospy.spin()
