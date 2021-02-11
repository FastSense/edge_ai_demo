#!/usr/bin/env python3

import json
import rospy
import numpy as np
import nnio
import pyrealsense2 as rs2

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

class ObjectDetector:

    def __init__(self, name='object_detector'):
        self._name = name
        self._accept_params()

        rospy.init_node(name)

        self._det_model, self._reid_model = self._create_models()

        self._reid_preproc = nnio.Preprocessing(
            resize=(128, 256),
            dtype='float32',
            divide_by_255=True,
            means=[0.485, 0.456, 0.406],
            scales=1/np.array([0.229, 0.224, 0.225]),
            channels_first=True,
            batch_dimension=True,
            padding=True
        )

        self._input_image = None
        self._input_image_raw = None
        self._depth_image = None
        self._depth_image_raw = None
        self._intrinsics = None
        self._boxes = []

        self._database = {}

        self._input_image_sub = rospy.Subscriber(self._camera_image_topic,
                                                 Image, self._input_image_cb)

        self._image_pub = rospy.Publisher('/image_overlay', Image, queue_size=1)

        self._vis_pub = rospy.Publisher('/detections', String, queue_size=1)

        self._timer = rospy.Timer(rospy.Duration(1.0 / self._max_inference_rate),
                                  self.inference)
    
    def _input_image_cb(self, msg):
        self._input_image_raw = msg

    def _create_detection(self, box):
        d = {'bbox': {'x1': float(box.x_1), 'y1': float(box.y_1),
                      'x2': float(box.x_2), 'y2': float(box.y_2)}}
        return d

    def find_closest(self, vec, treshold=0.7):
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
        if id_min is None or distances[id_min] > treshold:
            new_key = str(len(self._database))
            print('adding', new_key)
            if id_min is not None:
                print(distances[id_min])
            self._database[new_key] = vec
            return new_key
        return keys[id_min]

    def inference(self, event=None):
        det = {}

        if self._input_image_raw is not None:
            self._input_image = np.frombuffer(self._input_image_raw.data,
                                              dtype='uint8').reshape((self._input_image_raw.height,
                                                                      self._input_image_raw.width, 3))

            img = self._det_model.get_preprocessing()(self._input_image)
            self._boxes = self._det_model(img)
            output_image = np.copy(self._input_image)

            for box in self._boxes:
                if box.label == 'person':
                    h, w, _ = self._input_image.shape
                    crop = self._input_image[
                        int(h * box.x_1): int(h * box.x_2),
                        int(w * box.y_1): int(w * box.y_2)
                    ]
                    crop_prepared = self._reid_preproc(crop)
                    vec = self._reid_model(crop_prepared)[0][0]

                    key = self.find_closest(vec)
                    box.label = 'person ' + key

                box.draw(output_image)

            self._vis_pub.publish(String(data=json.dumps(det)))

            if self._input_image_raw is not None:
                output = Image()
                output.width = self._input_image_raw.width
                output.height = self._input_image_raw.height
                output.data = tuple(output_image.reshape(1, -1)[0])
                output.encoding = 'rgb8'
                output.step = len(output.data) // output.height
                self._image_pub.publish(output)

    def _create_models(self):
        det_model = nnio.zoo.onnx.detection.SSDMobileNetV1()

        if self._inference_framework in ('EDGE', 'EDGETPU', 'EDGE_TPU'):
            det_model = nnio.zoo.edgetpu.detection.SSDMobileNet(device=self._inference_device)
        if self._inference_framework in ('OPEN_VINO', 'OPENVINO', 'MYRIAD'):
            det_model = nnio.zoo.openvino.detection.SSDMobileNetV2(device=self._inference_device)
        else:
            pass
            #return nnio.zoo.onnx.detection.SSDMobileNetV1()

        reid_model = nnio.ONNXModel('http://192.168.123.4:8345/onnx/reid/osnet_x1_0_op10.onnx')

        return det_model, reid_model

    def _accept_params(self):
        self._camera_image_topic = rospy.get_param('/%s/camera_image_topic' % self._name,
                                                   '/camera/color/image_raw')

        self._inference_framework = rospy.get_param('/%s/inference_framework' % self._name,
                                                    'ONNX').upper()
        
        self._inference_device = rospy.get_param('/%s/inference_device' % self._name,
                                                 'CPU').upper()

        self._camera_frame = rospy.get_param('/%s/camera_frame' % self._name,
                                             'camera_link')

        self._max_inference_rate = rospy.get_param('/%s/max_inference_rate' % self._name,
                                                   20)


if __name__ == '__main__':
    detecor = ObjectDetector('object_detector')

    rospy.spin()
