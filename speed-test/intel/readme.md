# Downloading and converting models:

Run docker for converting models:

```
docker run --rm -it -v "$(pwd):/input" openvino/ubuntu18_dev
```

Download some models for tests:

```
python3 deployment_tools/tools/model_downloader/downloader.py --name mobilenet-v2 -o /input/
python3 deployment_tools/tools/model_downloader/downloader.py --name ssd_mobilenet_v2_coco -o /input/
python3 deployment_tools/tools/model_downloader/downloader.py --name ssdlite_mobilenet_v2 -o /input/
```

They will be downloaded into the current directory in `./public/`.

Convert the `mobilenet-v2` model to openvino with different data types:

```
python3 deployment_tools/model_optimizer/mo.py --input_model /input/public/mobilenet-v2/mobilenet-v2.caffemodel --output_dir /input/ --model_name mobilenet-v2-fp16 --data_type FP16
python3 deployment_tools/model_optimizer/mo.py --input_model /input/public/mobilenet-v2/mobilenet-v2.caffemodel --output_dir /input/ --model_name mobilenet-v2-fp32 --data_type FP32
```

Convert other models (they are from tensorflow):
```
python3 deployment_tools/model_optimizer/mo_tf.py --saved_model_dir /input/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/saved_model/ --output_dir /input --model_name mobilenet_v2_coco_fp16 --data_type FP16 --input_shape [1,300,300,3] --transformations_config deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /input/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config

python3 deployment_tools/model_optimizer/mo_tf.py --saved_model_dir /input/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/saved_model/ --output_dir /input --model_name mobilenet_v2_coco_fp32 --data_type FP32 --input_shape [1,300,300,3] --transformations_config deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /input/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config

python3 deployment_tools/model_optimizer/mo_tf.py --saved_model_dir /input/public/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model/ --output_dir /input --model_name ssdlite_mobilenet_v2_coco_fp32 --data_type FP32 --input_shape [1,300,300,3] --transformations_config deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /input/public/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config

python3 deployment_tools/model_optimizer/mo_tf.py --saved_model_dir /input/public/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model/ --output_dir /input --model_name ssdlite_mobilenet_v2_coco_fp16 --data_type FP16 --input_shape [1,300,300,3] --transformations_config deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /input/public/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config
```

# Running tests:

To run tests on your machine with CPU, stay in the same docker container.

To run models on a machine with Myriads, copy all files from this folder to that machine and run this docker:
```
docker run -itu root:root --rm \
-v /var/tmp:/var/tmp \
--device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' \
-v /dev/bus/usb:/dev/bus/usb \
-v /etc/timezone:/etc/timezone:ro \
-v /etc/localtime:/etc/localtime:ro \
-v "$(pwd):/input" openvino/ubuntu18_runtime
```

Go to the `/input` directory to which the current working directory is mounted:

```
cd /input
```

Install nnio
```
pip3 install nnio
```

Run the speed tests:

```
python3 speed_test.py --model-bin mobilenet-v2-fp16.bin --model-xml mobilenet-v2-fp16.xml --width 224 --device CPU --n-iters 10
python3 speed_test.py --model-bin mobilenet-ssd-fp16.bin --model-xml mobilenet-ssd-fp16.xml --width 300 --device CPU --n-iters 10
```

Here you can change the device to `GPU` or `MYRIAD` if your machine has any of them.
