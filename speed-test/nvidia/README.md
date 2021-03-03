# SSD Mobilenet v2

## Downloading and converting the model

Download converted models:

```
./download_models.sh
```

Convert the network to TensorRT:

```
trtexec --uff=/home/iowa/Downloads/ssd_mobilenet_v2_coco.uff \
--uffInput=Input,3,300,300 \
--int8 \
--output=MarkOutput_0 \
--workspace=4096 \
--saveEngine=ssd_mobilenet_v2_int8.trt
```

## Running the model

*Mobilenet (classification)* with INT8 precision:
```
trtexec --loadEngine=mobilenetv2_int8.trt \
--duration=600 \
--int8 \
--exportTimes=output_mobilenetv2_int8.json
```

*Mobilenet (classification)* with FP16 precision:
```
trtexec --loadEngine=midas_fp16.trt \
--duration=600 \
--fp16 \
--exportTimes=output_mobilenetv2_fp16.json
```

*SSD Mobilenet V2 (object detection)* with INT8 precision:
```
trtexec --loadEngine=ssd_mobilenet_v2_int8.trt \
--duration=600 \
--int8 \
--exportTimes=output_ssd_mobilenet_v2_int8.json
```

*SSD Mobilenet V2 (object detection)* with FP16 precision:
```
trtexec --loadEngine=ssd_mobilenet_v2_fp16.trt \
--duration=600 \
--fp16 \
--exportTimes=output_ssd_mobilenet_v2_fp16.json
```

*MiDaS (depth estimation)* with INT8 precision:
```
trtexec --loadEngine=midas_int8.trt \
--duration=600 \
--int8 \
--exportTimes=output_midas_int8.json
```

*MiDaS (depth estimation)* with FP16 precision:
```
trtexec --loadEngine=midas_fp16.trt \
--duration=600 \
--fp16 \
--exportTimes=output_midas_fp16.json
```

Scripts will print inference time and put the results in the `output-<model name>.json`.