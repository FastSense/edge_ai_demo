# SSD Mobilenet v2

## Downloading and converting the model

Download the model:

```
./download_model.sh
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

```
trtexec --loadEngine=ssd_mobilenet_v2_int8.trt \
--duration=600 \
--int8 \
--exportTimes=output.json
```

Script will print inference time and put the results in the `output.json`.