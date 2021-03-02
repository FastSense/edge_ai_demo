# EdgeTPU speed test
First download the models:
```
./download_models.sh
```

Running speed test:

```
usage: speed_test.py [-h] --model MODEL [--device DEVICE] [--width WIDTH] [--height HEIGHT]

Measure inference time on dummy image input

optional arguments:
  -h, --help       show this help message and exit
  --model MODEL    Path to the model file
  --device DEVICE  Device. Set TPU:0, TPU:1 or TPU:2 to use EdgeTPU. If not specified, will use CPU.
  --width WIDTH    Width of the input image
  --height HEIGHT  Height of the input image
  --n-iters N_ITERS  N iterations to measure time
```

**WARNING**: If `--device` option is not set, inference will run on CPU, not EdgeTPU.

`--width` is the input size. It is hard-written into the model files. You cannot run a model with wrong input size.

## SSD MobileNet V1
```
python3 speed_test.py \
        --model model_files/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite \
        --device TPU \
        --width 300
```

## SSD MobileNet V2
```
python3 speed_test.py \
        --model model_files/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
        --device TPU \
        --width 300
```

## Classification MobileNet V2
```
python3 speed_test.py \
        --model model_files/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
        --device TPU \
        --width 224
```

## Segmentation DeeplabV3
```
python3 speed_test.py \
        --model model_files/deeplabv3_mnv2_pascal_quant_edgetpu.tflite \
        --device TPU\
        --width 513
```
