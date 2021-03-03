# MiDaS
This neural network estimates depth from a single image.  
Repo: https://github.com/intel-isl/MiDaS  

Trained with pytorch. There are big and small variants.
Small one works with 256х256 images.

## Downloading and converting the model

Download the model in the onnx format:
```
./download_model.sh
```

Run openVINO docker and mount the current directory to it:

```
docker run --rm -it \
-v /etc/timezone:/etc/timezone:ro \
-v /etc/localtime:/etc/localtime:ro \
-v "$(pwd):/input" openvino/ubuntu18_dev
```

Convert the network from ONNX to openVINO:

```
python3 deployment_tools/model_optimizer/mo.py \
--input_model /input/model-small.onnx \
--model_name midas_vino_fp16 \
--data_type FP16 \
--output_dir /input/
```

As a result, `midas_vino_fp16.bin`, `midas_vino_fp16.xml`, `midas_vino_fp16.mapping` are created in this directory.

## Running the model

To run the model on CPU, you may stay in the same docker container used for converting model.

To run the model on MYRIAD, exit the docker and run this one (possibly on another machine):

```
docker run -itu root:root --rm \
-v /var/tmp:/var/tmp \
--device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' \
-v /dev/bus/usb:/dev/bus/usb \
-v /etc/timezone:/etc/timezone:ro \
-v /etc/localtime:/etc/localtime:ro \
-v "$(pwd):/input" openvino/ubuntu18_runtime
```

```
cd /input/
pip install nnio
python3 test_nnio.py --model-bin midas_vino_fp16.bin --model-xml midas_vino_fp16.xml --device MYRIAD --n-iters 10
```

Script will print inference time and put the results in the `./output` folder.

