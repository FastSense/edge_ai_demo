mkdir model_files


######### Coral

# SSD MobileNet V1
wget https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v1_coco_quant_postprocess.tflite -O model_files/ssd_mobilenet_v1_coco_quant_postprocess.tflite

wget https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite -O model_files/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite


# SSD MobileNet V2
wget https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite -O model_files/ssd_mobilenet_v2_coco_quant_postprocess.tflite

wget https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite -O model_files/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite


# MobileNet V2 Classification
wget https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_v2_1.0_224_quant_edgetpu.tflite -O model_files/mobilenet_v2_1.0_224_quant_edgetpu.tflite


# DeepLabV3
wget https://github.com/google-coral/edgetpu/raw/master/test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite -O model_files/deeplabv3_mnv2_pascal_quant_edgetpu.tflite

wget https://github.com/google-coral/edgetpu/raw/master/test_data/deeplabv3_mnv2_pascal_quant.tflite -O model_files/deeplabv3_mnv2_pascal_quant.tflite


# Human Pose Estimation
wget https://github.com/google-coral/project-posenet/raw/master/models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder.tflite -O model_files/posenet_mobilenet_v1_075_481_641_quant_decoder.tflite
wget https://github.com/google-coral/project-posenet/raw/master/models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite -O model_files/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite


# Human body segmentation
wget https://github.com/google-coral/project-bodypix/raw/master/models/bodypix_mobilenet_v1_075_640_480_16_quant_edgetpu_decoder.tflite -O model_files/bodypix_mobilenet_v1_075_640_480_16_quant_edgetpu_decoder.tflite
