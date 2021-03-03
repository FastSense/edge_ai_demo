wget https://github.com/dusty-nv/jetson-inference/releases/download/model-mirror-190618/SSD-Mobilenet-v2.tar.gz
tar -zxvf SSD-Mobilenet-v2.tar.gz
mv ./SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff .
rm -fr SSD-Mobilenet-v2
rm -fr SSD-Mobilenet-v2.tar.gz