import numpy as np
import cv2
import os
import argparse
from pprint import pprint
import time

import nnio

def main():
    parser = argparse.ArgumentParser(
        description='Measure inference time on dummy image input'
    )
    parser.add_argument(
        '--model-bin', type=str,
        required=True, help='Path to the model .bin file')
    parser.add_argument(
        '--model-xml', type=str,
        required=True, help='Path to the model .xml file')
    parser.add_argument(
        '--device', type=str, default='CPU',
        required=False,
        help='Device. CPU or GPU or MYRIAD:0 or MYRIAD:1.')
    parser.add_argument(
        '--width', type=int, default=300,
        required=False, help='Width of the input image')
    parser.add_argument(
        '--height', type=int, default=None,
        required=False, help='Height of the input image')
    parser.add_argument(
        '--channels-first', type=bool, default=True,
        required=False, help='If True, process images in BCHW format')
    parser.add_argument(
        '--n-iters', type=int, default=10,
        required=False, help='Number of speed test interations')
    args = parser.parse_args()
    args.height = args.height or args.width
    print(args)

    # Create model
    model = nnio.OpenVINOModel(
        args.model_bin,
        args.model_xml,
        args.device
    )
    # Create preprocessor
    preproc = nnio.Preprocessing(
        resize=(args.width, args.height),
        dtype='float32',
        divide_by_255=True,
        channels_first=args.channels_first,
        batch_dimension=True,
    )

    # Process input files
    input_files = os.listdir('input')
    for f in input_files:
        image_path = os.path.join('input', f)
        frame, frame_orig = preproc(image_path, return_original=True)
        (img_h, img_w, _) = frame_orig.shape
        out = model(frame)
        out = cv2.resize(out[0], (img_w, img_h), interpolation=cv2.INTER_CUBIC)

    # Run model multiple times
    times = []
    for i in range(args.n_iters):
        res, info = model(frame, return_info=True)
        times.append(info['invoke_time'])
        print('{:.02f} ms'.format(info['invoke_time'] * 1000))
    
    print("\nAverage: {:.2f} ms".format(sum(times) / args.n_iters * 1000))
    print("FPS = {:.2f}".format(args.n_iters / sum(times)))

    percentiles = [0, 50, 99, 100]
    results = np.percentile(times, percentiles)
    print('Percentiles:')
    for p, res in zip(percentiles, results):
        print(f'{p}%:\t{res * 1000:.02f} ms')


if __name__ == '__main__':
    main()
