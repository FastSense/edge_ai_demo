import numpy as np
import cv2
import os
import argparse
from pprint import pprint
import time

import nnio

def write_depth(path, depth, bits=1):
    """Write depth map to png file.
    Args:
        path (str): filepath without extension
        depth (array): depth
    """

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1:
        cv2.imwrite(path, out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path, out.astype("uint16"))

    return

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
        '--width', type=int, default=256,
        required=False, help='Width of the input image')
    parser.add_argument(
        '--height', type=int, default=None,
        required=False, help='Height of the input image')
    parser.add_argument(
        '--n-iters', type=int, default=10,
        required=False, help='N iterations to measure time')
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
        scales=1/255,
        channels_first=True,
        batch_dimension=True,
    )

    # Process input files
    input_files = os.listdir('input')
    for f in input_files:
        image_path = os.path.join('input', f)
        img_prep, img = preproc(image_path, return_original=True)
        (img_h, img_w, _) = img.shape
        out = model(img_prep)
        out = cv2.resize(out[0], (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        write_depth(os.path.join('output', f), out)

    # Measure time
    N_TRIES = args.n_iters

    times = []
    for i in range(N_TRIES):
        start = time.time()
        out = model(img_prep)
        end = time.time()
        times.append(end - start)
    print("{:.4f}".format(sum(times)/N_TRIES))
    print("FPS = {:.4f}".format(N_TRIES/sum(times)))

    percentiles = [0, 50, 99, 100]
    results = np.percentile(times, percentiles)
    print('Percentiles:')
    for p, res in zip(percentiles, results):
        print(f'{p}%:\t{res}')


if __name__ == '__main__':
    main()
