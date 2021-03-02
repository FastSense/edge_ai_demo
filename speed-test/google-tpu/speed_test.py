import nnio

import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Measure inference time on dummy image input'
    )
    parser.add_argument(
        '--model', type=str,
        required=True, help='Path to the model file')
    parser.add_argument(
        '--device', type=str, default='CPU',
        required=False,
        help='Device. Set TPU:0, TPU:1 or TPU:2 to use EdgeTPU. If not specified, will use CPU.')
    parser.add_argument(
        '--width', type=int, default=300,
        required=False, help='Width of the input image')
    parser.add_argument(
        '--height', type=int, default=None,
        required=False, help='Height of the input image')
    parser.add_argument(
        '--n-iters', type=int, default=10,
        required=False, help='N iterations to measure time')
    args = parser.parse_args()
    print(args)

    model = nnio.EdgeTPUModel(args.model, args.device)

    # Create dummy input
    image = np.ones([1, args.height or args.width, args.width, 3])

    # Run model multiple times
    times = []
    for i in range(args.n_iters):
        res, info = model(image, return_info=True)
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
