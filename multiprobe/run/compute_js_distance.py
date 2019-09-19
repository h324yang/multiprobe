from pathlib import Path
import argparse
import json
import pickle

from tqdm import tqdm, trange
import numpy as np

from multiprobe.data import sum_js, sum_js_cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    distribution_list = []
    for filename in tqdm(args.files, desc='Loading'):
        with open(filename) as f:
            distribution_list.append(json.load(f))

    distance_matrix = np.zeros((len(distribution_list), len(distribution_list)))
    for idx_i in trange(len(distribution_list) - 1):
        for idx_j in range(idx_i + 1, len(distribution_list)):
            dist = sum_js_cuda(distribution_list[idx_i], distribution_list[idx_j])
            distance_matrix[idx_j, idx_i] = distance_matrix[idx_i, idx_j] = dist
    data = dict(matrix=distance_matrix, filenames=args.files)
    print(distance_matrix)
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
