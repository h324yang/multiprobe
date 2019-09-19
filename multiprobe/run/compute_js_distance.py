from pathlib import Path
import argparse
import json

from tqdm import tqdm, trange
import numpy as np

from multiprobe.data import sum_js


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=Path, nargs='+')
    args = parser.parse_args()

    distribution_list = []
    for filename in tqdm(args.files, desc='Loading'):
        with open(filename) as f:
            distribution_list.append(json.load(f))

    distance_matrix = np.zeros((len(distribution_list), len(distribution_list)))
    for idx_i in trange(len(distribution_list) - 1):
        for idx_j in range(idx_i + 1, len(distribution_list)):
            dist = sum_js(distribution_list[idx_i], distribution_list[idx_j])
            distance_matrix[idx_j, idx_i] = distance_matrix[idx_i, idx_j] = dist
    print(distance_matrix)


if __name__ == '__main__':
    main()
