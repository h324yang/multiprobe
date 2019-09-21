from pathlib import Path
from multiprocessing.sharedctypes import RawArray
import argparse
import ctypes as c
import multiprocessing as mp
import json
import pickle
import time

from tqdm import tqdm, trange
import numpy as np

from multiprobe.data import sum_js


class SliceComputingJob(object):

    def __init__(self, arr):
        self.arr = arr

    def __call__(self, idx_i):
        slice_data = []
        p_idx = mp.current_process()._identity[0] - 1
        for idx_j in trange(idx_i + 1, len(self.arr), position=p_idx):
            dist = sum_js(self.arr[idx_i], self.arr[idx_j])
            slice_data.append(((idx_i, idx_j), dist))
        return slice_data


job = None


def execute_job(idx):
    return job(idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-count', type=int, default=0)
    args = parser.parse_args()
    distribution_list = []

    for filename in tqdm(args.files, desc='Loading'):
        with open(filename) as f:
            distribution_list.append(json.load(f))

    d_arr = np.array(distribution_list)
    mp_arr = RawArray(c.c_double, int(np.prod(d_arr.shape)))
    arr = np.frombuffer(mp_arr).reshape(d_arr.shape)
    np.copyto(arr, d_arr)
    del distribution_list, d_arr

    distance_matrix = np.zeros((len(arr), len(arr)))
    global job
    job = SliceComputingJob(arr)
    p = mp.Pool(mp.cpu_count() if args.cpu_count <= 0 else args.cpu_count)
    slices_lst = p.map(execute_job, range(len(arr) - 1))
    for slices in slices_lst:
        for (idx_i, idx_j), dist in slices:
            distance_matrix[idx_i, idx_j] = distance_matrix[idx_j, idx_i] = dist
    data = dict(matrix=distance_matrix, filenames=args.files)
    print(distance_matrix)
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
