from pathlib import Path
import argparse
import pickle

from matplotlib import pyplot as plt
from sklearn.manifold import MDS
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', type=Path, required=True)
    args = parser.parse_args()

    with open(str(args.input_file), 'rb') as f:
        data = pickle.load(f)
    matrix = data['matrix']
    filenames = data['filenames']
    X = MDS(dissimilarity='precomputed').fit_transform(matrix)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


if __name__ == '__main__':
    main()