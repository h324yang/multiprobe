from pathlib import Path
import argparse
import pickle

from matplotlib import pyplot as plt
from sklearn.manifold import MDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', type=Path, required=True)
    args = parser.parse_args()

    with open(str(args.input_file), 'rb') as f:
        data = pickle.load(f)
    matrix = data['matrix']
    filenames = data['filenames']
    X = MDS(dissimilarity='precomputed').fit_transform(matrix)
    colors = []
    c_list = ('r', 'g', 'b')
    for (x, y), filename in zip(X, filenames):
        label = filename.name.split('-', 1)[-1].split('.', 1)[0]
        plt.annotate(label, (x, y))


    plt.scatter(X[:, 0], X[:, 1], color=colors)
    plt.show()


if __name__ == '__main__':
    main()