from collections import defaultdict
import argparse

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import pandas as pd
import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', required=True)
    parser.add_argument('--embedding-file', '-e', required=True)
    parser.add_argument('--family-file', '-f', required=True)
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'])
    args = parser.parse_args()

    with open(args.family_file) as f:
        family_map = yaml.load(f)
    df = pd.read_csv(args.data_file, sep='\t', quoting=3)
    cmap = defaultdict(lambda: None)
    for c_idx, family in enumerate(family_map):
        delta = 1 / len(family_map[family])
        base_color = np.array([0., 0., 0.])
        base_color[c_idx] = 1
        for language in family_map[family]:
            base_color += delta
            cmap[family_map[family][language]['code']] = np.clip(base_color, 0, 1).tolist()

    colors = np.array(list(map(cmap.__getitem__, df['language'])))
    X = torch.load(args.embedding_file).numpy()
    X = X[colors != None]
    colors = colors[colors != None]
    if args.method == 'pca':
        X = PCA(n_components=2).fit_transform(X)
    elif args.method == 'tsne':
        X = TSNE(n_jobs=24, verbose=True).fit_transform(X)
    plt.scatter(X[:, 0], X[:, 1], color=colors, s=1, alpha=1)
    plt.title('Romantic, Germanic, and Slavic Languages')
    plt.show()


if __name__ == '__main__':
    main()