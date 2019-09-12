from collections import defaultdict
import argparse

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from MulticoreTSNE import MulticoreTSNE as TSNE
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', required=True)
    parser.add_argument('--embedding-file', '-e', required=True)
    parser.add_argument('--method', type=str, default='pca', choices=['tsne', 'pca', 'lda'])
    parser.add_argument('--layer-idx', type=int)
    parser.add_argument('--transpose', action='store_true')
    parser.add_argument('--save', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.data_file, sep='\t', quoting=3)
    languages = df['language']
    filename = f'{args.embedding_file}-l{args.layer_idx}.pt' if args.layer_idx is not None else args.embedding_file
    X = torch.load(filename)
    X = X.view(-1, 12, 768 // 12).cuda()
    vars = []
    mean_x = []
    labels = []
    for language in tqdm(set(languages)):
        X_filt = X[torch.tensor(list(languages == language))]
        mean_x.append(X_filt.mean(0))
        labels.extend([language] * 12)
    X = torch.cat(mean_x).cpu().numpy()
    colors = ['r', 'g', 'b', 'y', 'cyan', 'magenta', 'black', 'grey', 'orange', 'purple', 'navy', 'pink'] * len(set(languages))

    print('Fitting...')
    if args.transpose:
        X = X.T
        colors = 'blue'
    if args.method == 'pca':
        X = PCA(n_components=2).fit_transform(X)
    elif args.method == 'tsne':
        X = TSNE(n_jobs=24, verbose=True).fit_transform(X)
    elif args.method == 'lda':
        X = LDA(n_components=2).fit_transform(X, colors)

    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], color=colors)#, s=0.5)
    for idx, txt in enumerate(labels):
        plt.annotate(txt, (X[idx, 0], X[idx, 1]))
    plt.title(f'All Languages (Layer {args.layer_idx + 1 if args.layer_idx >= 0 else 12} Attention Heads)')
    if args.save:
        plt.savefig(f'{args.save}-l{args.layer_idx}.png')
    else:
        plt.show()


if __name__ == '__main__':
    main()