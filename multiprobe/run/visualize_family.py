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
    parser.add_argument('--family-file', '-f', default='data/indoeuro-flat-small.yml')
    parser.add_argument('--method', type=str, default='pca', choices=['tsne', 'pca', 'lda'])
    parser.add_argument('--layer-idx', type=int)
    parser.add_argument('--transpose', action='store_true')
    parser.add_argument('--save', type=str)
    args = parser.parse_args()

    with open(args.family_file) as f:
        family_map = yaml.load(f)
    df = pd.read_csv(args.data_file, sep='\t', quoting=3)
    cmap = defaultdict(lambda: (0.1, 0.1, 0.1, 0.1))
    lang_family_map = {}
    for c_idx, family in enumerate(family_map):
        delta = 0.8 / len(family_map[family])
        base_color = np.array([0., 0., 0.])
        base_color[c_idx] = 1
        for language in family_map[family]:
            lang_family_map[family_map[family][language]['code']] = family
            base_color += delta
            cmap[family_map[family][language]['code']] = np.clip(base_color, 0, 1).tolist()

    languages = df['language']
    colors = np.array(list(map(cmap.__getitem__, languages)))
    filename = f'{args.embedding_file}-l{args.layer_idx}.pt' if args.layer_idx is not None else args.embedding_file
    X = torch.load(filename)
    X = X.view(-1, 12, 768 // 12).cuda()
    vars = []
    for language in tqdm(set(languages)):
        X_filt = X[torch.tensor(list(languages == language))]
        v = X_filt.norm(dim=-1, p=2).mean(0)
        # tqdm.write(repr((language, v)))
        vars.append(v)
    print(torch.stack(vars).var(0))
    X = X.cpu()
    X_family_map = defaultdict(list)
    for x, language in zip(X, languages):
        family = lang_family_map.get(language)
        X_family_map[family].append(x)
    X = []
    colors = np.zeros((48, 4))
    for idx in range(0, 36, 12):
        colors[idx:idx + 12] = np.repeat(np.expand_dims(np.arange(0, 0.8, 0.8 / 12), 1), 4, 1)
        colors[idx:idx + 12, idx // 12] = 1
    colors[24:36] = np.repeat(np.expand_dims(np.arange(0, 0.8, 0.8 / 12), 1), 4, 1)
    colors[:, -1] = 1
    centroid_distance = 0
    centroids = 0
    for key in ('Germanic', 'Romance', None):
        x_lst = X_family_map[key]
        X.append(torch.stack(x_lst).mean(0))
        centroids = torch.stack(x_lst).mean(0) + centroids
    centroids = centroids / 3
    for x_lst in X:
        centroid_distance += (x_lst - centroids).norm(p=2, dim=1)
    centroid_distance = (centroid_distance / 3).mean()
    print(f'Average centroid distance: {centroid_distance.item()}')
    X = torch.cat(X).numpy()

    print('Fitting...')
    if args.transpose:
        X = X.T
        colors = 'blue'
    if args.method == 'pca':
        X = PCA(n_components=2).fit_transform(X)
    elif args.method == 'tsne':
        X = TSNE(n_jobs=24, verbose=True).fit_transform(X)
    elif args.method == 'lda':
        X = LDA(n_components=2).fit_transform(X, list(range(12)) * 3)

    lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='green', lw=2), Line2D([0], [0], color='blue', lw=2)]
    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], color=colors)#, s=0.5)
    for idx, txt in enumerate(list(range(1, 13)) * 3):
        plt.annotate(txt, (X[idx, 0], X[idx, 1]))
    plt.legend(lines, list(family_map.keys()))
    plt.title(f'Top Romance and Germanic Languages (Layer {args.layer_idx + 1 if args.layer_idx >= 0 else 12} Attention Heads)')
    if args.save:
        plt.savefig(f'{args.save}-l{args.layer_idx}.png')
    else:
        plt.show()


if __name__ == '__main__':
    main()