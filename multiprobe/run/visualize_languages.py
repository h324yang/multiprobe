from collections import defaultdict
import argparse

from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib import pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from tqdm import tqdm
import pandas as pd
import torch

from multiprobe.data import LanguageFamilyData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', required=True)
    parser.add_argument('--embedding-file', '-e', required=True)
    parser.add_argument('--input-type', type=str, required=True, choices=['attn', 'word-emb', 'hidden-emb'])
    parser.add_argument('--family-file', '-f', type=str, default='data/indoeuro-flat.yml')
    parser.add_argument('--method', type=str, default='pca', choices=['tsne', 'pca', 'lda'])
    parser.add_argument('--layer-idx', type=int)
    parser.add_argument('--transpose', action='store_true')
    parser.add_argument('--save', type=str)
    parser.add_argument('--subset-range', type=int, nargs=2, default=(0, 12))
    parser.add_argument('--plot-heads', action='store_true')
    args = parser.parse_args()

    family_data = LanguageFamilyData.from_yaml(args.family_file)
    color_map = defaultdict(lambda: (0.4, 0.4, 0.4, 0.2))
    color_map['Germanic'] = (0.8, 0.1, 0.1, 0.2)
    color_map['Romance'] = (0.1, 0.8, 0.1, 0.2)
    color_map['Slavic'] = (0.1, 0.1, 0.8, 0.2)
    df = pd.read_csv(args.data_file, sep='\t', quoting=3)
    languages = df['language']
    if args.input_type == 'word-emb':
        filename = f'{args.embedding_file}-word-emb.pt'
    else:
        filename = f'{args.embedding_file}-l{args.layer_idx}.pt'
    X = torch.load(filename)
    if args.input_type == 'attn':
        X = X.view(-1, 12, 768 // 12).cuda()
        if args.subset_range:
            X = X[:, args.subset_range[0]:args.subset_range[1]]
    mean_x = []
    labels = []
    num_heads = args.subset_range[1] - args.subset_range[0]
    for language in tqdm(set(languages)):
        X_filt = X[torch.tensor(list(languages == language))]
        if args.plot_heads:
            mean_x.append(X_filt.mean(0))
            label_lst = [language] * num_heads
        else:
            label_lst = [language] * X_filt.size(0)
            if args.input_type == 'attn':
                mean_x.extend(X_filt)
            else:
                mean_x.append(X_filt)
        labels.extend(label_lst)
    X = torch.cat(mean_x).cpu().numpy()
    if args.plot_heads:
        colors = ['r', 'g', 'b', 'y', 'cyan', 'magenta', 'black', 'grey', 'orange', 'purple', 'navy', 'pink'][:num_heads] * len(set(languages))
    else:
        colors = list(map(color_map.__getitem__, map(family_data.find_family, labels)))
        lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='green', lw=2), Line2D([0], [0], color='blue', lw=2)]

    print('Fitting...')
    if args.transpose:
        X = X.T
        colors = 'blue'
    if args.method == 'pca':
        X = PCA(n_components=2).fit_transform(X)
    elif args.method == 'tsne':
        X = TSNE(n_jobs=24, verbose=True).fit_transform(X)
    elif args.method == 'lda':
        pseudo_labels = list(map(repr, colors))
        X = LDA(n_components=2).fit_transform(X, pseudo_labels)

    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], color=colors, s=2 if args.plot_heads else 0.5)
    if args.plot_heads:
        for idx, txt in enumerate(labels):
            plt.annotate(txt, (X[idx, 0], X[idx, 1]))
    else:
        plt.legend(lines, ['Germanic', 'Romance', 'Slavic'])
    if args.input_type == 'attn':
        plt.title(f'All Languages (Layer {args.layer_idx + 1 if args.layer_idx >= 0 else 12} Attention Heads)')
    elif args.input_type == 'word-emb':
        plt.title(f'All Languages (Word Embeddings)')
    elif args.input_type == 'hidden-emb':
        plt.title(f'All Languages (Layer {args.layer_idx + 1 if args.layer_idx >= 0 else 12} Output Embeddings)')

    if args.save:
        plt.savefig(f'{args.save}-l{args.layer_idx}.png')
    else:
        plt.show()


if __name__ == '__main__':
    main()