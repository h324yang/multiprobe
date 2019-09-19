from pathlib import Path
from collections import defaultdict
import argparse
import json
import random
import math

from pytorch_transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch

from multiprobe.data import detect_language, TokenLanguageStatistics
from multiprobe.model import remove_bert_heads, SingleInputBundle, predict_top_k
from multiprobe.utils import chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', type=str, required=True)
    parser.add_argument('--token-stats-file', '-s', type=Path, required=True)
    parser.add_argument('--output-distribution-json', '-o', type=Path, required=True)
    parser.add_argument('--ablate', '-a', type=str, nargs='+', default=[])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--bert-model', type=str, default='bert-base-multilingual-uncased')
    parser.add_argument('--mask-prob', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = BertForMaskedLM.from_pretrained(args.bert_model).cuda()
    if args.parallel:
        model = nn.DataParallel(model)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    df = pd.read_csv(args.data_file, sep='\t', quoting=3)

    remove_map = defaultdict(list)
    for ablate_str in args.ablate:
        l_idx, h_idx = ablate_str.split(':')
        remove_map[int(l_idx)].append(int(h_idx))
    for l_idx, heads in remove_map.items():
        remove_bert_heads(model.bert, l_idx, heads)

    stats = TokenLanguageStatistics.from_file(args.token_stats_file)
    distributions = []
    for data in tqdm(list(chunk(list(df.itertuples()), args.batch_size))):
        _, languages, sentences = list(zip(*data))
        sentences = list(map(str.split, sentences))
        for sentence in sentences:
            arange = list(range(len(sentence)))
            random.shuffle(arange)
            for idx in arange[:math.ceil(args.mask_prob * len(arange))]: sentence[idx] = '[MASK]'
        bundle = SingleInputBundle(sentences, tokenizer.vocab)
        bundle.cuda()
        with torch.no_grad():
            tokens_list_list, scores_list_list = predict_top_k(model, tokenizer.vocab, tokenizer.ids_to_tokens, bundle, k=10)
        for tokens_list, scores_list in zip(tokens_list_list, scores_list_list):
            distn_list = []
            for tokens, scores in zip(tokens_list, scores_list):
                weights = F.softmax(scores, 0).cpu().view(-1).numpy()
                try:
                    distn = stats.compute_distribution(tokens, weights=weights).probs.view(-1).tolist()
                    distn_list.append(distn)
                except ValueError:
                    pass
            avg_distn = np.round(np.mean(np.array(distn_list), 0), 7).tolist() if len(distn_list) > 0 else None
            distributions.append(avg_distn)

    with open(args.output_distribution_json, 'w') as f:
        json.dump(distributions, f)


if __name__ == '__main__':
    main()