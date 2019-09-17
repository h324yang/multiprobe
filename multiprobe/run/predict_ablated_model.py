from collections import defaultdict
import argparse
import random
import math

from pytorch_transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm
import pandas as pd
import torch

from multiprobe.model import remove_bert_heads, SingleInputBundle, predict_top_k
from multiprobe.utils import chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', type=str, required=True)
    parser.add_argument('--ablate', '-a', type=str, required=True, nargs='+')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--bert-model', type=str, default='bert-base-multilingual-uncased')
    parser.add_argument('--mask-prob', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    model = BertForMaskedLM.from_pretrained(args.bert_model).cuda()
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    df = pd.read_csv(args.data_file, sep='\t', quoting=3)

    remove_map = defaultdict(list)
    for ablate_str in args.ablate:
        l_idx, h_idx = ablate_str.split(':')
        remove_map[int(l_idx)].append(int(h_idx))
    for l_idx, heads in remove_map.items():
        remove_bert_heads(model.bert, l_idx, heads)
    for data in tqdm(list(chunk(list(df.itertuples()), args.batch_size))):
        _, languages, sentences = list(zip(*data))
        if 'zh' not in languages: continue
        sentences = list(map(str.split, sentences))
        for sentence in sentences:
            arange = list(range(len(sentence)))
            random.shuffle(arange)
            for idx in arange[:math.ceil(args.mask_prob * len(arange))]: sentence[idx] = '[MASK]'
        bundle = SingleInputBundle(sentences, tokenizer.vocab)
        bundle.cuda()
        with torch.no_grad():
            print(languages, predict_top_k(model, tokenizer.vocab, tokenizer.ids_to_tokens, bundle))


if __name__ == '__main__':
    main()