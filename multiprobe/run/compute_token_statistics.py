from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json

from pytorch_transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', type=Path, required=True)
    parser.add_argument('--output-json', '-o', type=Path, required=True)
    parser.add_argument('--bert-model', type=str, default='bert-base-multilingual-uncased')
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    df = pd.read_csv(str(args.data_file), sep='\t', quoting=3)
    token_stats = defaultdict(Counter)
    for _, language, tokens in tqdm(list(df.itertuples())):
        for token in tokens.split():
            token_stats[token][language] += 1
    data = dict(statistics=token_stats, vocab=tokenizer.vocab, languages=list(set(df['language'])))
    with open(args.output_json, 'w') as f:
        json.dump(data, f)
    print(f'Coverage: {len(token_stats) / len(tokenizer.vocab)}')


if __name__ == '__main__':
    main()
