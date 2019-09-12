from collections import defaultdict
import argparse
import glob
import os

from tqdm import tqdm
import pandas as pd

from multiprobe.data import WikipediaPage
from multiprobe.utils import find_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', type=str, required=True)
    parser.add_argument('--output-file', '-o', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default='bert_multilingual', choices=['bert_multilingual', 'bert_base', 'nltk'])
    parser.add_argument('--min-length', type=int, default=10)
    args = parser.parse_args()

    xml_files = glob.glob(os.path.join(args.data_dir, '*.xml'))
    tokenizer = find_tokenizer(args.tokenizer)
    data = defaultdict(list)
    pbar = tqdm(xml_files)
    num_skipped = 0
    for xml_file in pbar:
        with open(xml_file) as f:
            content = f.read()
        lang, _ = os.path.basename(xml_file).split('-')
        page = WikipediaPage.from_string(content)
        try:
            text = page.cleaned_text(remove_headings=False).split('\n', 1)[0].strip()
        except:
            text = ''
        if len(text) == 0 or text[0] == '=' or len(text) < args.min_length:
            num_skipped += 1
            pbar.set_postfix(skipped=str(num_skipped))
            continue
        data['language'].append(lang)
        data['abstract'].append(' '.join(tokenizer(text.replace('\t', ' '))))
    pd.DataFrame(data).to_csv(args.output_file, quoting=3, sep='\t', index=False)


if __name__ == '__main__':
    main()