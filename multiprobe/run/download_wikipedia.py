import argparse
import os
import sys

import wget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--base-url', type=str, default='https://dumps.wikimedia.org/{lang}wiki/latest/')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for name in sys.stdin:
        name = name.strip()
        print(f'Downloading {name}...')
        multistream_url = args.base_url.format(lang=name) + f'{name}wiki-latest-pages-articles-multistream.xml.bz2'
        multistream_index_url = args.base_url.format(lang=name) + f'{name}wiki-latest-pages-articles-multistream-index.txt.bz2'
        try:
            wget.download(multistream_url, out=args.output_dir)
            wget.download(multistream_index_url, out=args.output_dir)
        except:
            pass


if __name__ == '__main__':
    main()