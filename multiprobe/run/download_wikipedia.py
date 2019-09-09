import argparse
import os
import sys

import wget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--base-url', type=str, default='https://dumps.wikimedia.org/{lang}wiki/latest/')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--date', type=str, default='latest')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    files = ('{name}wiki-{date}-pages-articles-multistream.xml.bz2',
             '{name}wiki-{date}-pages-articles-multistream-index.txt.bz2',
             '{name}wiki-{date}-page_props.sql.gz')
    for name in sys.stdin:
        name = name.strip()
        print(f'Downloading {name}...')
        for filename in files:
            filename = filename.format(name=name, date=args.date)
            url = args.base_url.format(lang=name) + filename
            if not os.path.isfile(os.path.join(args.output_dir, filename)) or args.overwrite:
                try:
                    wget.download(url, out=args.output_dir)
                except:
                    pass


if __name__ == '__main__':
    main()