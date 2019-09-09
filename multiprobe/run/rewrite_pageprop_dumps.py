import argparse
import io
import gzip
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', '-l', type=str, required=True)
    parser.add_argument('--data-dir', '-d', type=str, required=True)
    parser.add_argument('--date', type=str, default='latest')
    args = parser.parse_args()
    print(f'Rewriting {args.language}...')

    pageprop_filename = os.path.join(args.data_dir, f'{args.language}wiki-{args.date}-page_props.sql.gz')
    f2 = io.BytesIO()
    with gzip.open(pageprop_filename) as f:
        f2.write(f.read().replace('`page_props`'.encode(), f'`{args.language}_page_props`'.encode()))
    with gzip.open(pageprop_filename, 'w') as f:
        f.write(f2.getvalue())


if __name__ == '__main__':
    main()