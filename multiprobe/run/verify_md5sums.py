import argparse
import os
import subprocess
import sys


def main():
    def check_files(name):
        name = name.strip()
        md5sum_file = f'{name}wiki-{args.date}-md5sums.txt'
        with open(os.path.join(args.data_dir, md5sum_file)) as f:
            md5_content = f.read()
        name = name.strip()
        for filename in files:
            filename = os.path.join(args.data_dir, filename.format(name=name, date=args.date))
            print(f'Checking {filename}... ', end='')
            md5_str = subprocess.check_output(f'md5sum {filename}', shell=True).decode()
            print('Okay' if md5_str.split()[0] in md5_content else "Not okay")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--date', type=str, default='latest')
    args = parser.parse_args()
    files = ('{name}wiki-{date}-pages-articles-multistream.xml.bz2',
             '{name}wiki-{date}-pages-articles-multistream-index.txt.bz2')
    for name in sys.stdin:
        try:
            check_files(name)
        except:
            pass


if __name__ == '__main__':
    main()