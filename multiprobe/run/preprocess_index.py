import argparse

from multiprobe.data.wikidump import WikipediaIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', type=str, required=True)
    parser.add_argument('--language', '-l', type=str, required=True)
    args = parser.parse_args()

    index = WikipediaIndex.from_dir(args.data_dir, args.language, True)
    index.save()
    print(f'Saved {args.language} pickle file.')


if __name__ == '__main__':
    main()