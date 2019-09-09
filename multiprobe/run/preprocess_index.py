import argparse
import pickle

from multiprobe.data.wikidump import WikipediaIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', type=str, required=True)
    parser.add_argument('--language', '-l', type=str, required=True)
    args = parser.parse_args()

    index = WikipediaIndex.from_dir(args.data_dir, args.language, True)
    out_path = f'{index.path}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(index, f)
    print(f'Wrote to {out_path}')


if __name__ == '__main__':
    main()