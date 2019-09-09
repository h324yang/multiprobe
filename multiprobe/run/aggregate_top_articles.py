from collections import Counter
import argparse

from tqdm import tqdm

from multiprobe.data import WikipediaIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', type=str, required=True)
    parser.add_argument('--languages', '-l', type=str, nargs='+', required=True)
    parser.add_argument('--num-top', type=int, default=5000)
    args = parser.parse_args()

    entity_counter = Counter()
    for language in tqdm(args.languages):
        try:
            index = WikipediaIndex.from_dir(args.data_dir, language, False, pickled=True)
        except FileNotFoundError:
            continue
        seen_set = set()
        for index in index.index_infos:
            if index.entity_id is not None and index.entity_id not in seen_set:
                entity_counter[index.entity_id] += 1
                seen_set.add(index.entity_id)

    sorted_top = sorted(entity_counter.items(), key=lambda x: x[1], reverse=True)[:args.num_top]
    print('\n'.join(map(lambda x: ','.join((x[0], str(x[1]))), sorted_top)))


if __name__ == '__main__':
    main()