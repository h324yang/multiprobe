import argparse
import time

from tqdm import tqdm

from multiprobe.data.wikidump import WikipediaIndex
from multiprobe.data.wikidata import WikipediaApi
from multiprobe.utils import chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', type=str, required=True)
    parser.add_argument('--language', '-l', type=str, required=True)
    parser.add_argument('--pickled', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=50)
    parser.add_argument('--delay', type=int, default=1)
    args = parser.parse_args()

    index = WikipediaIndex.from_dir(args.data_dir, args.language, True, pickled=args.pickled)
    api = WikipediaApi()
    delay = args.delay / 1000
    for info_lst in tqdm(list(chunk(index.index_infos, args.chunk_size))):
        empty_infos = filter(lambda x: x.entity_id is None, info_lst)
        page_names = [x.page_name for x in empty_infos]
        print(api.find_qids(page_names, args.language))
        time.sleep(delay)


if __name__ == '__main__':
    main()