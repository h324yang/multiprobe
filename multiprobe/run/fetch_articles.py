import argparse
import os

from tqdm import tqdm

from multiprobe.data.wikidump import WikipediaIndex, WikipediaLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, default='output')
    parser.add_argument('--entity-ids', '-e', type=str, nargs='+', required=True)
    parser.add_argument('--language', '-l', type=str, default='en')
    args = parser.parse_args()

    print(args.language)
    os.makedirs(args.output_dir, exist_ok=True)
    index = WikipediaIndex.from_dir(args.data_dir, args.language, use_tqdm=True, pickled=True)
    entity_ids = set(args.entity_ids)
    loader = WikipediaLoader(index)
    index_infos = list(filter(lambda x: x.entity_id in entity_ids, index.index_infos))
    pages = loader.load_batch(index_infos)
    for page in tqdm(pages):
        with open(os.path.join(args.output_dir, f'{args.language}-{page.id}.xml'), 'w') as f:
            f.write(page.raw_text)


if __name__ == '__main__':
    main()