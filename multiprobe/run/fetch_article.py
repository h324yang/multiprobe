import argparse

from multiprobe.data.wikidump import WikipediaIndex, WikipediaLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('--data-dir', '-d', type=str, required=True)
    parser.add_argument('--language', '-l', type=str, default='en')
    args = parser.parse_args()

    index = WikipediaIndex.from_dir(args.data_dir, args.language, use_tqdm=True)
    print(WikipediaLoader(index).load_single(page_name=args.name).raw_text)


if __name__ == '__main__':
    main()