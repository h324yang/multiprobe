import argparse

import yaml

from .fetch_bert_languages import fetch_wiki_language_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', type=str, required=True)
    args = parser.parse_args()

    lang_map = fetch_wiki_language_map()
    with open(args.data_file) as f:
        data = yaml.load(f)
    for family in data:
        for language in data[family]:
            code = lang_map.get(language.replace('_', ' '))
            data[family][language]['code'] = 'none' if code is None else code
    with open(args.data_file, 'w') as f:
        yaml.dump(data, f)


if __name__ == '__main__':
    main()
