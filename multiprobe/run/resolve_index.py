from getpass import getpass
import argparse
import os

from tqdm import tqdm

from multiprobe.data import WikipediaIndex, PagePropertiesDatabase
from multiprobe.utils import chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', type=str, required=True)
    parser.add_argument('--language', '-l', type=str, required=True)
    parser.add_argument('--pickled', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=200)
    parser.add_argument('--username', type=str, required=True)
    parser.add_argument('--password', '-p', action='store_true')
    parser.add_argument('--env-password', '-e', action='store_true')
    args = parser.parse_args()

    password = getpass() if args.password else None
    password = os.environ.get('MYSQL_PASSWORD') if args.env_password else password
    index = WikipediaIndex.from_dir(args.data_dir, args.language, True, pickled=args.pickled)
    db = PagePropertiesDatabase(args.username, password)
    num_resolved = 0
    num_total = 0
    for info_lst in tqdm(list(chunk(index.index_infos, args.chunk_size)), desc='Looking up entity IDs'):
        infos = list(filter(lambda x: x.entity_id is None, info_lst))
        num_total += len(info_lst)
        num_resolved += len(info_lst) - len(infos)
        if not infos:
            continue
        qids = db.bulk_find_qid(args.language, [info.page_id for info in infos])
        for info, qid in zip(infos, qids):
            info.entity_id = qid
        num_resolved += len(list(filter(lambda x: x is not None, qids)))
    print(f'{100 * num_resolved / num_total:.4}% total resolved entities.')
    index.save()


if __name__ == '__main__':
    main()