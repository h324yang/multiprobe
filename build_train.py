import re
import random
import json
from collections import defaultdict
from util import *

TOP_LANG = ["en", "zh", "fr", "ja", "es", "de", "ru", "it"] # , "ceb", "pl"]
TEST_SIZE = 100

def gen_train(corpus, out_fn, train_size=300, hard_neg=False):
    left = train_size
    eids = read_top_eids()
    langs = TOP_LANG
    f = open(out_fn, "w")
    guid_map = GUIDMap()
    while left:
        print(f"{train_size-left}/{train_size}...")
        try:
            pos, neg = random.sample(eids, k=2)
            p_langs = set(corpus[pos].keys()).intersection(langs)
            n_langs = set(corpus[neg].keys()).intersection(langs)
            lang_c, lang_p = random.sample(p_langs, k=2)
            lang_n = random.sample(n_langs, k=1)[0]
            cid = f"{pos}-{lang_c}"
            c_text = corpus[pos][lang_c]
            cid = guid_map.take(cid)
            pid = f"{pos}-{lang_p}"
            p_text = corpus[pos][lang_p]
            pid = guid_map.take(pid)
            nid = f"{neg}-{lang_n}"
            n_text = corpus[neg][lang_n]
            nid = guid_map.take(nid)
            example = {"guid": cid, "text": c_text, "p_guid": pid, "text_p": p_text, "n_guid": nid, "text_n": n_text}
            f.write(json.dumps(example)+"\n")
            left -= 1
        except:
            continue
    f.close()
    print("Generating idmap...")
    guid_map.write(out_fn+".idmap")

def main():
    # output_desc("top_5k.desc", read_langs, read_top_eids)
    corpus = read_desc()
    gen_train(corpus, "train.json")

if __name__ == "__main__":
    main()


