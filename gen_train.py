import re
import random
import json
from collections import defaultdict
from multigen.util import (
    read_desc, 
    GuidAlloc, 
    read_top_eids
)


TOP_LANG = ["en", "zh", "fr", "ja", "es", "de", "ru", "it"] # "ceb", "pl"
NUM_TRIPLES = 300
OUT_FN = "top_5k_meta/triples.jsonl"
EID_LOG = "top_5k_meta/top_5k.log"
CORPUS_FN = "top_5k_meta/top_5k.desc"
REPORT_EVERY = 100


def main():
    print("Read descriptions...")
    corpus = read_desc(CORPUS_FN)
    guid_alloc = GuidAlloc()
    num_triples = NUM_TRIPLES
    f = open(OUT_FN, "w")
    print("Read the entity list...")
    eids = read_top_eids(EID_LOG)
    print("Start generating...")
    while num_triples > 0:
        try:
            pos_e, neg_e = random.sample(eids, k=2)
            pos_langs = set(corpus[pos_e].keys()).intersection(TOP_LANG)
            neg_langs = set(corpus[neg_e].keys()).intersection(TOP_LANG)
            cur_center_lang, cur_pos_lang = random.sample(pos_langs, k=2)
            cur_neg_lang = random.sample(neg_langs, k=1)[0]
            center_guid = f"{pos_e}-{cur_center_lang}"
            center_text = corpus[pos_e][cur_center_lang]
            center_record_id = guid_alloc.check(center_guid)
            pos_guid = f"{pos_e}-{cur_pos_lang}"
            pos_text = corpus[pos_e][cur_pos_lang]
            pos_record_id = guid_alloc.check(pos_guid)
            neg_guid = f"{neg_e}-{cur_neg_lang}"
            neg_text = corpus[neg_e][cur_neg_lang]
            neg_record_id = guid_alloc.check(neg_guid)
            
            example = {
                "center_guid": center_guid, 
                "center_text": center_text, 
                "center_record_id": center_record_id, 
                "pos_guid": pos_guid, 
                "pos_text": pos_text, 
                "pos_record_id": pos_record_id, 
                "neg_guid": neg_guid,
                "neg_text": neg_text,
                "neg_record_id": neg_record_id
            }
            
            f.write(json.dumps(example)+"\n")
            num_triples -= 1
            
            if ((NUM_TRIPLES - num_triples) % REPORT_EVERY) == 0:
                print(f"{NUM_TRIPLES - num_triples} triples are generated.")
            
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    main()