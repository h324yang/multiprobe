from multiprobe.data import WikipediaIndex, WikipediaLoader
import re
import random
import json
from collections import defaultdict


def get_loader(lang):
    index = WikipediaIndex.from_dir("/home/shared/multiwiki/", f"{lang}", False, pickled=True)
    index.path = f'/home/shared/multiwiki/{lang}wiki-latest-pages-articles-multistream-index.txt.bz2'
    loader = WikipediaLoader(index)
    return loader

def get_eid_map(loader):
    entity_id_map = {}
    for pid, info in loader.index.page_id_map.items():
        if info.entity_id:
            entity_id_map[info.entity_id] = pid
    return entity_id_map

def isDirty(text, min_len=30):
    if "http:" in text or "https:" in text or len(text) < min_len:
        return True
    else:
        return False

def postClean(text):
    text = text.replace('\t', ' ')
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    return text

def read_langs(fn="data/mbert-languages"):
    with open(fn) as f:
        langs = f.read().split("\n")[:-1]
        return langs

def read_top_eids(fn="top_5k.log"):
    with open(fn) as f:
        eids = []
        for line in f:
            eids.append(line.strip().split(",")[0])
    return eids

def output_desc(out_fn, langs, eids):
    with open(out_fn, "w") as f:
        for lang in langs:
            try:
                loader = get_loader(lang)
            except:
                continue

            entity_id_map = {}
            for pid, info in loader.index.page_id_map.items():
                entity_id_map[info.entity_id] = pid

            for eid in eids:
                try:
                    pid = entity_id_map[eid]
                    page = loader.load_single(page_id=pid)
                    text = page.cleaned_text(remove_headings=False).split('\n', 1)[0].strip()
                    if isDirty(text):
                        continue
                    text = postClean(text)
                    f.write(f"{lang}-{eid}-{pid}\t{text}\n")
                except:
                    pass

def read_desc(fn="top_5k.desc"):
    corpus = defaultdict(dict)
    with open(fn, "r") as f:
        for line in f:
            tid, text = line.strip().split("\t")
            lang, eid, pid = tid.split("-")
            corpus[eid][lang] = text
    return corpus

class GUIDMap:
    def __init__(self):
        self.count = 0
        self.map = {}

    def take(self, old):
        if old in self.map:
            return self.map[old]
        else:
            self.map[old] = self.count
            self.count += 1
            return self.count - 1

    def write(self, fn):
        with open(fn, "w") as f:
            for k,v in self.map.items():
                f.write(f"{v} {k}\n")


def write_candidate(candi, out, prefix=""):
    with open(out, "w") as f:
        if candi:
            f.write(json.dumps({"candidate":[prefix+c for c in candi]})+"\n")
        else:
            f.write(json.dumps({"candidate":None}))


def write_test_data(center, loader, eid_map, out, size, prefix=""):
    guid_map = GUIDMap()
    left = size
    with open(out, "w") as f:
        for i, eid in enumerate(center):
            if left <= 0:
                break
            try:
                pid = eid_map[eid]
                page = loader.load_single(page_id=pid)
                text = page.cleaned_text(remove_headings=False).split('\n', 1)[0].strip()
                if "refer to" in text or isDirty(text, 30):
                    continue
                guid = prefix+eid
                guid = guid_map.take(guid)
                example = {"guid":guid, "text":text}
                f.write(json.dumps(example)+"\n")
                left -= 1
                print("{} sample(s) left...".format(left))
            except:
                pass
    guid_map.write(out+".idmap")


def sort_by_popularity(eids, loader, eid_map):
    sent_len = {}
    for eid in eids:
        try:
            pid = eid_map[eid]
            page = loader.load_single(page_id=pid)
            size = len(page.cleaned_text(remove_headings=False))
            sent_len[eid] = size
        except:
            sent_len[eid] = 0
    sorted_list = [tup[0] for tup in sorted(sent_len.items(), key=lambda x:x[1], reverse=True)]
    return sorted_list


