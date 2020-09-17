# Multiprob
A tool for generating massive parallel corpus with Wikidata.

## Steps
1. extract descriptions
```
python extract_top_desc.py
```

2. generate parallel triples (center_sent, pos_sent, neg_sent), e.g., [this](top_5k_meta/triples.jsonl).
```
python gen_train.py
```
    *By default, it randomly picks three languages to generate triples each time. If you need to fix the center language, change the constant `FIXED_CENTER_LANG` inside to a specific language, e.g., "en"