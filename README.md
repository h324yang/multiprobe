# Multiprob
A tool for generating massive parallel corpus with Wikidata.

# Steps

1. extract descriptions
```
python extract_top_desc.py
```

2. generate parallel triples (center_sent, pos_sent, neg_sent)
```
python gen_train.py
```