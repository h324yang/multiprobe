#!/bin/sh
python -m multiprobe.run.fetch_bert_languages > data/mbert-languages
cat data/mbert-languages | python -m multiprobe.run.download_wikipedia --output-dir data/$(date +%Y%m%d)