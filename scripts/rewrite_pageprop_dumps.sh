#!/bin/sh
for lang in $(cat data/mbert-languages); do python -m multiprobe.run.rewrite_pageprop_dumps -l $lang -d DATA_DIR; done
for f in DATA_DIR/*.gz; do zcat $f | mysql -u ralph wikipedia; done