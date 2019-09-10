from functools import lru_cache

from pytorch_transformers import BertTokenizer
import nltk


@lru_cache(maxsize=100)
def _bert_tokenizer(name):
    return BertTokenizer.from_pretrained(name)


@lru_cache(maxsize=100000)
def multilingual_bert_tokenize(sentence):
    return _bert_tokenizer('bert-base-multilingual-uncased').tokenize(sentence)


@lru_cache(maxsize=100000)
def bert_base_tokenize(sentence):
    return _bert_tokenizer('bert-base-uncased').tokenize(sentence)


def nltk_tokenize(sentence):
    return nltk.word_tokenize(sentence)


def find_tokenizer(name):
    return TOKENIZERS[name]


TOKENIZERS = dict(bert_base=bert_base_tokenize,
                  bert_multilingual=multilingual_bert_tokenize,
                  nltk=nltk_tokenize)