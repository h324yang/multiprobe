from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List
import json

from scipy.spatial import distance as spd
from torch.distributions.categorical import Categorical
import langdetect
import numpy as np
import torch
import yaml


class LanguageFamilyData(object):

    def __init__(self, family_map):
        self.family_map = family_map
        self.code_family_map = defaultdict(lambda: 'none')
        for family, languages in family_map.items():
            for lang, data in languages.items():
                self.code_family_map[data['code']] = family

    @property
    def families(self):
        return list(self.family_map.keys())

    def find_family(self, lang_code):
        return self.code_family_map[lang_code]

    @classmethod
    def from_yaml(cls, filename):
        with open(filename) as f:
            family_map = yaml.load(f)
        return cls(family_map)


def sum_js(p_list, q_list):
    dist = 0
    for p, q in zip(p_list, q_list):
        js_dist = spd.jensenshannon(p, q, 2.0)
        # SciPy has numerical issues
        if np.isnan(js_dist):
            js_dist = 0
        dist += js_dist
    return dist


@dataclass
class TokenLanguageStatistics(object):
    data: Dict[str, Dict[str, int]]
    vocab: Dict[str, int]
    languages: List[str]

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            data = json.load(f)
        languages = sorted(data['languages'])
        return cls(data['statistics'], data['vocab'], languages)

    def compute_distribution(self, tokens, weights=None):
        probs = []
        new_weights = []
        for token, weight in zip(tokens, weights):
            token_probs = np.zeros(len(self.languages))
            if token not in self.data:
                continue
            for language, count in self.data[token].items():
                token_probs[self.languages.index(language)] += count
            if np.sum(token_probs) > 0:
                token_probs = token_probs / np.sum(token_probs)
                probs.append(token_probs)
                new_weights.append(weight)
        if len(probs) == 0:
            raise ValueError('No counts found for those tokens.')
        if weights is None:
            probs = np.mean(np.array(probs), 0)
        else:
            weights = np.array(new_weights)
            probs = np.array(probs)
            probs = np.sum(np.repeat(np.expand_dims(weights, 1), probs.shape[1], 1) * probs, 0) / np.sum(weights)
        return Categorical(probs=torch.Tensor(probs))


@lru_cache(maxsize=1000000)
def detect_language(string):
    return langdetect.detect(string)
