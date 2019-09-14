from collections import defaultdict

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