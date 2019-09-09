from dataclasses import dataclass
import urllib.parse

import requests


@dataclass
class JenaQueryApi(object):
    api_url: str

    def query(self, query):
        url = urllib.parse.urljoin(self.api_url, 'query')
        response = requests.post(url, data=dict(query=query)).json()
        return response['results']['bindings']


@dataclass
class WikidataApi(object):
    api_url: str = 'https://www.wikidata.org/w/api.php'

    def search_entities(self, entity, language):
        params = dict(action='wbsearchentities', search=entity, language=language, format='json', limit=1)
        response = requests.get(self.api_url, params=params).json()
        return response['search'][0]['id']


@dataclass
class WikipediaApi(object):
    api_url: str = 'https://{language}.wikipedia.org/w/api.php'

    def find_qids(self, page_names, language):
        url = self.api_url.format(language=language)
        json = requests.get(url, params=dict(action='query', prop='pageprops', ppprop='wikibase_item', redirects=1,
                                             titles='|'.join(page_names), format='json')).json()
        pages = json['query']['pages']
        qid_map = {x['title']: x['pageprops']['wikibase_item'] if 'pageprops' in x else None for x in pages.values()}
        for key in ('normalized', 'redirects'):
            try:
                qid_map.update({x['from']: x['to'] for x in json['query'][key]})
            except KeyError:
                pass
        return [qid_map[x] for x in page_names]


if __name__ == '__main__':
    print(WikipediaApi().find_qids(['AEX'], 'nl'))
