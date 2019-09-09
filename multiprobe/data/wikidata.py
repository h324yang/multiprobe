from dataclasses import dataclass
import urllib.parse

import mysql.connector as mconn
import requests


@dataclass
class PagePropertiesDatabase(object):
    mysql_username: str
    mysql_password: str
    mysql_host: str = 'localhost'
    mysql_database: str = 'wikipedia'

    def __post_init__(self):
        self.conn = mconn.connect(host=self.mysql_host,
                                  user=self.mysql_username,
                                  passwd=self.mysql_password,
                                  database=self.mysql_database)
        self.cursor = self.conn.cursor(prepared=True)

    def find(self, language, page_id, page_property):
        self.cursor.execute(f'SELECT pp_value FROM {language}_page_props WHERE pp_propname=%s AND pp_page=%s LIMIT 1',
                            (page_property, page_id))
        row = self.cursor.fetchone()
        if not row:
            return None
        return row[0]

    def bulk_find(self, language, page_ids, page_property):
        bulk_fmt = ','.join(["%s"] * len(page_ids))
        self.cursor.execute(f'SELECT pp_page, pp_value FROM {language}_page_props WHERE pp_propname=%s AND pp_page IN ({bulk_fmt}) LIMIT {len(page_ids)}',
                            (page_property,) + tuple(page_ids))
        rows = self.cursor.fetchall()
        data = {row[0]: row[1] for row in rows}
        return list(map(data.get, page_ids))

    def find_qid(self, language, page_id):
        return self.find(language, page_id, 'wikibase_item')

    def bulk_find_qid(self, language, page_ids):
        return self.bulk_find(language, page_ids, 'wikibase_item')


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
