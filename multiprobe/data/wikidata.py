from dataclasses import dataclass

import mysql.connector as mconn


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
