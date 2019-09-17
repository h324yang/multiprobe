from dataclasses import dataclass
from urllib.parse import urljoin

import requests


@dataclass
class GoogleTranslateApi(object):
    access_token: str
    base_url: str = 'POST https://translation.googleapis.com/language/translate/v2/'

    def detect(self, query):
        url = urljoin(self.base_url, 'detect')
        data = requests.post(url, json=dict(q=query)).json()
        return data['data']['detections']


