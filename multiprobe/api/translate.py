from dataclasses import dataclass


@dataclass
class GoogleTranslateApi(object):
    base_url: str = 'POST https://translation.googleapis.com/language/translate/v2'

