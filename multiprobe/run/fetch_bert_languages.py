import re

import requests


def fetch_wiki_language_map():
    wiki_url = 'https://en.wikipedia.org/wiki/List_of_Wikipedias'
    content = requests.get(wiki_url).content.decode().split('<span class="mw-headline" id="Detailed_list">Detailed list</span>')[1]
    lang_map = dict(re.findall(r'<tr style=""><td><a.+?>(.+?)</a>.+?https://(.+?).wikipedia.org', content))
    return lang_map


def main():
    bert_url = 'https://raw.githubusercontent.com/google-research/bert/master/multilingual.md'
    content = requests.get(bert_url).content.decode()
    m = re.match(r'^.*List of Languages.*?(\n\*.+)\n[^*].*$', content, flags=re.DOTALL)
    content_lang = m.group(1)
    languages = re.findall('\*\s+(.+)', content_lang)
    lang_map = fetch_wiki_language_map()
    for lang in languages:
        if lang == 'Chinese (Simplified)':
            lang = 'Chinese'
        elif lang == 'Chinese (Traditional)':
            lang = 'Classical Chinese'
        elif lang == 'Norwegian (Bokmal)':
            lang = 'Norwegian (Bokmål)'
        elif lang == 'Persian (Farsi)':
            lang = 'Persian'
        elif lang == 'Punjabi':
            lang = 'Western Punjabi'
        elif lang == 'South Azerbaijani':
            lang = 'Southern Azerbaijani'
        elif lang == 'Waray-Waray':
            lang = 'Waray'
        print(lang_map[lang])


if __name__ == '__main__':
    main()
