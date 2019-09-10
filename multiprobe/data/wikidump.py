from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional
import bz2
import html
import pickle
import re
import os

from lxml import etree
from tqdm import tqdm

from .wikiclean import clean


@dataclass
class IndexInfo(object):
    page_name: str
    page_id: int
    offset: int
    entity_id: Optional[str]


@dataclass
class WikipediaPagePropertyDatabase(object):
    sqlite_filename: str


@dataclass
class WikipediaPage(object):
    root: etree.Element

    def __post_init__(self):
        self.text = self.root.find('revision').find('text').text
        self.title = self.root.find('title').text
        self.id = int(self.root.find('id').text)
        self.raw_text = etree.tostring(self.root).decode()

    @property
    def clean_text(self):
        if not hasattr(self, 'clean_text_'):
            self.clean_text_ = html.unescape(clean(self.text))
        return self.clean_text_

    def cleaned_text(self, **kwargs):
        return html.unescape(clean(self.text, **kwargs))

    @classmethod
    def from_string(cls, xml_str):
        return cls(etree.fromstring(xml_str))


@dataclass
class WikipediaIndex(object):
    path: str
    language: str
    page_name_map: Dict[str, IndexInfo]
    page_id_map: Dict[int, IndexInfo]
    slice_map: Dict[int, List[IndexInfo]]
    slice_next_map: Dict[int, int]

    @property
    def index_infos(self):
        return list(self.page_name_map.values())

    @classmethod
    def from_dir(cls, folder, language, use_tqdm=False, pickled=False):
        page_name_map = {}
        page_id_map = {}
        slice_map = defaultdict(list)
        slice_next_map = {}

        path = os.path.join(folder, f'{language}wiki-latest-pages-articles-multistream-index.txt.bz2')
        if pickled:
            with open(path + '.pkl', 'rb') as f:
                return pickle.load(f)
        prev_offset = -1
        with bz2.open(path) as f:
            for line in tqdm(f, disable=not use_tqdm):
                line = line.decode().strip()
                offset, page_id, name = line.split(':', 2)
                index_info = IndexInfo(name, int(page_id), int(offset), None)
                page_name_map[name] = index_info
                page_id_map[index_info.page_id] = index_info
                slice_map[index_info.offset].append(index_info)
                slice_next_map[index_info.offset] = -1
                if prev_offset > 0:
                    slice_next_map[prev_offset] = index_info.offset
                prev_offset = index_info.offset
        return cls(path, language, page_name_map, page_id_map, slice_map, slice_next_map)

    def save(self, filename=None):
        if filename is None: filename = self.path
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(self, f)

    def find(self, page_id=None, page_name=None):
        if page_id is not None:
            return self.page_id_map[page_id]
        else:
            return self.page_name_map[page_name]

    def next_slice(self, offset):
        return self.slice_next_map[offset]


def parse_chunk(xml):
    pages = re.findall('(<page>.+?</page>)', xml, flags=re.DOTALL)
    return list(map(WikipediaPage, map(etree.fromstring, pages)))


class WikipediaLoader(object):

    def __init__(self, index):
        self.index = index
        self.path = os.path.join(os.path.dirname(index.path), f'{index.language}wiki-latest-pages-articles-multistream.xml.bz2')

    def load_single_slice(self, offset):
        with open(self.path, 'rb') as f:
            f.seek(offset)
            next_slice_offset = self.index.next_slice(offset)
            data = f.read(next_slice_offset - offset) if next_slice_offset > 0 else f.read()
        return parse_chunk(bz2.BZ2File(BytesIO(data)).read().decode())

    def load_batch_slices(self, offsets):
        offsets = sorted(list(set(offsets)))
        with open(self.path, 'rb') as f:
            base = 0
            for offset in offsets:
                f.seek(offset - base, 1)
                next_slice_offset = self.index.next_slice(offset)
                data = f.read(next_slice_offset - offset) if next_slice_offset > 0 else f.read()
                base = offset + len(data)
                try:
                    yield parse_chunk(bz2.BZ2File(BytesIO(data)).read().decode())
                except EOFError:
                    return

    def load_single(self, **kwargs):
        info = self.index.find(**kwargs)
        pages = self.load_single_slice(info.offset)
        for page in pages:
            if page.title == kwargs.get('page_name') or page.id == kwargs.get('page_id'):
                return page

    def load_batch(self, index_infos):
        offsets = [info.offset for info in index_infos]
        page_ids = set([info.page_id for info in index_infos])
        pages = []
        for chunk in self.load_batch_slices(offsets):
            pages.extend(list(filter(lambda x: x.id in page_ids, chunk)))
        return pages
