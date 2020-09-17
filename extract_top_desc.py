from multigen.util import (
    extract_desc, 
    read_langs, 
    read_top_eids
)


def main():
    # top k lang-covering entities
    top_entities = read_top_eids("top_5k_meta/top_5k.log")
    mbert_langs = read_langs("data/mbert-languages")
    extract_desc("top_5k_meta/top_5k_desc.txt", mbert_langs, top_entities)


if __name__ == "__main__":
    main()