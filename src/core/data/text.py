import copy
import logging
import pathlib
from typing import Optional

import spacy

from src.nlp.negation.utils import FeaturesOfInterest


def validate_negspacy_config(config) -> dict:
    _val_dict = {
        "neg_termset": None,
        "feat_types": None,
        "extension_name": "negex",
        "chunk_prefix": None,
        "neg_termset_file": None,
        "feat_of_interest": FeaturesOfInterest.NOUN_CHUNKS,
        "scope": None,
        "language": None,
    }
    _foi = {
        "nc": FeaturesOfInterest.NOUN_CHUNKS,
        "ne": FeaturesOfInterest.NAMED_ENTITIES,
        "both": FeaturesOfInterest.BOTH,
    }
    _return_dict = {}

    for k, v in _val_dict.items():
        v_alt = (
            getattr(config, k, None)
            if not isinstance(config, dict)
            else config.get(k, None)
        )
        if k == "feat_of_interest":
            v_alt = (
                _foi.get(v_alt.lower(), FeaturesOfInterest.NOUN_CHUNKS)
                if isinstance(v_alt, str)
                else (
                    v_alt if isinstance(v_alt, list) else FeaturesOfInterest.NOUN_CHUNKS
                )
            )
        if v_alt is not None:
            _return_dict[k] = v_alt
        else:
            if v is None:
                continue
            _return_dict[k] = v
    return _return_dict


def clean_span(
    chunk: spacy.tokens.Span,
    offset: tuple[int, int] = None,
) -> Optional[dict]:
    _chunk_root_text = chunk.root.text.strip().replace("\t", "")
    _text, _lemma, _pos = [], [], []
    if offset is not None:
        _begin, _end = offset

    _i = 0
    _global_offset_first = chunk[0].idx
    # _last_token = None
    _add_begin_from_last = 0
    _remove_end_from_last = 0
    for i in range(len(chunk)):
        _token = chunk[i]
        _relative_token_offset = (
            _token.idx - _global_offset_first
        )  # -- '.idx' is char offset relative to the whole doc (here, the line)
        _space_between_next = 0
        # if i > 0:
        #     _last_token = chunk[i - 1]
        #     _len_last_token = len(_last_token.text)
        if i < len(chunk) - 1:
            _next_token = chunk[i + 1]
            _relative_token_offset_next = _next_token.idx - _global_offset_first
            _space_between_next = (
                _relative_token_offset_next - _relative_token_offset - len(_token.text)
            )

        _token_text = _token.text.strip().replace("\t", "")
        if (
            (not _token.is_alpha)
            or _token.is_stop
            or _token.is_punct
            or _token.like_num
            or _token.pos_ in ["DET", "PRON"]
            or _token_text.isspace()
        ):
            if offset is not None and i == _i and i < (len(chunk) - 1):
                _add_begin_from_last += len(_token.text) + _space_between_next
                _i += 1
            elif offset is not None and i == len(chunk) - 1:
                _remove_end_from_last += len(_token.text) + _space_between_next
        else:
            _text.append(_token_text)
            _lemma.append(_token.lemma_.strip().replace("\t", ""))
            _pos.append(_token.pos_)

    try:
        _head_idx = _text.index(_chunk_root_text)
        return {
            "head_idx": _head_idx,
            "lemma": _lemma,
            "text": _text,
            "pos": _pos,
            "offset": (
                (
                    _begin + _add_begin_from_last,
                    _end - _remove_end_from_last,
                )
                if offset is not None
                else None
            ),
        }
    except ValueError:
        return None


def get_actual_str(
    chunk_dict: dict, modify_key: tuple, case_sensitive: bool = False
) -> str:
    prepend_head, use_lemma, head_only = modify_key
    if use_lemma:
        _key_str = "lemma"
    else:
        _key_str = "text"

    _head_idx: int = chunk_dict["head_idx"]
    _text: str
    if head_only:
        _text = chunk_dict[_key_str][_head_idx]
    elif prepend_head:
        _clone = copy.copy(chunk_dict[_key_str])
        _head, _tail = _clone.pop(_head_idx), _clone
        _text = " ".join([_head] + _tail)
    else:
        _text = " ".join(chunk_dict[_key_str])

    return _text if case_sensitive else _text.lower()


if __name__ == "__main__":
    from src.core.data.factory import DataProcessingFactory

    # _data = None
    # _zip_path = pathlib.Path("C:/Users/fra3066mat/Documents/Corpora/grascco.zip")
    # if not _zip_path.exists():
    #     sys.exit("zip doesnt exist")
    # with zipfile.ZipFile(_zip_path.resolve(), mode='r') as archive:
    #     _data = [{"name": pathlib.Path(f.filename).stem,
    #               "content": archive.read(f.filename).decode('utf-8'),
    #               "label": None
    #               } for f in archive.filelist if
    #              (not f.is_dir()) and (pathlib.Path(f.filename).suffix.lstrip('.') == 'txt')]
    # data_proc = DataProcessingFactory.create(
    #     pipeline=spacy.load("de_dep_news_trf"),
    #     base_data=_data[5:10],
    #     save_to_file=False,
    #     negspacy_config={
    #         "neg_termset_file": "C:/Users/fra3066mat/PycharmProjects/concept-graphs/conf/negex_files/negex_trigger_german_biotxtm_2016_extended.txt",
    #         "chunk_prefix": ["kein", "keine", "keinen"],
    #         "scope": 2
    #     }
    # )

    data_proc = DataProcessingFactory.load(
        pathlib.Path(
            "C:/Users/fra3066mat/PycharmProjects/concept-graphs/tmp/grascco_test/grascco_test_data.pickle"
        )
    )
    res = data_proc.process_external_docs(
        [
            {
                "name": "test",
                "content": "Das ist ein Test Dokument. Es hat ein paar Nomen Chunks; vielleicht. Auch gab es keinen Nachweis für einen Sinn. Aber es wurd ein Sinn nicht gesehen.",
                "label": None,
            }
        ]
    )
    logging.info("%s", res)
