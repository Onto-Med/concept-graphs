import copy
import io
import logging
import os
import pathlib
import re
import itertools
import zipfile
from collections import defaultdict
from random import sample

import numpy as np
from spacy.tokens.doc import Doc
from tqdm.autonotebook import tqdm
from typing import Optional, Generator, Union, Iterable, Dict, List, Set, Callable, Any

import spacy
from functools import lru_cache

from spacy import Language
from sklearn.feature_extraction.text import TfidfVectorizer as tfidfVec

from src.negspacy.utils import FeaturesOfInterest
from src.negspacy.negation import Negex
from util_functions import load_pickle, save_pickle


# ToDo: this needs to be called whenever a data_proc object is used/loaded by another class
def _set_extensions(
) -> None:
    from spacy.tokens import Doc
    if not Doc.has_extension("doc_id"):
        Doc.set_extension("doc_id", default=None)
    if not Doc.has_extension("doc_index"):
        Doc.set_extension("doc_index", default=None)
    if not Doc.has_extension("doc_name"):
        Doc.set_extension("doc_name", default=None)
    if not Doc.has_extension("doc_topic"):
        Doc.set_extension("doc_topic", default=None)


class DataProcessingFactory:

    @staticmethod
    def load(
            file_path: Union[pathlib.Path, str, io.IOBase]
    ) -> 'DataProcessing':
        _set_extensions()
        return load_pickle(file_path)

    @staticmethod
    def create(
            pipeline: Language,
            base_data: Union[pathlib.Path, Iterable[Dict[str, str]]],
            labels: Optional[Union[dict, Callable]] = None,
            sub_paths: Optional[list] = None,
            cache_path: Optional[pathlib.Path] = None,
            cache_name: Optional[str] = None,
            n_process: int = 1,
            file_encoding: str = 'utf-8',
            file_extension: Optional[str] = 'txt',
            save_to_file: bool = True,
            subset: Optional[int] = None,
            use_lemma: bool = False,
            prepend_head: bool = False,
            head_only: bool = False,
            case_sensitive: bool = False,
            filter_min_df: Union[int, float] = 1,
            filter_max_df: Union[int, float] = 1.,
            filter_stop: Optional[list] = None,
            disable: Optional[Iterable[str]] = None,
            categories: Optional[list] = None,
            omit_negated_chunks: bool = True,
            negspacy_config: Optional[dict] = None
    ):
        def _get_label_from_file(
                fi: pathlib.Path
        ) -> str:
            _label = None
            if isinstance(labels, dict):
                _label = labels.get(fi.stem, "None")
            elif isinstance(labels, Callable):
                _label = labels(fi.stem)
            return _label

        def _file_like_data_entries(
                base_path: pathlib.Path,
                sub_path: List[str],
                file_ext: str,
                sub: Optional[int] = None,
                target_categories: Optional[list] = None
        ) -> Generator[Dict[str, Optional[str]], None, None]:
            _iter = itertools.chain(
                *[(base_path / pathlib.Path(i)).glob(f"*.{file_ext}" if file_ext is not None else "*")
                  for i in sub_path]) if len(sub_path) > 0 else itertools.chain(base_path.glob(f"*.{file_ext}"))

            labels_dict = defaultdict(list)
            if target_categories is not None:
                target_categories = [c.lower() for c in target_categories]
            for _f in _iter:
                if _f.is_dir():
                    continue
                _label = _get_label_from_file(_f).lower()
                if target_categories is None or _label in target_categories:
                    labels_dict[_label].append(_f)

            if sub is not None:
                _doc_per_label = int(sub / len(labels_dict.keys()))
                _iter = itertools.chain(*[sample(s, _doc_per_label) for s in labels_dict.values()])
            else:
                _iter = itertools.chain(*list(labels_dict.values()))

            for fi in _iter:
                if fi.is_dir():
                    continue
                _label = _get_label_from_file(fi)
                yield {"name": fi.stem, "content": fi.read_text(encoding=file_encoding), "label": _label}

        _base_path = base_data.resolve() if isinstance(base_data, pathlib.Path) else None
        _sub_path = [] if sub_paths is None else sub_paths
        _cache_path = ((pathlib.Path(os.getcwd()) / pathlib.Path("cache")).absolute()
                       if cache_path is None else cache_path.resolve())
        _file_encoding = file_encoding
        _file_ext = file_extension
        _cache_name = cache_name if cache_name is not None else (
            _base_path.name if _base_path is not None else "processed_data")

        _data_processing = DataProcessingFactory.DataProcessing(
            pipeline=pipeline,
            data_entries=(_file_like_data_entries(_base_path, _sub_path, _file_ext, subset, categories)
                          if _base_path is not None else base_data),
            file_encoding=file_encoding,
            n_process=n_process,
            use_lemma=use_lemma,
            prepend_head=prepend_head,
            head_only=head_only,
            case_sensitive=case_sensitive,
            filter_min_df=filter_min_df,
            filter_max_df=filter_max_df,
            filter_stop=filter_stop,
            disable=disable,
            omit_negated_chunks=omit_negated_chunks,
            negspacy_config=negspacy_config
        )

        if save_to_file:
            delattr(_data_processing, '_data_entries')  # remove as it's not needed and makes problems when serializing
            final_cache = pathlib.Path(_cache_path / pathlib.Path(f"{_cache_name}.pickle"))
            save_pickle(_data_processing, final_cache)
        return _data_processing

    class DataProcessing:
        def __init__(
                self,
                pipeline: Language,
                data_entries: Iterable[Dict[str, str]],
                n_process: int = 1,
                file_encoding: str = 'utf-8',
                use_lemma: bool = False,
                prepend_head: bool = False,
                head_only: bool = False,
                case_sensitive: bool = False,
                filter_min_df: Union[int, float] = 1,
                filter_max_df: Union[int, float] = 1.,
                filter_stop: Optional[list] = None,
                disable: Optional[Iterable[str]] = None,
                omit_negated_chunks: bool = True,
                negspacy_config: Optional[dict] = None
        ) -> None:
            self._data_entries = [d for d in data_entries]
            self._file_encoding = file_encoding
            self._prepend_head = prepend_head
            self._use_lemma = use_lemma
            self._head_only = head_only
            self._data_corpus_tuples = list()
            self._text_id_to_doc_name = dict()
            self._processed_docs = list()
            self._document_chunk_matrix = list()
            self._chunk_set_dicts = list()
            self._true_labels = list()
            self._true_labels_dict = dict()
            self._view = None
            self._cache_obj = [self._document_list, self.get_document_by_id, self.get_document_by_name,
                               self.get_document_ids_by_topic, self.get_document_names_by_topic]
            self._options_key = (None, None, None,)
            self._tfidf_vec = None
            self._chunk_boundary = "<chunk-boundary/>"
            self._filter_min_df = filter_min_df
            self._filter_max_df = filter_max_df
            self._filter_stop = filter_stop if filter_stop is not None else []
            self._tfidf_filter = None
            self._tfidf_filter_vec = None
            self._negspacy = {"enabled": omit_negated_chunks, "config": negspacy_config}
            self._process_documents(
                pipeline=pipeline,
                n_process=n_process,
                case_sensitive=case_sensitive,
                disable=disable,
                omit_negated_chunks=omit_negated_chunks,
                negspacy_config=negspacy_config
            )

        # ToDo: some method to set 'doc_topic' outside init?

        @lru_cache()
        def _document_list(
                self
        ) -> List[str]:
            if self._view is None:
                return [v for k, v in sorted(self._text_id_to_doc_name.items(), key=lambda item: item[0])]
            else:
                return [v for k, v in sorted(self._text_id_to_doc_name.items(), key=lambda item: item[0])
                        if self._true_labels[k] in self._view['labels']]

        @property
        def document_list(
                self
        ) -> List[str]:
            return self._document_list()

        @property
        def true_labels_vec(
                self
        ) -> List[int]:
            return [self._true_labels_dict[i] for i in self.true_labels]

        @property
        def true_labels(
                self
        ) -> List[str]:
            if self._view is None:
                return self._true_labels
            else:
                return [l for l in self._true_labels if l.lower() in self._view['labels']]

        @property
        def topics(
                self
        ) -> Set[str]:
            if self._view is None:
                return set(self._true_labels)
            else:
                return set([l.lower() for l in self._true_labels]).intersection(self._view['labels'])

        @property
        def documents_n(
                self
        ) -> int:
            return len(self._document_list())

        @property
        @lru_cache()
        def data_chunk_sets(
                self
        ) -> list:
            if self._view is None:
                return self._chunk_set_dicts
            else:
                return [csd for i, csd in enumerate(self._chunk_set_dicts) if i in self._view['ids']]

        @property
        def processed_docs(
                self
        ) -> list:
            if self._view is None:
                return self._processed_docs
            else:
                return [d for d in self._processed_docs if d._.doc_topic.lower() in self._view['labels']]

        @property
        def chunk_sets_n(
                self
        ) -> int:
            return len(self.data_chunk_sets)

        @property
        def noun_chunks_corpus(
                self,
        ) -> Generator:
            # ToDo: utilize blacklist for noun chunks that should not be included [sie, er, die, etc.] - or check if later on this is done and switch accordingly
            #  because here every superfluous chunk will be run through negex and slows process down and probably  induces errors
            for doc in self._processed_docs:
                for ch in doc.noun_chunks:
                    _negated = not (not hasattr(ch, "_") or
                                    (hasattr(ch, "_") and not getattr(getattr(ch, "_"), "negex", True)))
                    if not (re.match(r"\W", ch.text) and len(ch.text) == 1):
                        if self._view is None or doc._.doc_topic in self._view['labels']:
                            yield {"spacy_chunk": ch, "doc_id": doc._.doc_id, "doc_index": doc._.doc_index,
                                   "doc_name": doc._.doc_name, "doc_topic": doc._.doc_topic, "negated": _negated}

        @property
        def document_chunk_matrix(
                self
        ) -> List[List[str]]:
            return self._document_chunk_matrix

        @property
        def tfidf_filter(
                self
        ) -> Optional[tfidfVec]:
            if self._tfidf_filter is None:
                if self._filter_min_df != 1 or self._filter_max_df != 1.0 or self._filter_stop is not None:
                    filter_stop = self._filter_stop if self._filter_stop is not None else []
                    self._tfidf_filter = tfidfVec(
                        min_df=self._filter_min_df, max_df=self._filter_max_df, stop_words=filter_stop,
                        analyzer=lambda x: re.split(self._chunk_boundary, x)
                    )
                    self._tfidf_filter_vec = self._tfidf_filter.fit_transform(self.document_chunk_matrix)
                else:
                    return None
            return self._tfidf_filter

        def reset_filter(
                self,
                filter_min_df: Union[int, float] = 1,
                filter_max_df: Union[int, float] = 1.,
                filter_stop: Optional[list] = None
        ) -> None:
            self._filter_min_df = filter_min_df
            self._filter_max_df = filter_max_df
            self._filter_stop = filter_stop
            self._tfidf_filter = None
            self._tfidf_filter_vec = None

        def doc_id_from_name(
                self,
                name: str
        ) -> int:
            _id = list(self._text_id_to_doc_name.keys())[list(self._text_id_to_doc_name.values()).index(name)]
            if (self._view is None) or (self._true_labels[_id].lower() in self._view['labels']):
                return _id

        def doc_name_from_id(
                self,
                doc_id: int
        ) -> str:
            if (self._view is None) or (self._true_labels[doc_id].lower() in self._view['labels']):
                return self._text_id_to_doc_name[doc_id]

        @lru_cache()
        def get_document_by_id(
                self,
                doc_id: int
        ) -> List[Doc]:
            if self._view is None:
                return [t for t in self._processed_docs if t._.doc_id == doc_id]
            else:
                return [t for t in self._processed_docs
                        if (t._.doc_id == doc_id and t._.doc_topic.lower() in self._view['labels'])]

        @lru_cache()
        def get_document_by_name(
                self,
                doc_name: str
        ) -> List[Doc]:
            if self._view is None:
                return [t for t in self._processed_docs if t._.doc_name == doc_name]
            else:
                return [t for t in self._processed_docs
                        if (t._.doc_name == doc_name and t._.doc_topic.lower() in self._view['labels'])]

        @lru_cache()
        def get_document_names_by_topic(
                self,
                topic: str
        ) -> List[str]:
            self._check_view_elements(topic)
            return sorted(set([d._.doc_name for d in self._processed_docs
                               if d._.doc_topic is not None and d._.doc_topic.lower() == topic.lower()]))

        @lru_cache()
        def get_document_ids_by_topic(
                self,
                topic: str
        ) -> List[int]:
            self._check_view_elements(topic)
            return sorted(set([d._.doc_id for d in self._processed_docs
                               if d._.doc_topic is not None and d._.doc_topic.lower() == topic.lower()]))

        def set_view_by_labels(
                self,
                labels: Optional[Iterable[str]] = None
        ) -> None:
            for _obj in self._cache_obj:
                _obj.cache_clear()
            if labels is not None:
                labels = [l.lower() for l in labels]
                self._view = {'ids': np.where(np.isin(self._true_labels, np.asarray(labels)))[0],
                              'labels': labels}
            else:
                self._view = None

        def rebuild_chunk_set_dict(
                self,
                use_lemma: bool = False,
                prepend_head: bool = False,
                head_only: bool = False,
                case_sensitive: bool = False,
                omit_negated_chunks: bool = True
        ) -> None:
            self._chunk_set_dicts.clear()
            self._build_chunk_set_dicts(use_lemma=use_lemma, prepend_head=prepend_head, head_only=head_only,
                                        case_sensitive=case_sensitive, omit_negated_chunks=omit_negated_chunks)

        def _check_view_elements(
                self,
                label: Union[str, Iterable[str]]
        ) -> None:
            if isinstance(label, str):
                if label.lower() not in (self._view['labels']
                if self._view is not None else [t.lower() for t in self.topics]):
                    raise KeyError(f"'{label}' is not in current view.")
            elif isinstance(label, Iterable):
                _missing = []
                for it in label:
                    try:
                        self._check_view_elements(it)
                    except KeyError:
                        _missing.append(it)
                if len(_missing) > 0:
                    raise KeyError(f"'{','.join(_missing)}' are not in current view.")

        def _build_data_tuples(
                self
        ) -> None:
            if len(self._data_corpus_tuples) == 0:
                _labels_count = 0
                for i, d in enumerate(self._data_entries):
                    _label = d.get("label", None)
                    if _label not in self._true_labels_dict:
                        self._true_labels_dict[_label] = _labels_count
                        _labels_count += 1
                    self._true_labels.append(_label)
                    self._text_id_to_doc_name[i] = d.get("name", "no_name")
                    for line in d.get("content", "").split('\n'):
                        if not (line.isspace() or len(line) == 0):
                            self._data_corpus_tuples.append(
                                (line, {"doc_id": d.get("id", i), "doc_index": i,
                                        "doc_name": d.get("name", "no_name"), "doc_topic": _label})
                            )

        def _build_chunk_set_dicts(
                self,
                prepend_head: bool,
                use_lemma: bool,
                head_only: bool,
                case_sensitive: bool = False,
                omit_negated_chunks: bool = True
        ) -> None:
            _key = (prepend_head, use_lemma, head_only,)
            if len(self._chunk_set_dicts) == 0 and (_key != self._options_key):
                self._options_key = copy.copy(_key)
                _csdt = {}
                self._document_chunk_matrix = ["" for i in range(self.documents_n)]
                for i, ch in enumerate(self.noun_chunks_corpus):
                    _chunk_dict = clean_span(ch["spacy_chunk"])
                    _negated_chunk = ch["negated"]
                    # return value looks like this:
                    #   {'head_idx': _head_idx (int), 'lemma': _lemma (list), 'text': _text (list), 'pos': _pos (list)}
                    if _chunk_dict is None:
                        continue

                    _text = get_actual_str(_chunk_dict, _key, case_sensitive=case_sensitive)
                    if not (_negated_chunk and omit_negated_chunks):
                        self._document_chunk_matrix[ch["doc_index"]] += f"{self._chunk_boundary}{_text}"

                    if _csdt.get(_text, False) and not (_negated_chunk and omit_negated_chunks):
                        _docs = set(_csdt[_text]["doc"])
                        _docs.add(ch["doc_id"])
                        _csdt[_text]["doc"] = list(_docs)
                        _csdt[_text]["count"] += 1
                    else:
                        _csdt[_text] = ({"doc": [ch["doc_id"]], "count": 1}
                                        if not (_negated_chunk and omit_negated_chunks) else {"doc": [], "count": 0})

                self._chunk_set_dicts = [{"text": _t, "doc": _ch["doc"], "count": _ch["count"]}
                                         for _t, _ch in _csdt.items()]

        def _process_documents(
                self,
                pipeline: spacy.Language,
                n_process: int = 1,
                case_sensitive: bool = False,
                disable: Optional[Iterable[str]] = None,
                omit_negated_chunks: bool = True,
                negspacy_config: Any = None
        ) -> None:
            _negspacy_config = {}
            if omit_negated_chunks and (negspacy_config is not None):
                _negspacy_config = validate_negspacy_config(negspacy_config)

            if omit_negated_chunks:
                logging.info(f"Omitting negated entities with following settings: {_negspacy_config}")
                pipeline.add_pipe("negex", last=True, config=_negspacy_config)
            disable = [] if disable is None else disable
            if len(self._processed_docs) == 0:
                _pipe_trf_type = True if "trf" in pipeline.meta["name"].split("_") else False
                _set_extensions()
                self._build_data_tuples()

                data_corpus = pipeline.pipe(self._data_corpus_tuples, as_tuples=True, n_process=n_process,
                                            disable=disable)
                #ToDo: it crashes here with 'not able to iterate over bool' when using json method only; data_corpus shows as 'generator'
                for _doc, _ctx in tqdm(data_corpus, total=len(self._data_corpus_tuples)):
                    _doc._.doc_id = _ctx.get("doc_id", None)
                    _doc._.doc_index = _ctx.get("doc_index", None)
                    _doc._.doc_name = _ctx.get("doc_name", None)
                    _doc._.doc_topic = _ctx.get("doc_topic", None)
                    self._processed_docs.append(_doc)
                    if _pipe_trf_type:
                        _doc._.trf_data = None  # clears cache and saves ram when using trf_pipelines

                self._build_chunk_set_dicts(prepend_head=self._prepend_head, head_only=self._head_only,
                                            use_lemma=self._use_lemma, case_sensitive=case_sensitive,
                                            omit_negated_chunks=omit_negated_chunks)


def validate_negspacy_config(config) -> dict:
    _val_dict = {
        "neg_termset": None,
        "feat_types": None,
        "extension_name": "negex",
        "chunk_prefix": None,
        "neg_termset_file": None,
        "feat_of_interest": FeaturesOfInterest.NOUN_CHUNKS,
        "scope": None,
        "language": "en"
    }
    _foi = {"NC": FeaturesOfInterest.NOUN_CHUNKS, "NE": FeaturesOfInterest.NAMED_ENTITIES, "BOTH": FeaturesOfInterest.BOTH}
    _return_dict = {}

    for k, v in _val_dict.items():
        v_alt = getattr(config, k, None)
        if k == "feat_of_interest":
            v_alt = _foi.get(v_alt, FeaturesOfInterest.NOUN_CHUNKS)
        if v_alt is not None:
            _return_dict[k] = v_alt
        else:
            if v is None:
                continue
            _return_dict[k] = v
    return _return_dict


def clean_span(
        chunk
) -> Optional[dict]:
    _chunk_root_text = chunk.root.text.strip().replace("\t", "")
    # _chunk_root_lemma_text = chunk.root.lemma_.strip()
    _text, _lemma, _pos = [], [], []

    # if head_only or prepend_head:
    #     _text.append(_chunk_root_text)
    #     _lemma.append(_chunk_root_lemma_text)

    # if not head_only:
    for i, _token in enumerate(chunk):
        _token_text = _token.text.strip().replace("\t", "")
        if _token.is_stop or _token.pos_ in ["DET", "PRON"] or _token.like_num or _token.text.isspace():
            # or (prepend_head and _token_text == _chunk_root_text)):
            continue
        _text.append(_token_text)
        _lemma.append(_token.lemma_.strip().replace("\t", ""))
        _pos.append(_token.pos_)

    # _new_noun_phrase = " ".join([t for t in _text])
    # _new_noun_phrase_lemma = " ".join([t for t in _lemma])

    try:
        _head_idx = _text.index(_chunk_root_text)
        return {'head_idx': _head_idx, 'lemma': _lemma, 'text': _text, 'pos': _pos}
    except ValueError:
        return None
    # if len(_new_noun_phrase) > 0 and not _new_noun_phrase.isspace():
    #     return re.sub(r"^[^\w]+", "", _new_noun_phrase), re.sub(r"^[^\w]+", "", _new_noun_phrase_lemma)
    # else:
    #     return None, None


def get_actual_str(
        chunk_dict: dict,
        modify_key: tuple,
        case_sensitive: bool = False
) -> str:
    prepend_head, use_lemma, head_only = modify_key
    if use_lemma:
        _key_str = 'lemma'
    else:
        _key_str = 'text'

    _head_idx: int = chunk_dict['head_idx']
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
    _data = None
    with zipfile.ZipFile("C:/Users/fra3066mat/Documents/Corpora/grassco.zip", mode='r') as archive:
        _data = [{"name": pathlib.Path(f.filename).stem,
                  "content": archive.read(f.filename).decode('utf-8'),
                  "label": None
                  } for f in archive.filelist if
                 (not f.is_dir()) and (pathlib.Path(f.filename).suffix.lstrip('.') == 'txt')]
    data_proc = DataProcessingFactory.create(
        pipeline=spacy.load("de_dep_news_trf"),
        base_data=_data[1:3],
        save_to_file=False
    )
