import copy
import io
import os
import pathlib
import re
import itertools
from collections import defaultdict
from random import sample

import numpy as np
from spacy.tokens.doc import Doc
from tqdm.autonotebook import tqdm
from typing import Optional, Generator, Union, Iterable, Dict, List, Set, Callable

import spacy
from functools import lru_cache

from spacy import Language
from sklearn.feature_extraction.text import TfidfVectorizer as tfidfVec

from util_functions import load_pickle, save_pickle


# ToDo: this needs to be called whenever a data_proc object is used/loaded by another class
def _set_extensions(
) -> None:
    from spacy.tokens import Doc
    if not Doc.has_extension("text_id"):
        Doc.set_extension("text_id", default=None)
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
            categories: Optional[list] = None
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
                _doc_per_label = int(sub/len(labels_dict.keys()))
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
            disable=disable
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
                disable: Optional[Iterable[str]] = None
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
            self._filter_stop = filter_stop
            self._tfidf_filter = None
            self._tfidf_filter_vec = None
            self._process_documents(
                pipeline=pipeline,
                n_process=n_process,
                case_sensitive=case_sensitive,
                disable=disable
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
            for doc in self._processed_docs:
                for ch in doc.noun_chunks:
                    if not (re.match(r"\W", ch.text) and len(ch.text) == 1):
                        if self._view is None or doc._.doc_topic in self._view['labels']:
                            yield {"spacy_chunk": ch, "text_id": doc._.text_id,
                                   "doc_name": doc._.doc_name, "doc_topic": doc._.doc_topic}

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
                    self._tfidf_filter = tfidfVec(
                        min_df=self._filter_min_df, max_df=self._filter_max_df, stop_words=self._filter_stop,
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
                return [t for t in self._processed_docs if t._.text_id == doc_id]
            else:
                return [t for t in self._processed_docs
                        if (t._.text_id == doc_id and t._.doc_topic.lower() in self._view['labels'])]

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
            return sorted(set([d._.text_id for d in self._processed_docs
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
                case_sensitive: bool = False
        ) -> None:
            self._chunk_set_dicts.clear()
            self._build_chunk_set_dicts(use_lemma=use_lemma, prepend_head=prepend_head, head_only=head_only,
                                        case_sensitive=case_sensitive)

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
                    _label = d["label"]
                    if _label not in self._true_labels_dict:
                        self._true_labels_dict[_label] = _labels_count
                        _labels_count += 1
                    self._true_labels.append(_label)
                    self._text_id_to_doc_name[i] = d["name"]
                    for line in d["content"].split('\n'):
                        if not (line.isspace() or len(line) == 0):
                            self._data_corpus_tuples.append(
                                (line, {"text_id": i, "doc_name": d["name"], "doc_topic": _label})
                            )

        def _build_chunk_set_dicts(
                self,
                prepend_head: bool,
                use_lemma: bool,
                head_only: bool,
                case_sensitive: bool = False
        ) -> None:
            _key = (prepend_head, use_lemma, head_only,)
            if len(self._chunk_set_dicts) == 0 and (_key != self._options_key):
                self._options_key = copy.copy(_key)
                _csdt = {}
                self._document_chunk_matrix = ["" for i in range(self.documents_n)]
                for i, ch in enumerate(self.noun_chunks_corpus):
                    _chunk_dict = clean_span(ch["spacy_chunk"])
                    # return value looks like this:
                    #   {'head_idx': _head_idx (int), 'lemma': _lemma (list), 'text': _text (list), 'pos': _pos (list)}
                    if _chunk_dict is None:
                        continue

                    _text = get_actual_str(_chunk_dict, _key, case_sensitive=case_sensitive)
                    self._document_chunk_matrix[ch["text_id"]] += f"{self._chunk_boundary}{_text}"

                    if _csdt.get(_text, False):
                        _docs = set(_csdt[_text]["doc"])
                        _docs.add(ch["text_id"])
                        _csdt[_text]["doc"] = list(_docs)
                        _csdt[_text]["count"] += 1
                    else:
                        _csdt[_text] = {"doc": [ch["text_id"]], "count": 1}

                self._chunk_set_dicts = [{"text": _t, "doc": _ch["doc"], "count": _ch["count"]}
                                         for _t, _ch in _csdt.items()]

        def _process_documents(
                self,
                pipeline: spacy.Language,
                n_process: int = 1,
                case_sensitive: bool = False,
                disable: Optional[Iterable[str]] = None
        ) -> None:
            disable = [] if disable is None else disable
            if len(self._processed_docs) == 0:
                _pipe_trf_type = True if "trf" in pipeline.meta["name"].split("_") else False
                _set_extensions()
                self._build_data_tuples()

                data_corpus = pipeline.pipe(self._data_corpus_tuples, as_tuples=True, n_process=n_process,
                                            disable=disable)
                for _doc, _ctx in tqdm(data_corpus, total=len(self._data_corpus_tuples)):
                    _doc._.text_id = _ctx.get("text_id", None)
                    _doc._.doc_name = _ctx.get("doc_name", None)
                    _doc._.doc_topic = _ctx.get("doc_topic", None)
                    self._processed_docs.append(_doc)
                    if _pipe_trf_type:
                        _doc._.trf_data = None  # clears cache and saves ram when using trf_pipelines

                self._build_chunk_set_dicts(prepend_head=self._prepend_head, head_only=self._head_only,
                                            use_lemma=self._use_lemma, case_sensitive=case_sensitive)


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
