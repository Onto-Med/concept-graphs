import copy
import io
import logging
import os
import pathlib
import re
import itertools
import sys
from collections import defaultdict
from random import sample

import numpy as np
from spacy.tokens import DocBin
from spacy.tokens.doc import Doc
from tqdm.autonotebook import tqdm
from typing import Optional, Generator, Union, Iterable, Dict, List, Set, Callable, Any

import spacy
from functools import lru_cache

from spacy import Language
from sklearn.feature_extraction.text import TfidfVectorizer as tfidfVec

from main_utils import load_spacy_model, get_default_spacy_model
from src.negspacy.utils import FeaturesOfInterest
from src.negspacy.negation import Negex
from src.util_functions import load_pickle, save_pickle, add_offset_to_documents_dicts_by_id, set_spacy_extensions


def _populate_chunk_set_dict_with_doc(
        dict2populate: dict, text: str, offset:tuple[int, int], noun_chunk_dict: dict
):
    add_offset_to_documents_dicts_by_id(dict2populate[text]["doc"], noun_chunk_dict["doc_id"], offset)
    dict2populate[text]["count"] += 1


class DataProcessingFactory:

    @classmethod
    def load(
            cls,
            data_obj_path: Union[pathlib.Path, str, io.IOBase]
    ) -> 'DataProcessing':
        set_spacy_extensions()
        _data_obj = load_pickle(data_obj_path)
        _doc_bin = DocBin().from_disk(data_obj_path.with_suffix(".spacy"))
        try:
            _ = _data_obj.data_chunk_sets
            _data_obj.processed_docs = _doc_bin
        except AttributeError:
            raise AssertionError(f"The loaded object '{_data_obj}' seems not to be a DataProcessing object or is missing important attribute 'data_chunk_sets'.")
        return _data_obj

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
            # Using DocBin -- mainly for future use of proper DBs
            doc_bin = DocBin(docs=_data_processing._processed_docs, store_user_data=True)
            doc_bin.to_disk(final_cache.with_suffix(".spacy"))
            _data_processing._processed_docs.clear()
            # DocBin end
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
            self._document_process_config = None
            self._process_documents(
                pipeline=pipeline,
                n_process=n_process,
                case_sensitive=case_sensitive,
                disable=disable if (isinstance(disable, Iterable) or disable is None) else [],
                omit_negated_chunks=omit_negated_chunks,
                negspacy_config=negspacy_config
            )

        # ToDo: some method to set 'doc_topic' outside init?

        @property
        def model_name(self) -> Optional[str]:
            if _pipe := self.document_process_config.get("pipeline", {}):
                if _pipe.get("lang", False) and _pipe.get("name", False):
                    return f"{_pipe['lang']}_{_pipe['name']}"
            return None

        @property
        def document_process_config(self) -> dict:
            return self._document_process_config

        @document_process_config.setter
        def document_process_config(self, value: dict):
            self._document_process_config = value

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

        @processed_docs.setter
        def processed_docs(self, value: Union[list, DocBin]):
            if isinstance(value, list):
                self._processed_docs = value
            elif isinstance(value, DocBin):
                self._processed_docs = list(value.get_docs(
                    load_spacy_model(self.model_name, logging.getLogger(__name__), get_default_spacy_model()).vocab
                ))

        @property
        def chunk_sets_n(
                self
        ) -> int:
            return len(self.data_chunk_sets)

        def noun_chunks_corpus(
                self,
                external: Optional[list[Doc]]
        ) -> Generator:
            # ToDo: utilize blacklist for noun chunks that should not be included [sie, er, die, etc.] - or check if later on this is done and switch accordingly
            #  because here every superfluous chunk will be run through negex and slows process down and probably  induces errors
            for doc in self.processed_docs if external is None else external:
                _offset_in_doc = doc._.offset_in_doc
                for ch in doc.noun_chunks:
                    ch: spacy.tokens.Span
                    # _negated = not (not hasattr(ch, "_") or
                    #                 (hasattr(ch, "_") and not getattr(getattr(ch, "_"), "negex", True)))
                    _negated = hasattr(ch, "_") and getattr(getattr(ch, "_"), "negex", False)
                    if len(ch.text) == 1 and re.match(r"\W", ch.text):
                        # _offset_in_doc += 1
                        continue
                    if self._view is None or doc._.doc_topic in self._view['labels']:
                        yield {"spacy_chunk": ch, "doc_id": doc._.doc_id, "doc_index": doc._.doc_index,
                               "doc_name": doc._.doc_name, "doc_topic": doc._.doc_topic, "negated": _negated,
                               "offset": (ch.start_char + _offset_in_doc, ch.end_char + _offset_in_doc, )}

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
                if self._filter_min_df != 1 or self._filter_max_df != 1.0 or self._filter_stop not in [None, False] or (isinstance(self._filter_stop, list) and len(self._filter_stop) != 0):
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
            self._build_chunk_set_dicts(data=None, use_lemma=use_lemma, prepend_head=prepend_head, head_only=head_only,
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
                self,
                split_on: str = "\n",
                external: Optional[list[dict]] = None
        ) -> list[tuple[str, dict[str, Optional[Union[str, int]]]]]:
            _data_corpus_tuples = []
            # if len(self._data_corpus_tuples) == 0:
            _labels_count = 0
            for i, d in enumerate(self._data_entries if external is None else external):
                _label = d.get("label", None)
                if _label not in self._true_labels_dict and external is None:
                    self._true_labels_dict[_label] = _labels_count
                    _labels_count += 1
                if external is None: self._true_labels.append(_label)
                if external is None: self._text_id_to_doc_name[i] = d.get("name", "no_name")
                _offset = 0
                for line in d.get("content", "").split(split_on):
                    if line.isspace():
                        _offset += (len(line) +len(split_on))
                    elif len(line) == 0:
                        _offset += len(split_on)
                    else:
                        _data_corpus_tuples.append(
                            (line, {"doc_id": d.get("id", i), "doc_index": i,
                                    "doc_name": d.get("name", "no_name"), "doc_topic": _label,
                                    "offset_in_doc": _offset})
                        )
                        _offset += (len(line) +len(split_on))
            return _data_corpus_tuples

        def _build_chunk_set_dicts(
                self,
                data: Optional[Iterable[spacy.tokens.Doc]],
                prepend_head: bool,
                use_lemma: bool,
                head_only: bool,
                case_sensitive: bool = False,
                omit_negated_chunks: bool = True
        ) -> Optional[list[dict]]:
            _key = (prepend_head, use_lemma, head_only,)
            if (len(self._chunk_set_dicts) == 0 and (_key != self._options_key) and data is None) or data is not None:
                if data is None: self._options_key = copy.copy(_key)
                _csdt = {}
                if data is None: self._document_chunk_matrix = ["" for i in range(self.documents_n)]

                for i, ch in enumerate(self.noun_chunks_corpus(data)):
                    ch: dict
                    _chunk_dict = clean_span(ch["spacy_chunk"], ch["offset"])
                    _negated_chunk: bool = ch["negated"]
                    # return value looks like this:
                    #   {'head_idx': _head_idx (int), 'lemma': _lemma (list), 'text': _text (list), 'pos': _pos (list)}
                    if _chunk_dict is None:
                        continue

                    _text = get_actual_str(_chunk_dict, _key, case_sensitive=case_sensitive)
                    _offset = _chunk_dict["offset"]
                    if (not (_negated_chunk and omit_negated_chunks)) and data is None:
                        self._document_chunk_matrix[ch["doc_index"]] += f"{self._chunk_boundary}{_text}"

                    if _csdt.get(_text, False) and not (_negated_chunk and omit_negated_chunks):
                        #ToDo: maybe string similarity check already here to merge similar phrases
                        _populate_chunk_set_dict_with_doc(_csdt, _text, _offset, ch)
                    else:
                        _csdt[_text] = ({"doc": [{"id": ch["doc_id"], "offsets": [_offset]}], "count": 1}
                                        if not (_negated_chunk and omit_negated_chunks) else {"doc": [], "count": 0})

                return [{"text": _t, "doc": _ch["doc"], "count": _ch["count"]}
                        for _t, _ch in _csdt.items()] #ToDo: omit chunks that have a doc count of 0?
            return None

        def _process_documents(
                self,
                pipeline: Optional[spacy.Language] = None,
                n_process: int = 1,
                case_sensitive: bool = False,
                disable: Optional[Iterable[str]] = None,
                omit_negated_chunks: bool = True,
                negspacy_config: Optional[dict] = None,
                external: Optional[list[dict]] = None
        ) -> Optional[list]:
            if pipeline is None and external is None:
                logging.error("No spacy pipeline specified and not referencing deserialized pipeline.")
                return None
            _negspacy_config = {}
            if external is not None and self.document_process_config is not None:
                _negspacy_config = self.document_process_config.get("negspacy_config", {})
                omit_negated_chunks = self.document_process_config.get("omit_negated_chunks", True)
                negex_ext_name = _negspacy_config.get("extension_name", "negex")
                Negex.set_extension(negex_ext_name)
            if external is None and omit_negated_chunks and (negspacy_config is not None):
                _negspacy_config = validate_negspacy_config(negspacy_config)
                logging.info(f"Omitting negated entities with following settings: {_negspacy_config}")
                pipeline.add_pipe("negex", last=True, config=_negspacy_config)

            disable = [] if disable is None else disable

            if self.document_process_config is None and external is None:
                self.document_process_config = {
                    "pipeline": {
                        "name": pipeline.meta.get("name", None),
                        "lang": pipeline.meta.get("lang", None),
                        "version": pipeline.meta.get("version", None),
                    },
                    "n_process": n_process,
                    "case_sensitive": case_sensitive,
                    "disable": disable,
                    "omit_negated_chunks": omit_negated_chunks,
                    "negspacy_config": _negspacy_config
                }
            else:
                case_sensitive, disable, n_process = [
                    kv[1] for kv in sorted(self.document_process_config.items(),
                                           key=lambda x: x[0]) if kv[0] not in ["negspacy_config", "omit_negated_chunks", "pipeline"]]
                pipeline = load_spacy_model(
                    self.model_name,
                    logging.getLogger(__name__),
                    get_default_spacy_model()
                )
                if omit_negated_chunks and (_negspacy_config is not None):
                    pipeline.add_pipe(_negspacy_config.get("extension_name", "negex"), last=True, config=_negspacy_config)

            if (len(self._processed_docs) == 0 and external is None) or external is not None:
                _pipe_trf_type = True if "trf" in pipeline.meta["name"].split("_") else False
                set_spacy_extensions()
                _data_corpus_tuples = self._build_data_tuples(external=external)

                data_corpus = pipeline.pipe(_data_corpus_tuples[:], as_tuples=True, n_process=n_process,
                                            disable=disable)
                _processed_docs = []
                for _doc, _ctx in tqdm(data_corpus, total=len(_data_corpus_tuples)):
                    _doc._.doc_id = _ctx.get("doc_id", None)
                    _doc._.doc_index = _ctx.get("doc_index", None)
                    _doc._.doc_name = _ctx.get("doc_name", None)
                    _doc._.doc_topic = _ctx.get("doc_topic", None)
                    _doc._.offset_in_doc = _ctx.get("offset_in_doc", None)
                    if _pipe_trf_type:
                        _doc._.trf_data = None  # clears cache and saves ram when using trf_pipelines
                    _processed_docs.append(_doc)
                if external is None: self._processed_docs = _processed_docs

                _chunk_set_dicts = self._build_chunk_set_dicts(
                    data=_processed_docs if external is not None else None, prepend_head=self._prepend_head,
                    head_only=self._head_only, use_lemma=self._use_lemma, case_sensitive=case_sensitive,
                    omit_negated_chunks=omit_negated_chunks
                )
                if external is None and _chunk_set_dicts is not None:
                    self._chunk_set_dicts = _chunk_set_dicts
                else:
                    return _chunk_set_dicts
            return None

        def process_external_docs(self, content: list[dict[str, Optional[str]]]):
            # dict: {"name": document name, "content": document content, "label": label/category of document if present}
            return self._process_documents(external=content)


def validate_negspacy_config(config) -> dict:
    _val_dict = {
        "neg_termset": None,
        "feat_types": None,
        "extension_name": "negex",
        "chunk_prefix": None,
        "neg_termset_file": None,
        "feat_of_interest": FeaturesOfInterest.NOUN_CHUNKS,
        "scope": None,
        "language": None
    }
    _foi = {"nc": FeaturesOfInterest.NOUN_CHUNKS, "ne": FeaturesOfInterest.NAMED_ENTITIES, "both": FeaturesOfInterest.BOTH}
    _return_dict = {}

    for k, v in _val_dict.items():
        v_alt = getattr(config, k, None) if not isinstance(config, dict) else config.get(k, None)
        if k == "feat_of_interest":
            v_alt = _foi.get(v_alt.lower(), FeaturesOfInterest.NOUN_CHUNKS) if isinstance(v_alt, str) else (
                v_alt if isinstance(v_alt, list) else FeaturesOfInterest.NOUN_CHUNKS
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
        _relative_token_offset = _token.idx - _global_offset_first  # -- '.idx' is char offset relative to the whole doc (here, the line)
        _space_between_next = 0
        # if i > 0:
        #     _last_token = chunk[i - 1]
        #     _len_last_token = len(_last_token.text)
        if i < len(chunk) - 1:
            _next_token = chunk[i + 1]
            _relative_token_offset_next = _next_token.idx - _global_offset_first
            _space_between_next = _relative_token_offset_next - _relative_token_offset - len(_token.text)

        _token_text = _token.text.strip().replace("\t", "")
        if ((not _token.is_alpha) or _token.is_stop or _token.is_punct or _token.like_num or
                _token.pos_ in ["DET", "PRON"] or _token_text.isspace()):
            if offset is not None and i == _i and i < (len(chunk) - 1):
                _add_begin_from_last += (len(_token.text) + _space_between_next)
                _i += 1
            elif offset is not None and i == len(chunk) - 1:
                _remove_end_from_last += (len(_token.text) + _space_between_next)
        else:
            _text.append(_token_text)
            _lemma.append(_token.lemma_.strip().replace("\t", ""))
            _pos.append(_token.pos_)

    try:
        _head_idx = _text.index(_chunk_root_text)
        return {
            'head_idx': _head_idx,
            'lemma': _lemma,
            'text': _text,
            'pos': _pos,
            'offset': (_begin + _add_begin_from_last, _end - _remove_end_from_last,) if offset is not None else None
        }
    except ValueError:
        return None

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

    data_proc = DataProcessingFactory.load(pathlib.Path("C:/Users/fra3066mat/PycharmProjects/concept-graphs/tmp/grascco_test/grascco_test_data.pickle"))
    res = data_proc.process_external_docs([{"name": "test",
                                            "content": "Das ist ein Test Dokument. Es hat ein paar Nomen Chunks; vielleicht. Auch gab es keinen Nachweis für einen Sinn. Aber es wurd ein Sinn nicht gesehen.",
                                            "label": None}])
    print(res)
