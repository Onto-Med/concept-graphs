import logging
from typing import Iterable, Any, Generator, Union, Optional

from langchain_text_splitters import TextSplitter
from spacy.tokens import Doc


class PreprocessedSpacyTextSplitter(TextSplitter):
    """Splitting text from preprocessed Spacy data.

    :param separator:
    :param strip_whitespace:
    """

    def __init__(
            self,
            separator: str = "\n\n",
            strip_whitespace: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._separator = separator
        self._strip_whitespace = strip_whitespace

    def split_text(
            self,
            text: str
    ) -> None:
        raise NotImplementedError("This class needs preprocessed Spacy documents. If you want to split raw text,"
                                  " consider using 'langchain_text_splitters.SpacyTextSplitter'.")

    def split_preprocessed_documents(
            self,
            docs: Iterable[Doc],
            keep_metadata: Optional[list[str]] = None
    ) -> Generator[Union[list[str], tuple[list[str], dict]], None, None]:
        for doc in docs:
            splits = (
                s.text if self._strip_whitespace else s.text_with_ws
                for s in doc.sents
            )
            if keep_metadata is None:
                yield self._merge_splits(splits, self._separator)
            else:
                _meta_data = {}
                if hasattr(doc, "_"):
                    for k in getattr(doc, "_").__dict__.get("_extensions", {}):
                        if k in keep_metadata:
                            _meta_data[k] = getattr(getattr(doc, "_"), k)
                yield self._merge_splits(splits, self._separator), _meta_data

    def split_preprocessed_sentences(
            self,
            sentences: Iterable[Doc],
            doc_metadata_key: str,
            keep_metadata: Optional[list[str]] = None,
    ) -> Generator[Union[list[str], tuple[list[str], dict]], None, None]:
        docs = []
        current_doc_id = None
        meta_data = {}
        warned_once = False
        for sentence in sentences:
            if doc_id := getattr(getattr(sentence, "_", {}), doc_metadata_key, None):
                splits = (
                    s.text if self._strip_whitespace else s.text_with_ws
                    for s in docs
                )
                if doc_id != current_doc_id:
                    if keep_metadata is None:
                        yield self._merge_splits(splits, self._separator)
                    else:
                        for k in getattr(sentence, "_", {}).__dict__.get("_extensions", {}):
                            if k in keep_metadata:
                                meta_data[k] = getattr(getattr(sentence, "_"), k)
                        yield self._merge_splits(splits, self._separator), meta_data
                    docs = []
                    meta_data = {}
                docs.append(sentence)
            else:
                if not warned_once:
                    logging.warning(f"There seems to be no metadata for '{doc_metadata_key}'"
                                    f" that could be used to distinguish documents; continuing without.")
                    warned_once = True
                docs.append(sentence)
        if warned_once and len(docs) > 0:
            splits = (
                s.text if self._strip_whitespace else s.text_with_ws
                for s in docs
            )
            yield self._merge_splits(splits, self._separator)