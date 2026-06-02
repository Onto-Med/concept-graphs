import logging
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import TextSplitter
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextSplitWithOffsets:
    text: str
    start: int | None
    end: int | None


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

    def split_text(self, text: str) -> None:
        raise NotImplementedError(
            "This class needs preprocessed Spacy documents. If you want to split raw text,"
            " consider using 'langchain_text_splitters.SpacyTextSplitter'."
        )

    def split_preprocessed_documents(
        self, docs: Iterable[Doc], keep_metadata: list[str] | None = None
    ) -> Generator[list[str] | tuple[list[str], dict], None, None]:
        for doc in docs:
            splits = (
                s.text if self._strip_whitespace else s.text_with_ws for s in doc.sents
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
        keep_metadata: list[str] | None = None,
    ) -> Generator[list[str] | tuple[list[str], dict], None, None]:
        docs = []
        current_doc_id = None
        warned_once = False

        def yield_current_docs():
            if not docs:
                return None
            splits_with_offsets = [self._split_with_offsets(doc) for doc in docs]
            chunks, chunk_offsets = self._merge_splits_with_offsets(
                splits_with_offsets, self._separator
            )
            if keep_metadata is None:
                return chunks
            meta_data = self._metadata_from_doc(docs[0], keep_metadata)
            meta_data["chunk_offsets"] = chunk_offsets
            return chunks, meta_data

        for sentence in sentences:
            doc_id = getattr(getattr(sentence, "_", {}), doc_metadata_key, None)
            if not doc_id:
                if not warned_once:
                    logging.warning(
                        f"There seems to be no metadata for '{doc_metadata_key}'"
                        f" that could be used to distinguish documents; continuing without."
                    )
                    warned_once = True
                docs.append(sentence)
                continue

            if current_doc_id is None:
                current_doc_id = doc_id
            if doc_id != current_doc_id:
                result = yield_current_docs()
                if result is not None:
                    yield result
                docs = []
                current_doc_id = doc_id
            docs.append(sentence)

        result = yield_current_docs()
        if result is not None:
            yield result

    def _split_with_offsets(self, doc: Doc) -> TextSplitWithOffsets:
        text = doc.text if self._strip_whitespace else doc.text_with_ws
        start = getattr(getattr(doc, "_", {}), "offset_in_doc", None)
        end = None if start is None else start + len(text)
        return TextSplitWithOffsets(text=text, start=start, end=end)

    def _metadata_from_doc(self, doc: Doc, keep_metadata: list[str]) -> dict:
        metadata = {}
        if hasattr(doc, "_"):
            for key in getattr(doc, "_").__dict__.get("_extensions", {}):
                if key in keep_metadata:
                    metadata[key] = getattr(getattr(doc, "_"), key)
        return metadata

    def _merge_splits_with_offsets(
        self, splits: list[TextSplitWithOffsets], separator: str
    ) -> tuple[list[str], list[tuple[int | None, int | None]]]:
        separator_len = self._length_function(separator)
        chunks = []
        offsets = []
        current_doc: list[TextSplitWithOffsets] = []
        total = 0

        def append_current_chunk() -> None:
            chunk = self._join_docs([split.text for split in current_doc], separator)
            if chunk is None:
                return
            starts = [split.start for split in current_doc if split.start is not None]
            ends = [split.end for split in current_doc if split.end is not None]
            chunks.append(chunk)
            offsets.append(
                (
                    min(starts) if starts else None,
                    max(ends) if ends else None,
                )
            )

        for split in splits:
            split_len = self._length_function(split.text)
            if (
                total + split_len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        "Created a chunk of size %s, which is longer than the specified %s",
                        total,
                        self._chunk_size,
                    )
                if len(current_doc) > 0:
                    append_current_chunk()
                    while total > self._chunk_overlap or (
                        total
                        + split_len
                        + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0].text) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(split)
            total += split_len + (separator_len if len(current_doc) > 1 else 0)

        if current_doc:
            append_current_chunk()
        return chunks, offsets


if __name__ == "__main__":
    import pathlib

    from src.pipeline.load_utils import FactoryLoader

    _data = FactoryLoader.load_data(
        str(pathlib.Path("../../tmp/grascco_stem").resolve()), "grascco_stem"
    )
    _splitter = PreprocessedSpacyTextSplitter(chunk_size=400, chunk_overlap=100)
    _splits = list(
        _splitter.split_preprocessed_sentences(
            _data.processed_docs, "doc_id", keep_metadata=["doc_id", "doc_name"]
        )
    )
    logger.info("%s", _splits)
