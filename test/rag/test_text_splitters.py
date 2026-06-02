from spacy.tokens import Doc
from spacy.vocab import Vocab

from src.common.spacy_extensions import set_spacy_extensions
from src.rag.text_splitters import PreprocessedSpacyTextSplitter


def _doc(text, doc_id="doc-1", offset=0):
    words = text.split(" ")
    spaces = [True] * (len(words) - 1) + [False]
    doc = Doc(Vocab(), words=words, spaces=spaces)
    doc._.doc_id = doc_id
    doc._.doc_name = "doc.txt"
    doc._.offset_in_doc = offset
    return doc


def test_split_preprocessed_sentences_propagates_document_offsets():
    set_spacy_extensions()
    splitter = PreprocessedSpacyTextSplitter(chunk_size=100, chunk_overlap=0)

    chunks, metadata = next(
        splitter.split_preprocessed_sentences(
            [_doc("Alpha beta", offset=10), _doc("Gamma", offset=30)],
            "doc_id",
            keep_metadata=["doc_id", "doc_name"],
        )
    )

    assert chunks == ["Alpha beta\n\nGamma"]
    assert metadata["doc_id"] == "doc-1"
    assert metadata["doc_name"] == "doc.txt"
    assert metadata["chunk_offsets"] == [(10, 35)]
