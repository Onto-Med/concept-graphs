import logging
from collections import defaultdict
from operator import itemgetter
from pydoc import locate
from typing import Union, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from chatters.AbstractChatter import Chatter
from src.rag.embedding_stores.MarqoChunkEmbeddingStore import MarqoChunkEmbeddingStore
from src.rag.marqo_rag_utils import extract_text_from_highlights


class RAG:
    def __init__(
        self,
        chatter: Union[Chatter, str],
        language: Optional[str] = None
    ):
        self._language = language
        self._documents = None
        self._prompt = None
        self._initialized_chatter = None

        if isinstance(chatter, str):
            self._chatter = locate(chatter)
        elif isinstance(chatter, Chatter):
            self._chatter = chatter
        else:
            raise TypeError(f"'chatter' must be an implementation of the Chatter class, or a string denoting the location of said class.")

    @classmethod
    def with_chatter(
        cls,
        chatter: Union[Chatter, str] = "src.rag.chatters.BlabladorChatter.BlabladorChatter",
        **kwargs
    ) -> "RAG":
        _rag = cls(chatter, language=kwargs.pop("language", None))
        if len(kwargs) > 0:
            return _rag.with_chatter_options(**kwargs)
        else:
            return _rag

    def _get_language(
        self,
        lang: str
    ) -> str:
        _lang_options = ["de", "en"]
        _lang = lang if self._language is None else self._language
        return _lang if _lang in _lang_options else "en"

    def _concatenate_by_metadata(self, doc_tuples: list[tuple], concat_by: str, concat_str: str = "\n\n") -> list[tuple]:
        _text_dict = defaultdict(list)
        _meta_dict = {}
        for text, meta in doc_tuples:
            if not meta.get(concat_by, False):
                logging.warning(f"Encountered 'concat_by' value ({concat_by}) that is not in metadata.")
                continue
            _text_dict[meta[concat_by]].append(text)
            if not _meta_dict.get(meta[concat_by], False):
                _meta_dict[meta[concat_by]] = meta
        return [(concat_str.join(_text_dict[_id]), _meta_dict[_id], ) for _id in _text_dict.keys()]

    def with_chatter_options(
        self,
        api_key: str,
        **kwargs
    ) -> "RAG":
        if not "api_key" in kwargs:
            kwargs["api_key"] = api_key
        self._initialized_chatter =  self._chatter.with_kwargs(**kwargs)
        return self

    def with_prompt(
        self,
        lang: str = "en"
    ) -> "RAG":
        templates = {
            "en":
                """
                Given the following extracted parts of several different documents ("SOURCES") and a question ("QUESTION"), create a final answer one paragraph long. 
                Don't try to make up an answer and use the text in the SOURCES only for the answer. If you don't know the answer, just say that you don't know. 
                QUESTION: {question}
                =========
                SOURCES:
                {summaries}
                =========
                ANSWER:
                """,
            "de":
                """
                Gegeben sind die die folgenden Teile verschiedener Dokumente ("QUELLEN") und eine Frage ("FRAGE"), erstelle eine kurze abschließende Antwort mit etwa einer Länge eines Absatz.
                Dabei sollen die "QUELLEN" individuell betrachtet werden! Referenziere in der Antwort die QUELLEN!
                Versuche niemals eine Antwort zu erfinden! Benutze außschließlich die Texte aus den QUELLEN für die Antwort. Wenn du keine Antwort hast, sage einfach, dass du es nicht weißt! 
                FRAGE: {question}
                =========
                QUELLEN:
                {summaries}
                =========
                ANSWER:
                """
        }
        self._prompt = PromptTemplate(
            template=templates.get(self._get_language(lang)),
            input_variables=["summaries", "question"]
        )
        return self

    def with_documents(
        self,
        documents: Union[list[str], list[tuple[str, dict]]],
        lang: str = "en",
        concat_by: str = None,
        concat_str: str = "\n\n"
    ) -> "RAG":
        _source_str_map = {"en": "Source", "de": "Quelle"}
        _language = self._get_language(lang)
        if len(documents) == 0:
            logging.warning("No documents given!")
            return self
        with_metadata = isinstance(documents[0], tuple)
        if with_metadata and concat_by is not None:
            documents = self._concatenate_by_metadata(documents, concat_by, concat_str)
        self._documents = [
            Document(
                page_content=f"{_source_str_map.get(self._get_language(lang))} [{ind}]: " + (d[0] if with_metadata else d),
                metadata=d[1] if with_metadata else {}
            )
            for ind, d in enumerate(documents)]
        return self

    def build(
        self
    ) -> Runnable:
        return self._prompt | self._initialized_chatter

    def build_and_invoke(
        self,
        question: str
    ):
        return (self._prompt | self._initialized_chatter).invoke(
            {
                "summaries": self._documents,
                "question": question
            },
            return_only_outputs=True
        )

if __name__ == "__main__":
    import sys
    import pathlib
    from load_utils import FactoryLoader
    from TextSplitters import PreprocessedSpacyTextSplitter

    _args = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:]}
    _api_key = _args.pop("api_key")

    _data = FactoryLoader.load_data(str(pathlib.Path("../../tmp/grascco_stem").resolve()), "grascco_stem")
    _splitter = PreprocessedSpacyTextSplitter(chunk_size=400, chunk_overlap=100)
    _documents = _splitter.split_preprocessed_sentences(_data.processed_docs, "doc_id", keep_metadata=["doc_id", "doc_name"])

    chunk_embedding_store = MarqoChunkEmbeddingStore.from_config(
        index_name="grascco_stem_rag",
        config={
            "model": "multilingual-e5-base",
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": 3,
                "splitOverlap": 1,
                "splitMethod": "sentence"
            },
        }
    )

    if not chunk_embedding_store.is_filled():
        docs = [
            {
                "text": d,
                "doc_id": t[1]["doc_id"],
                "doc_name": t[1]["doc_name"],
            } for t in _documents for d in t[0]
        ]
        chunk_embedding_store.add_chunks(docs)

    question = "Gibt es einen Patienten mit einer Lungenentzündung?"
    result = (
        RAG
        .with_chatter(api_key=_api_key, language="de")
        .with_prompt()
        .with_documents(
            list(zip(
                *itemgetter(1, -1)(extract_text_from_highlights(chunk_embedding_store.get_chunks(question), token_limit=150, lang="de"))
            )), concat_by="doc_id"
        )
        .build_and_invoke(question)
    )
    print(result)