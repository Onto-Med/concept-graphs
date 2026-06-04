import logging
from collections import defaultdict
from operator import itemgetter
from pydoc import locate
from typing import Any

from langchain_core.documents import Document
from langchain_core.exceptions import LangChainException
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from src.rag.chatters.base import Chatter
from src.rag.embedding_stores.marqo import MarqoChunkEmbeddingStore
from src.rag.marqo_rag_utils import extract_text_from_highlights
from src.rag.prompts import resolve_rag_prompt_config


def no_source_answer(language: str | None = None) -> str:
    """Return a deterministic answer for RAG requests without source documents."""
    if language == "de":
        return "Keine Quelle die ich finden kann."
    return "No source I can find."


def _clean_answer(answer: Any) -> Any:
    """Remove common prompt-echo tails while keeping the generated answer."""
    content = getattr(answer, "content", answer)
    if not isinstance(content, str):
        return answer

    text = content.strip()
    for marker in ["=========", "\nFRAGE:", "\nQUESTION:", "\nQUELLEN:", "\nSOURCES:"]:
        marker_index = text.find(marker)
        if marker_index > 0:
            return text[:marker_index].strip()
    return text


class RAG:
    def __init__(self, chatter: Chatter | str, language: str | None = None):
        self._language = language
        self._documents = None
        self._prompt = None
        self._initialized_chatter = None

        if isinstance(chatter, str):
            self._chatter = locate(chatter)
        elif isinstance(chatter, Chatter):
            self._chatter = chatter
        else:
            raise TypeError(
                "'chatter' must be an implementation of the Chatter class, or a string denoting the location of said class."
            )

    @classmethod
    def with_chatter(
        cls,
        chatter: Chatter | str | None = "src.rag.chatters.blablador.BlabladorChatter",
        **kwargs,
    ) -> "RAG":
        if chatter is None:
            raise TypeError(
                "'chatter' seems to be 'None' but must be an implementation of the Chatter class, or a string denoting the location of said class."
            )
        chatter = (
            chatter
            if chatter is not None
            else "src.rag.chatters.blablador.BlabladorChatter"
        )
        _rag = cls(chatter, language=kwargs.pop("language", None))
        if len(kwargs) > 0:
            return _rag.with_chatter_options(**kwargs)
        else:
            return _rag

    @property
    def documents(self) -> dict[str, Document]:
        return self._documents

    @property
    def language(self) -> str:
        return self._language

    def get_rag_language(self, lang: str) -> str:
        _lang_options = ["de", "en"]
        _lang = lang if self.language is None else self.language
        return _lang if _lang in _lang_options else "en"

    def _concatenate_by_metadata(
        self, doc_tuples: list[tuple], concat_by: str, concat_str: str = "\n\n"
    ) -> list[tuple]:
        _text_dict = defaultdict(list)
        _meta_dict = {}
        for text, meta in doc_tuples:
            if not meta.get(concat_by, False):
                logging.warning(
                    f"Encountered 'concat_by' value ({concat_by}) that is not in metadata."
                )
                continue
            _text_dict[meta[concat_by]].append(text)
            if not _meta_dict.get(meta[concat_by], False):
                _meta_dict[meta[concat_by]] = meta
        return [
            (
                concat_str.join(_text_dict[_id]),
                _meta_dict[_id],
            )
            for _id in _text_dict.keys()
        ]

    def with_chatter_options(self, api_key: str, **kwargs) -> "RAG":
        if "api_key" not in kwargs:
            kwargs["api_key"] = api_key
        self._initialized_chatter = self._chatter.with_kwargs(**kwargs)
        if self._initialized_chatter is None:
            raise ValueError(
                "Chatter failed to initialize!"
                "Please consult the logs; maybe no model was provided or model couldn't be loaded?"
            )
        return self

    def with_prompt(
        self, lang: str = "en", prompt_template_config: dict[str, Any] | None = None
    ) -> "RAG":
        """Configure the RAG answer prompt.

        Defaults are loaded from ``conf/rag/localization/{language}.yml``.
        ``prompt_template_config`` remains backwards compatible with the previous
        request-body shape: ``{templates: {language: template_str}, input_variables: variables_list}``.
        It may also contain ``profile`` and/or a direct ``template`` override.
        """
        prompt_config = resolve_rag_prompt_config(
            self.get_rag_language(lang), prompt_template_config
        )
        self._prompt = PromptTemplate(
            template=prompt_config.template,
            input_variables=prompt_config.input_variables,
        )
        return self

    def documents_from(
        self,
        documents: list[str] | list[tuple[str, dict]],
        lang: str = "en",
        concat_by: str = None,
        concat_str: str = "\n\n",
    ) -> dict[str, Document]:
        _source_str_map = {"en": "Source", "de": "Quelle"}
        if len(documents) == 0:
            logging.warning("No documents given!")
            return {}
        with_metadata = isinstance(documents[0], tuple)
        if with_metadata and concat_by is not None:
            documents = self._concatenate_by_metadata(documents, concat_by, concat_str)
        return {
            f"[{ind}]": Document(
                page_content=f"{_source_str_map.get(self.get_rag_language(lang))} [{ind}]: "
                + (d[0] if with_metadata else d),
                metadata=d[1] if with_metadata else {},
            )
            for ind, d in enumerate(documents)
        }

    def with_documents(
        self,
        documents: list[str] | list[tuple[str, dict]],
        lang: str = "en",
        concat_by: str = None,
        concat_str: str = "\n\n",
    ) -> "RAG":
        self._documents = self.documents_from(documents, lang, concat_by, concat_str)
        return self

    def build(self) -> Runnable:
        return self._prompt | self._initialized_chatter

    def no_source_answer(self) -> str:
        """Return a deterministic answer for missing retrieved documents."""
        return no_source_answer(self.language)

    def build_and_invoke(
        self, question: str, documents: dict[str, Document] | None = None
    ):
        documents = self.documents if documents is None else documents
        if not documents:
            logging.warning(
                "No RAG source documents available; skipping LLM invocation."
            )
            return True, self.no_source_answer()
        summaries = "\n\n".join(
            document.page_content for document in documents.values()
        )
        try:
            answer = (self._prompt | self._initialized_chatter).invoke(
                {"summaries": summaries, "question": question},
                return_only_outputs=True,
            )
            return True, _clean_answer(answer)
        except (LangChainException, RuntimeError, ValueError, TypeError) as e:
            logging.warning("RAG invocation failed: %s", e)
            return False, e


if __name__ == "__main__":
    import pathlib
    import sys

    from src.pipeline.load_utils import FactoryLoader
    from src.rag.text_splitters import PreprocessedSpacyTextSplitter

    _args = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:]}
    _api_key = _args.pop("api_key")

    _data = FactoryLoader.load_data(
        str(pathlib.Path("../../tmp/grascco_stem").resolve()), "grascco_stem"
    )
    _splitter = PreprocessedSpacyTextSplitter(chunk_size=400, chunk_overlap=100)
    _documents = list(
        _splitter.split_preprocessed_sentences(
            _data.processed_docs, "doc_id", keep_metadata=["doc_id", "doc_name"]
        )
    )

    chunk_embedding_store = MarqoChunkEmbeddingStore.from_config(
        index_name="grascco_stem_rag",
        url="http://localhost",
        port=8882,
        index_settings={
            "type": "structured",
            "model": "multilingual-e5-base",
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": 3,
                "splitOverlap": 1,
                "splitMethod": "sentence",
            },
            "allFields": [
                {
                    "name": "doc_id",
                    "type": "text",
                    "features": ["lexical_search", "filter"],
                },
                {
                    "name": "doc_name",
                    "type": "text",
                    "features": ["lexical_search", "filter"],
                },
                {"name": "text", "type": "text", "features": ["lexical_search"]},
            ],
            "tensorFields": ["text"],
        },
    )

    if not chunk_embedding_store.is_filled():
        docs = [
            {
                "text": d,
                "doc_id": t[1]["doc_id"],
                "doc_name": t[1]["doc_name"],
            }
            for t in _documents
            for d in t[0]
        ]
        chunk_embedding_store.add_chunks(docs)

    question_rag = "Welche Dokumente beinhalten psychotische Diagnosen?"
    result = (
        RAG.with_chatter(api_key=_api_key, language="de")
        .with_prompt(lang="de")
        .with_documents(
            list(
                zip(
                    *itemgetter(1, -1)(
                        extract_text_from_highlights(
                            chunk_embedding_store.get_chunks(
                                question_rag,
                                # filter_by={"doc_id": ["986d8ddb-c1fb-4c68-b553-61a2cec3755c"]}
                                filter_by=None,
                            ),
                            token_limit=150,
                            lang="de",
                        )
                    )
                )
            ),
            concat_by="doc_id",
        )
        .build_and_invoke(question_rag)
    )
    logging.info("%s", result)
