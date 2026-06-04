from types import SimpleNamespace

from main import create_app
from src.api.context import ActiveRAG


class FakeVectorStore:
    def __init__(self, process="default", filled=True, chunks=None):
        self.process = process
        self.filled = filled
        self.chunks = chunks
        self.chunk_requests = []

    def is_filled(self):
        return self.filled

    def get_chunks(self, question, filter_by=None, limit=10):
        self.chunk_requests.append(
            {"question": question, "filter_by": filter_by, "limit": limit}
        )
        if self.chunks is not None:
            return self.chunks
        return [{"text": f"chunk for {self.process}"}]


class FakeRAG:
    def __init__(self, language="en", answer="answer"):
        self.language = language
        self.answer = answer
        self.documents = None
        self.invocations = []

    def with_prompt(self, lang="en", prompt_template_config=None):
        self.prompt_language = lang
        self.prompt_template_config = prompt_template_config
        return self

    def documents_from(self, documents, concat_by=None):
        return {
            "[0]": SimpleNamespace(
                metadata={"doc_id": f"doc-{self.answer}"}, page_content=str(documents)
            )
        }

    def build_and_invoke(self, question, documents=None):
        self.invocations.append({"question": question, "documents": documents})
        return True, self.answer


class FakeRAGFactory:
    created = []

    @classmethod
    def with_chatter(cls, **kwargs):
        process_answer = kwargs.get("api_key", "answer")
        rag = FakeRAG(language=kwargs.get("language", "en"), answer=process_answer)
        cls.created.append(rag)
        return rag


def _app(tmp_path):
    return create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])


def test_rag_init_keeps_active_rag_per_process(monkeypatch, tmp_path):
    import src.api.routes.rag as rag_routes

    FakeRAGFactory.created = []
    monkeypatch.setattr(rag_routes, "RAG", FakeRAGFactory)
    monkeypatch.setattr(
        rag_routes,
        "initialize_chunk_vectorstore",
        lambda process, config, force_init=False: FakeVectorStore(process=process),
    )

    app = _app(tmp_path)
    client = app.test_client()

    response_a = client.post(
        "/rag/init?process=corpus_a",
        json={"api_key": "answer-a", "language": "en"},
    )
    response_b = client.post(
        "/rag/init?process=corpus_b",
        json={"api_key": "answer-b", "language": "de"},
    )

    assert response_a.status_code == 200
    assert response_b.status_code == 200

    rag_context = app.extensions["concept_graphs_context"].rag
    assert set(rag_context.active_by_process) == {"corpus_a", "corpus_b"}
    assert rag_context.active_by_process["corpus_a"].ready is True
    assert rag_context.active_by_process["corpus_b"].ready is True
    assert (
        rag_context.active_by_process["corpus_a"].rag
        is not rag_context.active_by_process["corpus_b"].rag
    )

    assert client.get("/status/rag?process=corpus_a").json == {
        "active": True,
        "error": None,
        "initializing": False,
        "name": "corpus_a",
        "vectorstore_document_count": None,
        "vectorstore_filled": True,
        "vectorstore_index": None,
    }
    assert client.get("/status/rag?process=corpus_b").json == {
        "active": True,
        "error": None,
        "initializing": False,
        "name": "corpus_b",
        "vectorstore_document_count": None,
        "vectorstore_filled": True,
        "vectorstore_index": None,
    }


def test_rag_question_uses_selected_process_without_mutating_rag(monkeypatch, tmp_path):
    import src.api.routes.rag as rag_routes

    monkeypatch.setattr(
        rag_routes,
        "extract_text_from_highlights",
        lambda results, token_limit, lang: (
            ["highlight"],
            [f"text from {results[0]['text']}"],
            [{"doc_id": "doc-1"}],
        ),
    )
    app = _app(tmp_path)
    rag_context = app.extensions["concept_graphs_context"].rag
    rag_a = FakeRAG(answer="answer-a")
    rag_b = FakeRAG(answer="answer-b")
    vector_a = FakeVectorStore(process="corpus_a")
    vector_b = FakeVectorStore(process="corpus_b")
    rag_context.active_by_process["corpus_a"] = ActiveRAG(
        rag=rag_a, vectorstore=vector_a, process="corpus_a", ready=True
    )
    rag_context.active_by_process["corpus_b"] = ActiveRAG(
        rag=rag_b, vectorstore=vector_b, process="corpus_b", ready=True
    )

    response = app.test_client().post(
        "/rag/question?process=corpus_b&q=question",
        json={"doc_ids": ["doc-1"], "limit": 7},
    )

    assert response.status_code == 200
    assert response.json["answer"] == "answer-b"
    assert vector_a.chunk_requests == []
    assert vector_b.chunk_requests == [
        {"question": "question", "filter_by": {"doc_id": ["doc-1"]}, "limit": 7}
    ]
    assert rag_a.invocations == []
    assert rag_b.invocations[0]["question"] == "question"
    assert rag_a.documents is None
    assert rag_b.documents is None


def test_rag_question_returns_no_source_answer_when_retrieval_is_empty(tmp_path):
    app = _app(tmp_path)
    rag_context = app.extensions["concept_graphs_context"].rag
    rag_instance = FakeRAG(language="de", answer="should-not-be-called")
    vector_store = FakeVectorStore(process="corpus", chunks=[])
    rag_context.active_by_process["corpus"] = ActiveRAG(
        rag=rag_instance, vectorstore=vector_store, process="corpus", ready=True
    )

    response = app.test_client().get("/rag/question?process=corpus&q=question")

    assert response.status_code == 200
    assert response.json == {
        "answer": "Keine Quelle die ich finden kann.",
        "info": "{}",
    }
    assert rag_instance.invocations == []


def test_rag_question_returns_not_found_for_uninitialized_process(tmp_path):
    response = (
        _app(tmp_path).test_client().get("/rag/question?process=missing&q=question")
    )

    assert response.status_code == 404
    assert "No active and ready rag component" in response.json
