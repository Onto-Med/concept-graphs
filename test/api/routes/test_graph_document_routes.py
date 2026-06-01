from main import create_app
from src.api.responses import HTTPResponses


class FakeThread:
    def __init__(self, target_args=None, target_kwargs=None, **kwargs):
        self.target_args = target_args
        self.target_kwargs = target_kwargs
        self.kwargs = kwargs
        self.return_value = None


def _app(tmp_path):
    return create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])


def test_graph_document_add_rejects_non_json(tmp_path):
    response = (
        _app(tmp_path)
        .test_client()
        .post("/graph/document/add?process=corpus", data="not-json")
    )

    assert response.status_code == HTTPResponses.NOT_IMPLEMENTED
    assert response.json == {"error": "Only json request body is supported."}


def test_graph_document_add_rejects_malformed_json(tmp_path):
    response = (
        _app(tmp_path)
        .test_client()
        .post(
            "/graph/document/add?process=corpus",
            json="not-an-object",
        )
    )

    assert response.status_code == HTTPResponses.BAD_REQUEST
    assert response.json == {"error": "Could not parse json provided in request."}


def test_graph_document_add_starts_thread_for_valid_json(monkeypatch, tmp_path):
    import src.api.routes.graph_documents as graph_document_routes

    started_threads = []
    monkeypatch.setattr(graph_document_routes, "StoppableThread", FakeThread)
    monkeypatch.setattr(
        graph_document_routes,
        "start_thread",
        lambda app, thread_id, thread, status: started_threads.append(
            (thread_id, thread, status)
        ),
    )
    app = _app(tmp_path)

    response = app.test_client().post(
        "/graph/document/add?process=corpus",
        json={
            "language": "en",
            "documents": [{"id": "doc-1", "name": "doc.txt", "content": "content"}],
        },
    )

    assert response.status_code == HTTPResponses.OK
    assert started_threads[0][0] == "document_addition_corpus"
    thread = app.extensions["concept_graphs_context"].processes.threads[
        "document_addition_corpus"
    ]
    assert thread.target_args[0].documents[0]["id"] == "doc-1"
    assert thread.target_kwargs["process_name"] == "corpus"


def test_graph_document_add_status_responses(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()

    assert client.get("/graph/document/add/status?process=corpus").status_code == 404

    context = app.extensions["concept_graphs_context"]
    running_thread = FakeThread()
    context.processes.threads["document_addition_corpus"] = running_thread
    assert client.get("/graph/document/add/status?process=corpus").status_code == 202

    running_thread.return_value = ({"ok": True}, HTTPResponses.OK)
    response = client.get("/graph/document/add/status?process=corpus")
    assert response.status_code == HTTPResponses.OK
    assert response.json == {"ok": True}


def test_graph_document_delete_calls_deletion_workflow(monkeypatch, tmp_path):
    import src.api.routes.graph_documents as graph_document_routes

    calls = []

    def fake_delete_document_from_concept_graphs(**kwargs):
        calls.append(kwargs)
        return {"deleted": kwargs["document_id"]}, HTTPResponses.OK

    monkeypatch.setattr(
        graph_document_routes,
        "delete_document_from_concept_graphs",
        fake_delete_document_from_concept_graphs,
    )

    response = (
        _app(tmp_path)
        .test_client()
        .delete(
            "/graph/document/doc-1?process=corpus&remove_unreferenced_nodes=false&delete_unreferenced_embeddings=true",
            json={"vectorstore_server": {"index_name": "idx"}},
        )
    )

    assert response.status_code == HTTPResponses.OK
    assert response.json == {"deleted": "doc-1"}
    assert calls[0]["document_id"] == "doc-1"
    assert calls[0]["process_name"] == "corpus"
    assert calls[0]["vectorstore_server"] == {"index_name": "idx"}
    assert calls[0]["remove_unreferenced_nodes"] is False
    assert calls[0]["delete_unreferenced_embeddings"] is True
