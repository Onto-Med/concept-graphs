from io import BytesIO
from types import SimpleNamespace

from flask import Flask
from werkzeug.datastructures import FileStorage

from src.api.pipeline_support.models import DEFAULT_VECTOR_STORE
from src.api.pipeline_support.request_data import (
    parse_json_pipeline_request,
    parse_multipart_pipeline_request,
)
from src.api.request_parsing import (
    parse_document_adding_json,
    parse_pipeline_config_json,
    parse_rag_config_json,
)
from src.api.services.configuration import read_exclusion_ids
from src.pipeline.status import PipelineQueryParams


def _query_params(process_name="proc"):
    return PipelineQueryParams(
        process_name=process_name,
        language="en",
        skip_present=True,
        omitted_pipeline_steps=[],
        return_statistics=False,
    )


def test_parse_pipeline_config_json_accepts_vectorstore_aliases():
    parsed = parse_pipeline_config_json(
        {
            "name": "My Corpus",
            "language": "de",
            "document_server": {"replace_keys": {"text": "body"}},
            "vector_store_server": {"url": "http://vector", "port": 1234},
            "config": {
                "data": {"spacy_model": "model"},
                "embedding": {"model": "embedding"},
                "clustering": {"algorithm": "kmeans"},
                "graph": {"restrict_to_cluster": True},
            },
        }
    )

    assert parsed.name == "my_corpus"
    assert parsed.language == "de"
    assert parsed.vectorstore_server == {"url": "http://vector", "port": 1234}
    assert parsed.data.spacy_model == "model"
    assert parsed.graph.restrict_to_cluster is True


def test_parse_json_pipeline_request_requires_document_server():
    parsed_config = parse_pipeline_config_json({"name": "proc", "config": {}})

    with Flask(__name__).app_context():
        request_data, error = parse_json_pipeline_request(
            parsed_config, _query_params("proc")
        )

    assert request_data is None
    response, status = error
    assert status == 400
    assert response.json["name"] == "proc"
    assert "No configuration entry" in response.json["error"]


def test_parse_json_pipeline_request_uses_defaults_and_config_sections():
    parsed_config = parse_pipeline_config_json(
        {
            "name": "proc",
            "document_server": {
                "url": "http://docs",
                "replace_keys": {"text": "content"},
                "label_key": "label",
            },
            "config": {
                "data": {"n_process": 1},
                "embedding": {"model": "m"},
                "clustering": {"algorithm": "kmeans"},
                "graph": {"restrict_to_cluster": False},
            },
        }
    )

    request_data, error = parse_json_pipeline_request(parsed_config, _query_params())

    assert error is None
    assert request_data.document_server_config["url"] == "http://docs"
    assert request_data.vector_store_config == DEFAULT_VECTOR_STORE
    assert request_data.replace_keys == {"text": "content"}
    assert request_data.label_getter == "label"
    assert request_data.content_type_json is True
    assert request_data.data_config.n_process == 1


def test_parse_multipart_pipeline_request_rejects_missing_data_and_document_server(
    tmp_path,
):
    app = Flask(__name__)
    app_context = SimpleNamespace(storage=SimpleNamespace(file_storage_dir=tmp_path))

    with app.test_request_context("/pipeline", method="POST", data={}):
        request_data, error = parse_multipart_pipeline_request(
            app_context, _query_params("proc")
        )

    assert request_data is None
    response, status = error
    assert status == 400
    assert response.json["name"] == "proc"
    assert "Neither data provided" in response.json["error"]


def test_parse_multipart_pipeline_request_saves_uploads_and_reads_vector_config(
    tmp_path,
):
    app = Flask(__name__)
    app_context = SimpleNamespace(storage=SimpleNamespace(file_storage_dir=tmp_path))

    with app.test_request_context(
        "/pipeline",
        method="POST",
        data={
            "data": (BytesIO(b"zip-content"), "docs.zip"),
            "labels": (BytesIO(b"labels"), "labels.yaml"),
            "vectorstore_server_config": (
                BytesIO(b"url: http://vector\nport: 9999\n"),
                "vector.yaml",
            ),
        },
    ):
        request_data, error = parse_multipart_pipeline_request(
            app_context, _query_params("proc")
        )

    assert error is None
    assert request_data.data_upload is True
    assert request_data.data.read_bytes() == b"zip-content"
    assert request_data.labels.read_bytes() == b"labels"
    assert request_data.vector_store_config == {"url": "http://vector", "port": 9999}


def test_parse_document_adding_json_and_rag_config_aliases():
    document_config = parse_document_adding_json(
        {
            "language": "en",
            "documents": [{"id": "doc-1"}],
            "vectorstore_server": {"url": "http://vector"},
        }
    )
    rag_config = parse_rag_config_json(
        {
            "api_key": "key",
            "language": "de",
            "chatter": {"model": "m"},
            "vector-store_server": {"url": "http://vector"},
            "prompt_template": {},
        }
    )

    assert document_config.documents == [{"id": "doc-1"}]
    assert document_config.vectorstore_server == {"url": "http://vector"}
    assert rag_config.api_key == "key"
    assert rag_config.language == "de"
    assert rag_config.vectorstore_server == {"url": "http://vector"}
    assert rag_config.prompt_template is None


def test_read_exclusion_ids_from_string_and_filestorage():
    assert read_exclusion_ids("[1, 2, 3]") == [1, 2, 3]
    assert read_exclusion_ids("not-a-list") == []
    assert read_exclusion_ids(
        FileStorage(stream=BytesIO(b"4,5"), filename="exclude.txt")
    ) == [4, 5]
