from pathlib import Path

from main import create_app, create_app_context
from src.api.context import AppContext

EXPECTED_ROUTES = {
    "/",
    "/openapi",
    "/pipeline",
    "/pipeline/configuration",
    "/processes",
    "/status",
    "/status/document-server",
    "/status/rag",
    "/rag/init",
    "/rag/question",
    "/graph/document/<path_arg>",
    "/graph/document/add/status",
    "/preprocessing/<path_arg>",
    "/embedding/<path_arg>",
    "/clustering/<path_arg>",
    "/graph/<path_arg>",
    "/processes/<process_id>/delete",
    "/processes/<process_id>/stop",
}


def test_create_app_registers_context_and_expected_routes(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])

    assert "concept_graphs_context" in app.extensions
    assert isinstance(app.extensions["concept_graphs_context"], AppContext)
    assert EXPECTED_ROUTES.issubset({rule.rule for rule in app.url_map.iter_rules()})


def test_create_app_context_initializes_grouped_state_and_storage(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    context = app.extensions["concept_graphs_context"]

    assert context.app is app
    assert context.processes.running == {}
    assert context.processes.threads == {}
    assert context.pipeline.active_objects == {}
    assert context.rag.active_by_process == {}
    assert context.storage.file_storage_dir == Path(tmp_path)
    assert context.storage.file_storage_dir.exists()


def test_app_context_compatibility_aliases(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    context = app.extensions["concept_graphs_context"]

    assert context.running_processes is context.processes.running
    assert context.pipeline_threads_store is context.processes.threads
    assert context.current_active_pipeline_objects is context.pipeline.active_objects
    assert context.file_storage_dir is context.storage.file_storage_dir
    assert context.active_rag is None


def test_create_app_context_can_be_used_directly(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path / "app"), logging_setup_tuples=[])
    direct_context = create_app_context(app, file_storage_dir=str(tmp_path / "direct"))

    assert direct_context.storage.file_storage_dir == tmp_path / "direct"
    assert direct_context.storage.file_storage_dir.exists()
