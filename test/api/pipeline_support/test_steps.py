from types import SimpleNamespace

from src.api.pipeline_support.models import PipelineRequestData
from src.api.pipeline_support.steps import (
    embedding_storage_method,
    pipeline_process_definitions,
    prepare_pipeline_processes,
)
from src.pipeline.status import PipelineQueryParams, ProcessStatus, StepsName


class FakeProcessor:
    existing_steps = set()
    deleted = []
    configured = []

    def __init__(self, app, file_storage):
        self.app = app
        self.file_storage = file_storage
        self.storage_method = "vectorstore"
        self.config = {}
        self.process_name = None

    def has_process(self, process_name):
        return self.step_name in self.existing_steps

    def delete_process(self, process_name):
        self.deleted.append((self.step_name, process_name))

    def read_labels(self, labels):
        self.labels = labels

    def read_data(self, data, replace_keys=None, label_getter=None):
        self.data = data
        self.replace_keys = replace_keys
        self.label_getter = label_getter


class FakeDataProcessor(FakeProcessor):
    step_name = StepsName.DATA


class FakeEmbeddingProcessor(FakeProcessor):
    step_name = StepsName.EMBEDDING


class FakeClusteringProcessor(FakeProcessor):
    step_name = StepsName.CLUSTERING


class FakeGraphProcessor(FakeProcessor):
    step_name = StepsName.GRAPH


class FakeIntegrationProcessor(FakeProcessor):
    step_name = StepsName.INTEGRATION


def _query_params(
    skip_present=True,
    omitted_pipeline_steps=None,
    process_name="proc",
):
    return PipelineQueryParams(
        process_name=process_name,
        language="en",
        skip_present=skip_present,
        omitted_pipeline_steps=omitted_pipeline_steps or [],
        return_statistics=False,
    )


def _app_context(tmp_path):
    return SimpleNamespace(
        app=SimpleNamespace(logger=SimpleNamespace(info=lambda *args, **kwargs: None)),
        storage=SimpleNamespace(file_storage_dir=tmp_path),
        pipeline=SimpleNamespace(active_objects={}),
        processes=SimpleNamespace(running={}),
    )


def _patch_processors(monkeypatch):
    import src.api.pipeline_support.steps as steps

    monkeypatch.setattr(steps, "PreprocessingUtil", FakeDataProcessor)
    monkeypatch.setattr(steps, "PhraseEmbeddingUtil", FakeEmbeddingProcessor)
    monkeypatch.setattr(steps, "ClusteringUtil", FakeClusteringProcessor)
    monkeypatch.setattr(steps, "GraphCreationUtil", FakeGraphProcessor)
    monkeypatch.setattr(steps, "ConceptGraphIntegrationUtil", FakeIntegrationProcessor)
    monkeypatch.setattr(
        steps,
        "read_config",
        lambda app, processor, process_type, process_name, config, language, mode: (
            setattr(processor, "process_name", process_name)
        ),
    )
    monkeypatch.setattr(
        steps,
        "load_skipped_step",
        lambda app_context, query_params, step_name, vector_store_config: (
            app_context.pipeline.active_objects[query_params.process_name].update(
                {step_name: f"loaded-{step_name}"}
            )
        ),
    )


def test_pipeline_process_definitions_adds_integration_only_with_vector_store():
    request_data = PipelineRequestData()

    without_vectorstore = pipeline_process_definitions(None, request_data)
    with_vectorstore = pipeline_process_definitions(
        {"url": "http://vector"}, request_data
    )

    assert [step for step, *_ in without_vectorstore] == [
        StepsName.DATA,
        StepsName.EMBEDDING,
        StepsName.CLUSTERING,
        StepsName.GRAPH,
    ]
    assert [step for step, *_ in with_vectorstore] == [
        StepsName.DATA,
        StepsName.EMBEDDING,
        StepsName.CLUSTERING,
        StepsName.GRAPH,
        StepsName.INTEGRATION,
    ]


def test_embedding_storage_method_selects_vectorstore_only_when_configured():
    processor = SimpleNamespace(storage_method="vectorstore")

    assert embedding_storage_method(processor, None) == ("pickle", None)
    assert embedding_storage_method(processor, {"url": "http://vector"}) == (
        "vectorstore",
        {"url": "http://vector"},
    )

    processor.storage_method = "pickle"
    assert embedding_storage_method(processor, {"url": "http://vector"}) == (
        "pickle",
        None,
    )


def test_prepare_pipeline_processes_skips_and_loads_present_steps(
    monkeypatch, tmp_path
):
    _patch_processors(monkeypatch)
    FakeProcessor.existing_steps = {StepsName.DATA, StepsName.EMBEDDING}
    app_context = _app_context(tmp_path)

    prepared = prepare_pipeline_processes(
        app_context,
        _query_params(skip_present=True),
        PipelineRequestData(data="data"),
        vector_store_config=None,
    )

    assert [step for _, _, step in prepared.processes_threading] == [
        StepsName.CLUSTERING,
        StepsName.GRAPH,
    ]
    assert app_context.pipeline.active_objects["proc"][StepsName.DATA] == "loaded-data"
    assert app_context.pipeline.active_objects["proc"][StepsName.EMBEDDING] == (
        "loaded-embedding"
    )
    statuses = app_context.processes.running["proc"]["status"]
    finished_steps = [
        status["name"]
        for status in statuses
        if status["status"] == ProcessStatus.FINISHED
    ]
    assert finished_steps == [StepsName.DATA, StepsName.EMBEDDING]


def test_prepare_pipeline_processes_omitted_step_is_not_scheduled(
    monkeypatch, tmp_path
):
    _patch_processors(monkeypatch)
    FakeProcessor.existing_steps = set()
    app_context = _app_context(tmp_path)

    prepared = prepare_pipeline_processes(
        app_context,
        _query_params(omitted_pipeline_steps=[StepsName.INTEGRATION]),
        PipelineRequestData(),
        vector_store_config={"url": "http://vector"},
    )

    assert StepsName.INTEGRATION not in [
        step for _, _, step in prepared.processes_threading
    ]
    integration_status = next(
        status
        for status in app_context.processes.running["proc"]["status"]
        if status["name"] == StepsName.INTEGRATION
    )
    assert integration_status["status"] == ProcessStatus.FINISHED


def test_prepare_pipeline_processes_deletes_existing_step_when_not_skipping(
    monkeypatch, tmp_path
):
    _patch_processors(monkeypatch)
    FakeProcessor.existing_steps = {StepsName.DATA}
    FakeProcessor.deleted = []
    app_context = _app_context(tmp_path)

    prepared = prepare_pipeline_processes(
        app_context,
        _query_params(skip_present=False),
        PipelineRequestData(data="data"),
        vector_store_config=None,
    )

    assert FakeProcessor.deleted == [(StepsName.DATA, "proc")]
    assert [step for _, _, step in prepared.processes_threading] == [
        StepsName.DATA,
        StepsName.EMBEDDING,
        StepsName.CLUSTERING,
        StepsName.GRAPH,
    ]
