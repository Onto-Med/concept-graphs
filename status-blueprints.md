# Flask Blueprint Refactor Status

## Summary

The API route modules now use Flask Blueprint factories instead of direct `app.route(...)` registration.

Blueprint factories now receive narrower dependencies instead of the full `AppContext`.

## Completed Blueprint conversion

Converted route modules:

```text
src/api/routes/static.py
src/api/routes/status.py
src/api/routes/pipeline.py
src/api/routes/processes.py
src/api/routes/graph_documents.py
src/api/routes/artifacts.py
src/api/routes/rag.py
```

Routes and URLs were preserved.

## Narrow dependency cleanup

All Blueprint factories now avoid receiving the full `AppContext`.

Current factory signatures:

```python
create_static_blueprint(app)
create_status_blueprint(app, rag)
create_artifact_blueprint(app, storage, pipeline)
create_graph_document_blueprint(app, processes, pipeline, storage)
create_pipeline_blueprint(app, processes, pipeline, storage)
create_process_blueprint(app, processes, pipeline, storage)
create_rag_blueprint(rag, processes, storage, pipeline)
```

## Pipeline cleanup

`create_pipeline_blueprint()` no longer receives `app_context`.

`run_complete_pipeline()` now receives the explicit dependencies it needs:

```python
run_complete_pipeline(app, processes, pipeline, storage)
```

Internally, `src/api/pipeline.py` creates a small `PipelineRouteContext` for the existing orchestration helper stack:

```python
@dataclass
class PipelineRouteContext:
    app: flask.Flask
    processes: object
    pipeline: object
    storage: object
```

This keeps the external Blueprint dependency surface narrow while avoiding a risky rewrite of the whole pipeline orchestration module.

## RAG cleanup

`create_rag_blueprint()` no longer receives `app_context`.

The background vector-store fill helper was narrowed from:

```python
fill_chunk_vectorstore(process, app_context, **kwargs)
```

to:

```python
fill_chunk_vectorstore(process, rag, storage, pipeline, **kwargs)
```

The RAG initialization thread now receives only those narrower dependencies.

## Route registration

`src/api/routes/__init__.py` passes explicit subcontexts to every Blueprint factory.

## Validation

Ran Black successfully:

```bash
uv run --group test black src/api/routes src/api/pipeline.py main_methods.py
```

Ran compile check successfully:

```bash
uv run python -m compileall -q main.py main_methods.py main_utils.py src test
```

Ran Flask smoke checks successfully:

```text
GET /                                                 -> 200
GET /openapi                                          -> 200
GET /pipeline/configuration?default=true&language=en  -> 200
GET /status/rag                                       -> 404 when RAG is not initialized
GET /processes                                        -> 404 when no processes exist
GET /rag/question?q=test                              -> 404 when RAG is not initialized
```

## Remaining Blueprint-related cleanup

Potential later steps:

1. Consider an app factory:

```python
def create_app(settings: AppSettings) -> Flask:
    app = Flask(__name__)
    app_context = create_app_context(settings)
    register_routes(app_context)
    return app
```

2. Optionally introduce `url_prefix` for route groups where it makes route definitions clearer.

3. Replace `object` annotations in `PipelineRouteContext` and Blueprint factories with concrete context/protocol types.

## Full pytest status

`uv run pytest -q` still fails for pre-existing unrelated test issues:

- `NameError: name 'ig' is not defined` in `src/pruning/test_unimodal.py`
- incomplete `TestGraphCreator` fixture setup
- missing pickle fixtures under `tmp/`

These failures do not appear related to the Blueprint conversion.
