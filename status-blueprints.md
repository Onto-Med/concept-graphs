# Flask Blueprint Refactor Status

## Phase 1: convert static routes

Completed.

### What changed

Converted `src/api/routes/static.py` from direct app-route registration to a Blueprint factory.

### Routes affected

```text
GET /
GET /openapi
```

### Validation

Ran compile check and Flask smoke test successfully.

## Phase 2: convert status routes

Completed.

### What changed

Converted `src/api/routes/status.py` from direct app-route registration to a Blueprint factory.

### Routes affected

```text
GET/POST /status/document-server
GET      /status/rag
```

### Validation

Ran Black, compile check, and Flask smoke tests successfully.

## Phase 3: convert pipeline routes

Completed.

### What changed

Converted `src/api/routes/pipeline.py` from direct app-route registration to a Blueprint factory.

Before:

```python
def register_pipeline_routes(app_context):
    @app_context.app.route("/pipeline", methods=["POST"])
    def complete_pipeline():
        ...
```

After:

```python
def create_pipeline_blueprint(app_context):
    blueprint = Blueprint("pipeline_routes", __name__)

    @blueprint.route("/pipeline", methods=["POST"])
    def complete_pipeline():
        ...

    return blueprint
```

`src/api/routes/__init__.py` now registers this blueprint with:

```python
app_context.app.register_blueprint(create_pipeline_blueprint(app_context))
```

### Routes affected

```text
POST /pipeline
GET  /pipeline/configuration
```

### Scope

This phase kept behavior and URLs unchanged. It only changed route registration style for the pipeline route module.

The heavier pipeline orchestration code in `src/api/pipeline.py` was not changed.

### Validation

Ran Black, compile check, and Flask smoke tests successfully:

```bash
uv run --group test black src/api/routes/pipeline.py src/api/routes/__init__.py
uv run python -m compileall -q main.py main_methods.py main_utils.py src test
```

Smoke-tested:

```text
GET /pipeline/configuration?default=true&language=en -> 200 or 404 depending on config availability
URL map contains POST /pipeline
URL map contains GET /pipeline/configuration
```

Result:

```text
blueprint phase 3 validation ok
```

## Current blueprint conversion status

Converted:

```text
src/api/routes/static.py
src/api/routes/status.py
src/api/routes/pipeline.py
```

Still pending:

```text
src/api/routes/artifacts.py
src/api/routes/graph_documents.py
src/api/routes/processes.py
src/api/routes/rag.py
```

## Next proposed phase

Convert `src/api/routes/processes.py` next. It is more involved than static/status/pipeline, but still smaller than artifacts and RAG.
