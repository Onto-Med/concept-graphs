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

### Routes affected

```text
POST /pipeline
GET  /pipeline/configuration
```

### Scope

This phase kept behavior and URLs unchanged. The heavier pipeline orchestration code in `src/api/pipeline.py` was not changed.

### Validation

Ran Black, compile check, and Flask smoke tests successfully.

## Phase 4: convert process routes

Completed.

### What changed

Converted `src/api/routes/processes.py` from direct app-route registration to a Blueprint factory.

Before:

```python
def register_process_routes(app_context):
    @app_context.app.route("/processes", methods=["GET"])
    def get_all_processes_api():
        ...
```

After:

```python
def create_process_blueprint(app_context):
    blueprint = Blueprint("process_routes", __name__)

    @blueprint.route("/processes", methods=["GET"])
    def get_all_processes_api():
        ...

    return blueprint
```

`src/api/routes/__init__.py` now registers this blueprint with:

```python
app_context.app.register_blueprint(create_process_blueprint(app_context))
```

### Routes affected

```text
GET    /processes
DELETE /processes/<process_id>/delete
GET    /processes/<process_id>/stop
GET    /status
```

### Scope

This phase kept behavior and URLs unchanged. It only changed route registration style for the process route module.

### Validation

Ran Black, compile check, and Flask smoke tests successfully:

```bash
uv run --group test black src/api/routes/processes.py src/api/routes/__init__.py
uv run python -m compileall -q main.py main_methods.py main_utils.py src test
```

Smoke-tested:

```text
GET /processes                  -> 404 when no saved processes exist
GET /status?process=missing...  -> 404 when no process exists
URL map contains DELETE /processes/<process_id>/delete
URL map contains GET    /processes/<process_id>/stop
```

Result:

```text
blueprint phase 4 validation ok
```

## Current blueprint conversion status

Converted:

```text
src/api/routes/static.py
src/api/routes/status.py
src/api/routes/pipeline.py
src/api/routes/processes.py
```

Still pending:

```text
src/api/routes/artifacts.py
src/api/routes/graph_documents.py
src/api/routes/rag.py
```

## Next proposed phase

Convert `src/api/routes/graph_documents.py` next. It is smaller than artifacts and RAG, but includes background thread creation, so it should be validated carefully.
