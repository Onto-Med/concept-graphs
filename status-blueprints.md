# Flask Blueprint Refactor Status

## Phase 1: convert static routes

Completed.

### What changed

Converted `src/api/routes/static.py` from direct app-route registration to a Blueprint factory.

Before:

```python
def register_static_routes(app_context):
    @app_context.app.route("/", methods=["GET"])
    def index():
        ...
```

After:

```python
def create_static_blueprint(app_context):
    blueprint = Blueprint("static_routes", __name__)

    @blueprint.route("/", methods=["GET"])
    def index():
        ...

    return blueprint
```

`src/api/routes/__init__.py` now registers this blueprint with:

```python
app_context.app.register_blueprint(create_static_blueprint(app_context))
```

### Routes affected

```text
GET /
GET /openapi
```

### Scope

This was intentionally the smallest route conversion:

- no behavior changes
- no URL changes
- no other route modules converted yet
- still uses the current `AppContext` dependency style

### Validation

Ran compile check and Flask smoke test successfully.

## Phase 2: convert status routes

Completed.

### What changed

Converted `src/api/routes/status.py` from direct app-route registration to a Blueprint factory.

Before:

```python
def register_status_routes(app_context):
    @app_context.app.route("/status/rag", methods=["GET"])
    def get_rag_status():
        ...
```

After:

```python
def create_status_blueprint(app_context):
    blueprint = Blueprint("status_routes", __name__)

    @blueprint.route("/status/rag", methods=["GET"])
    def get_rag_status():
        ...

    return blueprint
```

`src/api/routes/__init__.py` now registers this blueprint with:

```python
app_context.app.register_blueprint(create_status_blueprint(app_context))
```

### Routes affected

```text
GET/POST /status/document-server
GET      /status/rag
```

### Scope

This phase kept behavior and URLs unchanged. It only changed route registration style for the status module.

### Validation

Ran Black, compile check, and Flask smoke tests successfully:

```bash
uv run --group test black src/api/routes/status.py src/api/routes/__init__.py
uv run python -m compileall -q main.py main_methods.py main_utils.py src test
```

Smoke-tested:

```text
GET /status/rag              -> 404 when RAG is not initialized
GET /status/document-server  -> 200 with current placeholder GET behavior
```

Result:

```text
blueprint phase 2 validation ok
```

## Next proposed phase

Convert another relatively small route module, likely `src/api/routes/processes.py` or `src/api/routes/pipeline.py`, to a Blueprint factory.
