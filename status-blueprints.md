# Flask Blueprint Refactor Status

## Summary

The API route modules have been converted from direct `app.route(...)` registration to Flask Blueprint factories.

Each route module now owns a `create_*_blueprint(app_context)` function, and `src/api/routes/__init__.py` registers all returned blueprints on the Flask app.

## Completed phases

### Phase 1: static routes

Converted `src/api/routes/static.py`.

Routes:

```text
GET /
GET /openapi
```

### Phase 2: status routes

Converted `src/api/routes/status.py`.

Routes:

```text
GET/POST /status/document-server
GET      /status/rag
```

### Phase 3: pipeline routes

Converted `src/api/routes/pipeline.py`.

Routes:

```text
POST /pipeline
GET  /pipeline/configuration
```

The heavier orchestration code in `src/api/pipeline.py` was not changed.

### Phase 4: process routes

Converted `src/api/routes/processes.py`.

Routes:

```text
GET    /processes
DELETE /processes/<process_id>/delete
GET    /processes/<process_id>/stop
GET    /status
```

### Phase 5: graph document routes

Converted `src/api/routes/graph_documents.py`.

Routes:

```text
POST/DELETE /graph/document/<path_arg>
GET         /graph/document/add/status
```

### Phase 6: artifact routes

Converted `src/api/routes/artifacts.py`.

Routes:

```text
GET      /preprocessing/<path_arg>
GET      /embedding/<path_arg>
GET      /clustering/<path_arg>
GET/POST /graph/<path_arg>
```

### Phase 7: RAG routes

Converted `src/api/routes/rag.py`.

Routes:

```text
POST     /rag/init
GET/POST /rag/question
```

## Current structure

`src/api/routes/__init__.py` now imports blueprint factories:

```python
from src.api.routes.artifacts import create_artifact_blueprint
from src.api.routes.graph_documents import create_graph_document_blueprint
from src.api.routes.pipeline import create_pipeline_blueprint
from src.api.routes.processes import create_process_blueprint
from src.api.routes.rag import create_rag_blueprint
from src.api.routes.static import create_static_blueprint
from src.api.routes.status import create_status_blueprint
```

and registers them with:

```python
app_context.app.register_blueprint(create_static_blueprint(app_context))
app_context.app.register_blueprint(create_artifact_blueprint(app_context))
app_context.app.register_blueprint(create_graph_document_blueprint(app_context))
app_context.app.register_blueprint(create_pipeline_blueprint(app_context))
app_context.app.register_blueprint(create_process_blueprint(app_context))
app_context.app.register_blueprint(create_rag_blueprint(app_context))
app_context.app.register_blueprint(create_status_blueprint(app_context))
```

## Validation

Ran Black successfully:

```bash
uv run --group test black src/api/routes
```

Ran compile check successfully:

```bash
uv run python -m compileall -q main.py main_methods.py main_utils.py src test
```

Ran Flask route smoke checks successfully for converted routes, including URL-map validation for:

```text
/preprocessing/<path_arg>
/embedding/<path_arg>
/clustering/<path_arg>
/graph/<path_arg>
/graph/document/<path_arg>
/graph/document/add/status
/rag/init
/rag/question
```

Also smoke-tested selected endpoint responses:

```text
GET /preprocessing/unknown       -> 400
GET /embedding/unknown           -> 400
GET /clustering/unknown          -> 400
GET /graph/document/add/status   -> 404 when no document-addition thread exists
GET /rag/question?q=test         -> 404 when RAG is not initialized
```

## Full pytest status

`uv run pytest -q` still fails for the same pre-existing unrelated test issues:

- `NameError: name 'ig' is not defined` in `src/pruning/test_unimodal.py`
- incomplete `TestGraphCreator` fixture setup
- missing pickle fixtures under `tmp/`

These failures do not appear related to the Blueprint conversion.

## Result

All route modules under `src/api/routes/` now use Blueprint factories. The project has a more idiomatic Flask structure while preserving the existing URL paths and behavior.
