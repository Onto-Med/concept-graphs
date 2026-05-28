# App Factory Refactor Status

## Completed

Converted `main.py` to a Flask app factory pattern without keeping global compatibility objects.

## What changed

### Added `configure_logging()`

Logging setup is now isolated in:

```python
def configure_logging(logging_setup_tuples=None) -> None:
    ...
```

### Added `create_app_context()`

Runtime state creation is now explicit:

```python
def create_app_context(app: flask.Flask, file_storage_dir: str = "tmp") -> AppContext:
    ...
```

This creates the `AppContext`, initializes storage, and populates running processes.

### Added `create_app()`

The Flask app is now created through:

```python
def create_app(...) -> flask.Flask:
    ...
```

It creates the Flask app, creates the app context, registers routes, stores the context under:

```python
app.extensions["concept_graphs_context"]
```

and returns the Flask app.

### Removed global app-context creation

Removed import-time setup like:

```python
app_context = setup(...)
register_routes(app_context)
```

Importing `main` no longer creates/registers the application immediately.

### Updated local entrypoint

Running directly still works:

```python
if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=9010)
```

### Updated Docker entrypoint

Docker now uses Waitress' factory-call mode:

```dockerfile
ENTRYPOINT [ "uv", "run", "--no-sync", "waitress-serve", "--call", "--port=9007", "main:create_app" ]
```

## Validation

Ran Black successfully:

```bash
uv run --group test black main.py
```

Ran compile check successfully:

```bash
uv run python -m compileall -q main.py main_methods.py main_utils.py src test
```

Ran app factory smoke test successfully:

```python
import main
assert not hasattr(main, "app_context")
app = main.create_app(file_storage_dir="tmp")
ctx = app.extensions["concept_graphs_context"]
assert ctx.app is app
```

Smoke-tested selected routes:

```text
GET /            -> 200
GET /openapi     -> 200
GET /status/rag  -> 404 when RAG is not initialized
GET /processes   -> 404 when no processes exist
```

Validated Waitress factory startup:

```bash
uv run --no-sync waitress-serve --call --port=9019 main:create_app
```

and confirmed:

```text
GET /openapi -> success
```

## Result

The app now follows the standard Flask app factory pattern. This makes imports cleaner, testing easier, and deployment explicit.
