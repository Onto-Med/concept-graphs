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

Ran compile check successfully:

```bash
uv run python -m compileall -q main.py main_methods.py main_utils.py src test
```

Ran Flask smoke test successfully:

```bash
uv run python - <<'PY'
import main
client = main.app_context.app.test_client()
for path in ['/', '/openapi']:
    response = client.get(path)
    assert response.status_code == 200, (path, response.status_code)
print('blueprint phase 1 validation ok')
PY
```

Result:

```text
blueprint phase 1 validation ok
```

## Next proposed phase

Convert another low-risk route module, likely `src/api/routes/status.py`, to a Blueprint factory.
