# App Context Refactor Status

## Phase 1: terminology cleanup

Completed.

### What changed

Renamed the broad runtime object usage from `main_objects` to `app_context` throughout the codebase.

Also renamed local setup variable `persistent_objects` to `app_context` in `main.py`.

### Why

`main_objects` was vague and made the object look like a miscellaneous bag of unrelated values. `app_context` better communicates that this object contains shared application runtime context/state.

### Scope

This phase was intentionally low-risk:

- no behavior changes
- no field restructuring
- no dataclass redesign yet
- no Flask Blueprint changes

### Validation

Ran compile check and import smoke test successfully.

## Phase 2: introduce `AppContext`

Completed.

### What changed

Renamed the runtime context dataclass from `PersistentObjects` to `AppContext` in `main_utils.py`.

Updated new code imports/type hints to use:

```python
from main_utils import AppContext
```

`main.py` now creates an `AppContext` directly:

```python
app_context = AppContext(...)
```

### Compatibility

Kept a backwards-compatible alias:

```python
PersistentObjects = AppContext
```

This keeps older references or serialized objects safer while new code uses the clearer name.

### Scope

This phase preserved the existing flat field layout:

```python
app
running_processes
pipeline_threads_store
current_active_pipeline_objects
file_storage_dir
active_rag
```

No behavior was changed yet. Grouping state into smaller subcontexts was reserved for Phase 3.

### Validation

Ran compile check and import smoke test successfully.

## Phase 3: group runtime state into subcontexts

Completed.

### What changed

Added focused context dataclasses in `main_utils.py`:

```python
ProcessContext
PipelineContext
StorageContext
RagContext
```

`AppContext` now stores grouped state:

```python
AppContext.processes.running
AppContext.processes.threads
AppContext.pipeline.active_objects
AppContext.storage.file_storage_dir
AppContext.rag.active
```

`main.py` now constructs the grouped context directly:

```python
app_context = AppContext(
    app=app,
    processes=ProcessContext(running={}, threads={}),
    pipeline=PipelineContext(active_objects={}),
    storage=StorageContext(file_storage_dir=pathlib.Path(file_storage_dir)),
    rag=RagContext(),
)
```

### Compatibility

Kept compatibility properties on `AppContext`, so existing code still works:

```python
app_context.running_processes
app_context.pipeline_threads_store
app_context.current_active_pipeline_objects
app_context.file_storage_dir
app_context.active_rag
```

These now delegate to the grouped subcontexts.

Also kept:

```python
PersistentObjects = AppContext
```

### Scope

This phase changed the internal shape of `AppContext`, but intentionally did not update every route and service to use the nested fields yet. That was reserved for Phase 4.

### Validation

Ran compile check and smoke tests successfully.

## Phase 4: use explicit grouped context access

Completed.

### What changed

Replaced direct use of compatibility fields with explicit grouped context access across the codebase.

Examples:

```python
app_context.running_processes
```

became:

```python
app_context.processes.running
```

```python
app_context.pipeline_threads_store
```

became:

```python
app_context.processes.threads
```

```python
app_context.current_active_pipeline_objects
```

became:

```python
app_context.pipeline.active_objects
```

```python
app_context.file_storage_dir
```

became:

```python
app_context.storage.file_storage_dir
```

```python
app_context.active_rag
```

became:

```python
app_context.rag.active
```

### Compatibility

The compatibility properties remain on `AppContext` for now, but normal project code no longer uses them directly.

A grep check confirms no remaining direct `app_context.<old_field>` usages in Python files.

### Formatting

Ran Black after the replacements.

### Validation

Ran compile check successfully:

```bash
uv run python -m compileall -q main.py main_methods.py main_utils.py src test
```

Ran grouped-context smoke test successfully:

```bash
uv run python - <<'PY'
import main
ctx = main.app_context
assert ctx.processes.running is ctx.running_processes
assert ctx.processes.threads is ctx.pipeline_threads_store
assert ctx.pipeline.active_objects is ctx.current_active_pipeline_objects
assert ctx.storage.file_storage_dir == ctx.file_storage_dir
assert ctx.rag.active is ctx.active_rag
client = ctx.app.test_client()
for path in ['/', '/openapi']:
    assert client.get(path).status_code == 200
print('phase 4 validation ok')
PY
```

Result:

```text
phase 4 validation ok
```

## Current state

The broad application context is now clearer and more Pythonic:

- `AppContext` is the explicit top-level runtime container.
- Runtime state is grouped by responsibility.
- Project code uses grouped access rather than flat compatibility aliases.
- Backwards-compatible aliases remain available for a future cleanup pass.

## Possible next cleanup

A later phase could remove the compatibility properties and `PersistentObjects` alias once we are confident no old references or serialized objects require them.

Another good next step would be converting route registration functions to Flask Blueprints, using `AppContext` or narrower subcontexts as dependencies.
