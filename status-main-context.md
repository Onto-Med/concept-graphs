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

No behavior was changed yet. Grouping state into smaller subcontexts is reserved for Phase 3.

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

This phase changed the internal shape of `AppContext`, but intentionally did not update every route and service to use the nested fields yet. That is Phase 4.

### Validation

Ran compile check successfully:

```bash
uv run python -m compileall -q main.py main_methods.py main_utils.py src test
```

Ran compatibility smoke test successfully:

```bash
uv run python - <<'PY'
import main
from main_utils import AppContext
ctx = main.app_context
assert isinstance(ctx, AppContext)
assert ctx.running_processes is ctx.processes.running
assert ctx.pipeline_threads_store is ctx.processes.threads
assert ctx.current_active_pipeline_objects is ctx.pipeline.active_objects
assert ctx.file_storage_dir is ctx.storage.file_storage_dir
assert ctx.active_rag is ctx.rag.active
print('phase 3 compatibility aliases ok')
PY
```

Also smoke-tested basic Flask endpoints:

```text
/        200
/openapi 200
```

## Next proposed phase

Phase 4 should start replacing compatibility-property usage with explicit grouped access where it improves clarity, for example:

```python
app_context.running_processes
```

to:

```python
app_context.processes.running
```

and:

```python
app_context.active_rag
```

to:

```python
app_context.rag.active
```

This can be done module-by-module to keep the diff reviewable.
