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

Ran compile check successfully:

```bash
uv run python -m compileall -q main.py main_methods.py main_utils.py src test
```

Ran import smoke test successfully:

```bash
uv run python - <<'PY'
import main
from main_utils import AppContext, PersistentObjects
assert isinstance(main.app_context, AppContext)
assert PersistentObjects is AppContext
print(type(main.app_context).__name__)
print('phase 2 imports ok')
PY
```

Result:

```text
AppContext
phase 2 imports ok
```

## Next proposed phase

Phase 3 should group the flat runtime fields into smaller dataclasses, while preserving compatibility properties where useful.

Likely groups:

```text
AppContext.processes.running
AppContext.processes.threads
AppContext.pipeline.active_objects
AppContext.storage.file_storage_dir
AppContext.rag.active
```
