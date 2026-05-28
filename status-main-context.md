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

The underlying class is still currently named `PersistentObjects`; that should be addressed in the next phase when we introduce clearer typed context dataclasses.

### Validation

Ran compile check successfully:

```bash
uv run python -m compileall -q main.py main_methods.py main_utils.py src test
```

Ran import smoke test successfully:

```bash
uv run python - <<'PY'
import main
print(type(main.app_context).__name__)
print('phase 1 imports ok')
PY
```

Result:

```text
PersistentObjects
phase 1 imports ok
```

## Next proposed phase

Phase 2 should introduce clearer context dataclasses, likely starting by renaming/replacing `PersistentObjects` with `AppContext` while preserving the existing fields for compatibility.
