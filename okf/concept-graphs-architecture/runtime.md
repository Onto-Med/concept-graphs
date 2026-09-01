---
type: Runtime Architecture
title: Flask runtime and application context
description: App factory and grouped shared state stored under Flask extensions.
tags: [runtime, flask, context, process-state]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: main
    resource: /main.py
    title: App factory entrypoint
  - id: context
    resource: /src/api/context.py
    title: Runtime context dataclasses
  - id: processes
    resource: /src/pipeline/processes.py
    title: Process/thread helpers
---

# Entry point

`main.py` exposes the Flask application factory `create_app()`. The factory configures logging, creates a Flask app, creates shared runtime context, registers routes, and stores the context at:

```python
app.extensions["concept_graphs_context"]
```

Running `python main.py` starts the development server on `0.0.0.0:9010`.[^main]

# Context objects

`src/api/context.py` defines dataclasses that group mutable runtime state:

| Context | Responsibility |
|---|---|
| `ProcessContext` | Running process status and background thread registry. |
| `PipelineContext` | Active in-memory pipeline step objects by process. |
| `StorageContext` | File storage root, default `tmp/`. |
| `RagContext` | Per-process active RAG components. |
| `AppContext` | Composes Flask app plus all grouped contexts. |
| `ActiveRAG` | A ready/initializing RAG object, its chunk vector store, and its process name. |

`AppContext` also provides compatibility properties for older names such as `running_processes`, `current_active_pipeline_objects`, and `active_rag`.[^context]

# Startup behavior

`create_app_context()` ensures the storage directory exists and calls `populate_running_processes()` so serialized or known processes can be reflected in runtime status after startup.[^main]

# Concurrency model

Long-running operations are started on `StoppableThread` instances and tracked in `ProcessContext.threads`. Process status is kept separately in `ProcessContext.running`. This lets the API respond while pipeline, RAG initialization, or document-addition work continues in the background.

# Related concepts

* [Pipeline package](modules/pipeline.md) explains thread/status management.
* [Artifacts and storage](operations/artifacts-and-storage.md) explains serialized artifacts and active-object loading.

[^main]: App factory entrypoint
[^context]: Runtime context dataclasses
