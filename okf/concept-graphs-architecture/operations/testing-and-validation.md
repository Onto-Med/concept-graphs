---
type: Operational Reference
title: Testing and validation
description: Test suite layout and standard validation commands for the workspace.
tags: [operations, testing, ruff, pytest]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: readme
    resource: /README.md
    title: README development notes
  - id: status
    resource: /STATUS.md
    title: Project Cleanup Status testing section
  - id: pyproject
    resource: /pyproject.toml
    title: Project tool configuration
  - id: tests
    resource: /test/
    title: Test tree
---

# Standard commands

The README/status files list these validation commands:

```bash
uv run --group test ruff format .
uv run --group test ruff check .
uv run --no-sync python -m compileall -q main.py src test
uv run --no-sync pytest -q
```

Ask the user before running the full test suite if it may be long-running in the current environment.

# Test layout

Tests live under `test/`, outside the production package. Current groups include:

* `test/api/` for app factory, route, OpenAPI, configuration, and pipeline-support tests,
* `test/core/` for data, graph, and corpus clustering behavior,
* `test/pipeline/` for document deletion and legacy utility behavior,
* `test/pruning/` for graph pruning,
* `test/query_expansion/` for query expansion service behavior,
* `test/rag/` for RAG context behavior,
* `test/storage/marqo/` for Marqo provenance behavior.

# Tooling

`pyproject.toml` configures Python `>=3.11,<3.12`, `uv`, pytest `pythonpath = ["."]`, and Ruff formatting/linting with rule groups `E`, `F`, `I`, and `UP`.[^pyproject]

[^pyproject]: Project tool configuration
