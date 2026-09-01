---
type: Operational Reference
title: Version management
description: Scripted synchronization of project, API, Docker, and README version references.
tags: [operations, versioning, release, scripts]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: script
    resource: /src/scripts/set_version.py
    title: Version synchronization script
  - id: readme
    resource: /README.md
    title: README development notes
  - id: pyproject
    resource: /pyproject.toml
    title: Project version
---

# Current version

The merged branch updates the project version to `1.1.2` in `pyproject.toml` and README Docker image examples.

# Script

Use `src/scripts/set_version.py` to synchronize version references across project-owned files without blindly replacing dependency versions:

```bash
uv run --no-sync python -m src.scripts.set_version 1.2.0
```

Preview or check without writing:

```bash
uv run --no-sync python -m src.scripts.set_version 1.2.0 --dry-run
uv run --no-sync python -m src.scripts.set_version 1.2.0 --check
```

# Files considered by the script

The script updates or checks `VERSION`, `pyproject.toml`, `uv.lock`, `api/concept-graphs-api.yml`, README image references, Docker Compose files, and the Docker image build workflow when present.[^script]

[^script]: Version synchronization script
