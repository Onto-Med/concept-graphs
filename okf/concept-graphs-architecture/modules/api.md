---
type: Module
title: src.api package
description: Flask API layer containing context, routes, request parsing, services, and pipeline-route support.
tags: [module, api, flask]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: context
    resource: /src/api/context.py
    title: API runtime context
  - id: routes
    resource: /src/api/routes/
    title: API route modules
  - id: services
    resource: /src/api/services/
    title: API service modules
  - id: pipeline-support
    resource: /src/api/pipeline_support/
    title: Pipeline support modules
---

# Responsibility

`src.api` is the HTTP boundary of the application. It keeps Flask-specific routing, request parsing, response shaping, and route-level orchestration separate from domain logic.

# Internal structure

* `context.py` defines grouped runtime state dataclasses. See [Runtime architecture](/runtime.md).
* `routes/` contains Blueprint factory modules for each endpoint group, including the merged `query_expansion.py` route for `POST /query-expansion`. See [API surface](/interfaces/api-surface.md).
* `request_parsing.py` defines simple request data classes and JSON parsers for pipeline, document addition, and RAG configuration.
* `responses.py` defines HTTP status values.
* `services/` contains route-adjacent business helpers for artifacts, configuration loading, document-server checks, pipeline query params, process deletion, and RAG vector-store initialization.
* `pipeline_support/` decomposes `POST /pipeline` into parsing, vector-store normalization, document-server loading, step preparation, and thread/response handling.

# Design notes

The API layer generally delegates long-running or domain-heavy work to [Pipeline package](/modules/pipeline.md), [Core package](/modules/core.md), [RAG package](/modules/rag.md), [Query expansion package](/modules/query-expansion.md), and [Storage package](/modules/storage.md). Route functions should stay thin and should preserve the app-factory/context pattern.
