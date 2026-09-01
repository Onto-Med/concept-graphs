---
type: Interface
title: HTTP API surface
description: Main Flask endpoint groups and where they are implemented.
tags: [api, flask, endpoints, openapi]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: readme
    resource: /README.md
    title: Project README API overview
  - id: routes
    resource: /src/api/routes/
    title: Flask route modules
  - id: openapi
    resource: /api/concept-graphs-api.yml
    title: OpenAPI specification
---

# Endpoint groups

| Group | Main endpoints | Route module |
|---|---|---|
| Pipeline | `POST /pipeline`, `GET /pipeline/configuration` | `src/api/routes/pipeline.py` |
| Artifacts | `/preprocessing/*`, `/embedding/*`, `/clustering/*`, `/graph/*` | `src/api/routes/artifacts.py` |
| Graph documents | `POST /graph/document/add`, `GET /graph/document/add/status`, `DELETE /graph/document/{document_id}` | `src/api/routes/graph_documents.py` |
| Processes | `GET /processes`, `GET /status`, stop/delete endpoints | `src/api/routes/processes.py` |
| Status | `POST /status/document-server`, `GET /status/rag` | `src/api/routes/status.py` |
| RAG | `POST /rag/init`, `GET/POST /rag/question` | `src/api/routes/rag.py` |
| Static docs | `/`, `/openapi`, static assets | `src/api/routes/static.py` |

# Request conventions

Most endpoints use `process` as a query parameter, normalized through common parsing helpers. Pipeline requests accept JSON or multipart form data. RAG and graph-document addition expect JSON bodies for their main operations.[^readme]

# Response conventions

Route modules use `src/api/responses.py` for HTTP status enum values and service helpers under `src/api/services/` for response payloads.

# OpenAPI

The OpenAPI specification is kept at `api/concept-graphs-api.yml` and served through the static/docs routes. Tests include OpenAPI/Flask route parity checks.[^openapi]

[^readme]: Project README API overview
[^openapi]: OpenAPI specification
