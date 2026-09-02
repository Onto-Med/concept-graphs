---
type: Module
title: src.gui package
description: Optional Streamlit client for operating the Concept Graphs Flask API.
tags: [module, gui, streamlit, api-client]
status: draft
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: gui-app
    resource: /src/gui/app.py
    title: Streamlit application
  - id: gui-client
    resource: /src/gui/api_client.py
    title: GUI API client
---

# Responsibility

`src.gui` provides an optional Streamlit frontend for human operation of an already-running Concept Graphs Flask API. It is intentionally a thin client and does not import or mutate pipeline internals directly.

# Main screens

The GUI is organized into four tabs:

1. **Pipeline**: load default or process-specific pipeline configuration, edit the advanced JSON payload, choose document-server or ZIP-upload input, start `/pipeline`, and poll `/status` with a progress bar inferred from pipeline step statuses.
2. **Graphs**: inspect `/graph/statistics`, load individual `/graph/{id}` payloads, render a local NetworkX/SVG overview, or display the API's `draw=true` interactive HTML response.
3. **RAG**: initialize `/rag/init` with session-only provider credentials/configuration and ask `/rag/question` with optional document filters.
4. **Query Expansion**: send `/query-expansion` requests with LLM provider settings, session-only API key headers, category selection, prompt profile selection, and optional local grounding source configurations.

# Runtime assumptions

The Flask API must be running separately. The Streamlit sidebar stores a configurable API base URL, process name, language, process listing, stop, and delete controls. Provider API keys are entered as password fields and only used for the active Streamlit session/request.

# Dependencies

The optional `gui` dependency group adds `streamlit` and `requests`. Existing project dependencies provide `networkx`, used by the local SVG graph renderer.
