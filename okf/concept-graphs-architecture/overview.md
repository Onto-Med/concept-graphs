---
type: Software System
title: Concept Graphs
description: Flask API for building, storing, inspecting, extending, and querying concept graphs from document corpora.
tags: [architecture, concept-graphs, flask, nlp, graph]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: readme
    resource: /README.md
    title: Project README
  - id: status
    resource: /STATUS.md
    title: Project Cleanup Status
  - id: src-tree
    resource: /src/
    title: Source package tree
---

# Purpose

Concept Graphs processes a text corpus into concept-level graph artifacts. It exposes this functionality through a Flask API that can:

* run a multi-step concept-graph pipeline,
* inspect serialized pipeline artifacts,
* add or remove document provenance from existing graphs,
* integrate phrase/concept metadata into an external vector store,
* initialize per-process RAG over document chunks,
* answer questions using retrieved source snippets, and
* generate LLM-based query expansions with optional source grounding.[^readme]

# Main architecture

The software is organized around four layers:

1. **HTTP/API layer**: [API package](modules/api.md) registers Flask Blueprints and delegates business logic to services and workflows.
2. **Workflow layer**: [Pipeline package](modules/pipeline.md) coordinates long-running pipeline steps, process status, background threads, and serialized artifacts.
3. **Domain layer**: [Core package](modules/core.md) performs preprocessing, embeddings, clustering, graph creation, graph simplification, graph incorporation, and metrics.
4. **External integration layer**: [Storage](modules/storage.md), [RAG](modules/rag.md), and [query expansion](modules/query-expansion.md) adapt Marqo/vector stores, prompt profiles, and LLM/chat backends.

# End-to-end data product

A successful pipeline run creates a process-scoped artifact set under `tmp/<process>/` (or another configured storage root): preprocessing data, phrase embeddings, phrase clusters, concept graphs, and optional vector-store integration metadata. These artifacts are later loaded by inspection endpoints, document-add/delete workflows, and RAG initialization.

# Important runtime convention

Most business endpoints accept a `process` query parameter. A process represents one corpus and isolates its runtime state and persisted files from other corpora. If omitted, routes default to `default`.[^readme]

[^readme]: Project README
