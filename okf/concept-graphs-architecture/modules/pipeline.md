---
type: Module
title: src.pipeline package
description: Pipeline utilities, process/thread state management, artifact loading, and document add/delete workflows.
tags: [module, pipeline, threads, artifacts]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: base
    resource: /src/pipeline/base.py
    title: Base pipeline utility class
  - id: steps
    resource: /src/pipeline/steps/
    title: Pipeline step utilities
  - id: processes
    resource: /src/pipeline/processes.py
    title: Process management helpers
  - id: loader
    resource: /src/pipeline/load_utils.py
    title: FactoryLoader
  - id: doc-add
    resource: /src/pipeline/document_addition.py
    title: Document addition/deletion workflow
---

# Responsibility

`src.pipeline` coordinates how domain operations run as persisted, process-scoped steps. It bridges Flask route orchestration and domain factories in [Core package](/modules/core.md).

# BaseUtil contract

`BaseUtil` is the common abstraction for pipeline steps. Subclasses provide:

* `default_config`, `serializable_config`, and config key conventions,
* predecessor loading via `_load_pre_components()`,
* the domain callable via `_process_method()`,
* execution via `_start_process()`, and
* artifact presence/deletion behavior.

`BaseUtil.start_process()` updates process status, filters unsupported config kwargs based on the target callable, runs the step, stores active objects, and aborts downstream steps on failure.[^base]

# Step utility subclasses

| Utility | Step | Domain target |
|---|---|---|
| `PreprocessingUtil` | `data` | `DataProcessingFactory.create` |
| `PhraseEmbeddingUtil` | `embedding` | `SentenceEmbeddingsFactory.create` |
| `ClusteringUtil` | `clustering` | `PhraseClusterFactory.create` |
| `GraphCreationUtil` | `graph` | `WordEmbeddingClustering._ConceptGraphClustering.build_concept_graphs` |
| `ConceptGraphIntegrationUtil` | `integration` | `ConceptGraphIntegrationFactory.create` |

# Artifact loading

`FactoryLoader` loads process artifacts from `<storage>/<process>/` and reattaches predecessor objects where needed, for example attaching `DataProcessing` to loaded embeddings and embeddings to loaded clusters.[^loader]

# Document add/delete workflow

`document_addition.py` supports adding new JSON documents to existing graph/vector-store state and deleting document provenance from graph nodes and Marqo entries. It does not currently insert added full documents into an external document server.

[^base]: Base pipeline utility class
[^loader]: FactoryLoader
