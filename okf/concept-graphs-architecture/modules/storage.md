---
type: Module
title: src.storage package
description: Storage abstraction interfaces and Marqo-backed document/embedding implementations.
tags: [module, storage, marqo, vectorstore]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: interfaces
    resource: /src/storage/interfaces.py
    title: Storage interfaces
  - id: marqo
    resource: /src/storage/marqo/
    title: Marqo storage implementation
  - id: integration
    resource: /src/core/integration_functions.py
    title: ConceptGraphIntegrationFactory
---

# Responsibility

`src.storage` isolates vector/document-store behavior behind abstract interfaces so pipeline, document addition, and RAG code are not hard-wired directly to one storage implementation.

# Interfaces

`src/storage/interfaces.py` defines abstract base classes for:

* `Document`,
* `EmbeddingStore`, and
* `DocumentStore`.

These describe the operations needed for storing embeddings, updating metadata, searching/retrieving entries, and document-level provenance behavior.

# Marqo implementation

`src/storage/marqo/` contains concrete classes:

* `MarqoDocument`,
* `MarqoEmbeddingStore`, and
* `MarqoDocumentStore`.

These classes back phrase embeddings, graph-cluster metadata, and document-addition provenance. Compatibility exports remain in `src/storage/marqo_external_utils.py` and package `__init__` files.

# Integration with concept graphs

`ConceptGraphIntegrationFactory.create()` walks generated NetworkX graphs, collects `graph_cluster` membership per phrase/node id, and updates matching entries in an `EmbeddingStore`.[^integration]

# Related concepts

* [Pipeline package](/modules/pipeline.md) uses storage during embedding, integration, and document add/delete.
* [RAG package](/modules/rag.md) uses a separate chunk-level embedding-store abstraction.

[^integration]: ConceptGraphIntegrationFactory
