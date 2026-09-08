---
okf_version: "0.2"
---

# Concept Graphs Architecture OKF Bundle

This bundle documents the architecture, working model, and modules of the Concept Graphs software in this workspace.

# Start here

* [System overview](overview.md) - What the software does and how major parts fit together.
* [Runtime architecture](runtime.md) - Flask app factory, shared context, process state, and artifact storage.
* [Concept graph pipeline](workflows/concept-graph-pipeline.md) - End-to-end document-to-graph flow.
* [API surface](interfaces/api-surface.md) - Main endpoint groups and their implementation modules.

# Modules

* [API package](modules/api.md) - Flask routes, request parsing, services, and pipeline route support.
* [Pipeline package](modules/pipeline.md) - Step utility classes, process control, persistence, and document add/delete workflows.
* [Core package](modules/core.md) - NLP preprocessing, embedding, clustering, graph creation, graph algorithms, and metrics.
* [Storage package](modules/storage.md) - Storage abstractions and Marqo implementations.
* [RAG package](modules/rag.md) - Retrieval-augmented generation orchestration, chatters, and chunk vector stores.
* [Query expansion package](modules/query-expansion.md) - LLM-generated and source-grounded terminology expansion.
* [GUI package](modules/gui.md) - Optional Streamlit client for pipeline operation, graph inspection, RAG, and query expansion.
* [NLP negation package](modules/nlp-negation.md) - Project-owned NegEx/negspacy-style negation support.
* [Pruning package](modules/pruning.md) - NetworkX graph pruning algorithms.

# Operations

* [Artifacts and storage](operations/artifacts-and-storage.md) - Process directories, serialized artifacts, active objects, and cache loading.
* [Prompt profiles](operations/prompt-profiles.md) - File-based localized prompt profiles for RAG and query expansion.
* [Version management](operations/version-management.md) - Scripted project/API/Docker version synchronization.
* [Testing and validation](operations/testing-and-validation.md) - Test layout and standard validation commands.
* [Future work](operations/future-work.md) - Recommended next steps, including query-expansion mini-ontology direction.
* [Query-expansion downstream query translation](operations/query-expansion-query-modes.md) - Superseded design note clarifying that concrete search-query translation belongs outside `src.query_expansion`.
