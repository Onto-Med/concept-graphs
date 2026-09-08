# Bundle Update Log

## 2026-09-03
* **Update**: Added backend-neutral query-expansion relation vocabulary and response model fields for optional semantic `concepts`/`relations`; clarified that search-engine-specific query construction is outside `src.query_expansion`.
* **Cleanup**: Refreshed query-expansion OKF notes after the hybrid mini-ontology implementation: clarified no default grounding source is used, documented `{relations_json}` prompt injection, updated future-work status, and marked the old query-mode note as superseded downstream-translation context.

## 2026-09-02
* **Historical design note**: Added `operations/query-expansion-query-modes.md`; this note is now superseded and retained only as downstream query-translation context.

## 2026-09-01
* **Creation**: Added optional `src.gui` Streamlit frontend notes, documenting its thin-client API role and four tabs for pipeline, graph inspection, RAG, and query expansion.
* **Update**: Added future-work notes for query expansion: keep categories as default, evaluate category-free prompts as an experiment, test category-plus-mini-ontology profiles, and consider runtime-configurable domain category profiles for non-medical SONs.
* **Update**: Refreshed bundle after branch merge: documented `POST /query-expansion`, file-based RAG/query-expansion prompt profiles, version synchronization script, version `1.1.2`, and expanded test coverage (`66 passed` in `STATUS.md`).
* **Creation**: Created the Concept Graphs Architecture OKF bundle with system, runtime, workflow, interface, module, and operations concepts.
