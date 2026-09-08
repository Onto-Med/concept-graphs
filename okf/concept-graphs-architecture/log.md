# Bundle Update Log

## 2026-09-03
* **Update**: Added backend-neutral query-expansion relation vocabulary and response model fields for optional semantic `concepts`/`relations`; clarified that search-engine-specific query construction is outside `src.query_expansion`.
* **Cleanup**: Refreshed query-expansion OKF notes after the hybrid mini-ontology implementation: clarified no default grounding source is used, documented `{relations_json}` prompt injection, updated future-work status, and marked the old query-mode note as superseded downstream-translation context.
* **Historical update**: Briefly centralized query-expansion category options via `ALL_EXPANSION_CATEGORIES`; this was superseded by runtime domain-profile category vocabularies while keeping `categories.py` as fallback helper data.
* **Update**: Promoted query-expansion profile YAML into runtime domain profiles: `category_descriptions` define allowed category IDs, `default_categories` drive omitted request categories, and `categories.py` is now fallback/default helper data rather than the primary source of truth.
* **Rename**: Moved built-in query-expansion domain profiles from `conf/query-expansion/localization/` to `conf/query-expansion/profiles/`; no legacy localization-path fallback is kept.
* **API/GUI**: Added query-expansion domain profile metadata endpoints and changed the Streamlit GUI to use an API-backed domain-profile select box instead of reading local profile files directly.

## 2026-09-02
* **Historical design note**: Added `operations/query-expansion-query-modes.md`; this note is now superseded and retained only as downstream query-translation context.

## 2026-09-01
* **Creation**: Added optional `src.gui` Streamlit frontend notes, documenting its thin-client API role and four tabs for pipeline, graph inspection, RAG, and query expansion.
* **Update**: Added future-work notes for query expansion: keep categories as default, evaluate category-free prompts as an experiment, test category-plus-mini-ontology profiles, and consider runtime-configurable domain category profiles for non-medical SONs.
* **Update**: Refreshed bundle after branch merge: documented `POST /query-expansion`, file-based RAG/query-expansion prompt profiles, version synchronization script, version `1.1.2`, and expanded test coverage (`66 passed` in `STATUS.md`).
* **Creation**: Created the Concept Graphs Architecture OKF bundle with system, runtime, workflow, interface, module, and operations concepts.
