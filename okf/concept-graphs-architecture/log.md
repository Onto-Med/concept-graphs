# Bundle Update Log

## 2026-09-03
* **Update**: Added backend-neutral query-expansion relation vocabulary and response model fields for optional semantic `concepts`/`relations`; clarified that search-engine-specific query construction is outside `src.query_expansion`.
* **Cleanup**: Refreshed query-expansion OKF notes after the hybrid mini-ontology implementation: clarified no default grounding source is used, documented `{relations_json}` prompt injection, updated future-work status, and marked the old query-mode note as superseded downstream-translation context.
* **Historical update**: Briefly centralized query-expansion category options via `ALL_EXPANSION_CATEGORIES`; this was superseded by runtime domain-profile category vocabularies while keeping `categories.py` as fallback helper data.
* **Historical update**: Promoted query-expansion profile YAML into runtime domain profiles with category/default metadata; this was later refined by the ontology/profile split where ontology files own stable IDs/defaults and profiles own localized descriptions.
* **Rename**: Moved built-in query-expansion domain profiles from `conf/query-expansion/localization/` to `conf/query-expansion/profiles/`; no legacy localization-path fallback is kept.
* **API/GUI**: Added query-expansion domain profile metadata endpoints and changed the Streamlit GUI to use an API-backed domain-profile select box instead of reading local profile files directly.
* **Update**: Standardized query-expansion domain profile IDs and filenames to canonical underscore names such as `medical_de`; profile inputs with spaces or hyphens normalize to underscores, and bare language codes no longer imply medical profiles.
* **API**: Extended query-expansion profile metadata with applicable backend-neutral semantic relations (`id`, `label`, `description`, source/target category constraints) so downstream clients can intersect supported relation IDs without duplicating Concept Graphs relation definitions or exposing retrieval strategies.
* **API**: Added `default_relations` to query-expansion profile metadata and built-in medical profiles as semantic defaults for client-side relation mapping pre-population; these defaults are filtered to available profile relations and do not imply retrieval/query behavior.
* **API**: Added UI-only category/relation display labels to query-expansion profile metadata; labels may come from profile `category_labels` / `relation_labels` and fall back to title-cased IDs, while IDs remain the stable payload values.
* **Historical update**: Briefly moved built-in query-expansion semantic relation definitions into `medical_en` / `medical_de` profile YAML via `relation_definitions`; this was superseded by the ontology/profile split below.
* **Refactor**: Split query-expansion prompt profiles from semantic ontologies: built-in profiles now reference `conf/query-expansion/ontologies/medical.yml`, ontologies own stable category/relation IDs and defaults, profiles own prompt text plus localized labels/descriptions, and `relations.py` remains fallback/helper data.
* **Docs**: Added README guidance for adding query-expansion ontologies/profiles, including YAML examples and profile field meanings.

## 2026-09-02
* **Historical design note**: Added `operations/query-expansion-query-modes.md`; this note is now superseded and retained only as downstream query-translation context.

## 2026-09-01
* **Update**: Added `main.py` development-server CLI parsing for `-p/--port`, `--host`, `--storage-dir`, and `--debug`; default bind is now documented as `127.0.0.1:9010`.
* **Creation**: Added optional `src.gui` Streamlit frontend notes, documenting its thin-client API role and four tabs for pipeline, graph inspection, RAG, and query expansion.
* **Update**: Added future-work notes for query expansion: keep categories as default, evaluate category-free prompts as an experiment, test category-plus-mini-ontology profiles, and consider runtime-configurable domain category profiles for non-medical SONs.
* **Update**: Refreshed bundle after branch merge: documented `POST /query-expansion`, file-based RAG/query-expansion prompt profiles, version synchronization script, version `1.1.2`, and expanded test coverage (`66 passed` in `STATUS.md`).
* **Creation**: Created the Concept Graphs Architecture OKF bundle with system, runtime, workflow, interface, module, and operations concepts.
