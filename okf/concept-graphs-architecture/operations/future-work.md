---
type: Operational Reference
title: Future work
description: Recommended next steps and design direction for the Concept Graphs project.
tags: [operations, roadmap, future-work, query-expansion, ontology]
status: draft
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: status
    resource: /STATUS.md
    title: Project cleanup status
  - id: query-expansion
    resource: /src/query_expansion/
    title: Query expansion package
  - id: query-expansion-okf
    resource: /okf/concept-graphs-architecture/modules/query-expansion.md
    title: Query expansion module notes
---

# Future work

This page tracks recommended next steps that are larger than immediate cleanup tasks.

# Query expansion direction

The current implementation is a medical MVP with an initial hybrid mini-ontology path: it exposes `POST /query-expansion`, uses LLM-first generation, validates structured output with Pydantic, supports localized prompt profiles, accepts category/relation constraints, can return optional semantic `concepts` and `relations`, and can ground candidates against explicitly configured local YAML/JSON terminology files.

Recommended direction:

1. **Keep categories as the default control mechanism.**
   Stable category IDs make the API response predictable and help prompt control, validation, grouping, and grounding. A completely category-free prompt may be useful as an experiment, but should not replace the default behavior without quality evidence.

2. **Evaluate prompt and structure strategies.**
   Compare category-only prompting, category-plus-mini-ontology prompting, and any general no-category experiment. Track quality, coverage, duplicates, hallucinations, grounding rate, concept grouping quality, and relation validity/usefulness.

3. **Refine mini-ontology behavior.**
   The initial mini-ontology path lets requests select relation IDs and relation definitions connecting categories. Continue refining the default medical relation set and validation behavior. Query expansion should keep returning backend-neutral semantic JSON only; search-engine-specific query construction belongs outside `src.query_expansion`.

4. **Expand domain-profile support if non-medical SONs matter.**
   Category IDs are now runtime-defined by the selected query-expansion domain profile, with `categories.py` kept as a built-in medical fallback. Future work should consider moving relation definitions and domain-specific grounding/source hints into the same profile model.

5. **Improve grounding sources.**
   Local file grounding is implemented. External terminology/ontology adapters, especially HTTP/API-backed sources, remain the most important grounding extension.

# Other recommended work

* Continue optional Ruff expansion with bugbear/logging/simplification rules.
* Consider deeper decomposition of larger modules such as `src/core/clustering/word_embedding.py`, `src/core/data/factory.py`, `src/storage/marqo/embedding_store.py`, and `src/pipeline/document_addition.py`.
* Decide whether `/graph/document/add` should also update the external document index server, not only graph/vector-store state.
* Remove obsolete generated caches/artifacts and old commented historical code when safe.
