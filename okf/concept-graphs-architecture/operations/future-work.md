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

The current implementation is a good medical MVP: it exposes `POST /query-expansion`, uses LLM-first generation, validates structured output with Pydantic, supports localized prompt profiles, and can ground candidates against local YAML/JSON terminology files.

Recommended direction:

1. **Keep categories as the default control mechanism.**
   Stable category IDs make the API response predictable and help prompt control, validation, grouping, and grounding. A completely category-free prompt may be useful as an experiment, but should not replace the default behavior without quality evidence.

2. **Evaluate three prompt strategies.**
   Compare current category-based prompting against a general no-category prompt and a category-plus-mini-ontology prompt. Track quality, coverage, duplicates, hallucinations, and grounding rate.

3. **Use mini-ontologies as a likely next improvement.**
   A mini-ontology should add relation hints between categories/concepts, not necessarily replace categories. For example, a medical profile can define categories such as symptom, diagnosis, medication, and procedure, plus relations such as `symptom may_indicate diagnosis` or `diagnosis treated_by medication`. Query expansion should return this as backend-neutral semantic JSON only; search-engine-specific query construction belongs outside `src.query_expansion`.

4. **Move toward domain profiles if non-medical SONs matter.**
   At the moment, allowed category IDs are fixed in code. Prompt profiles can override category descriptions, but they cannot introduce arbitrary new category IDs without code changes. If the project needs non-medical SONs, promote categories into domain-specific profiles and validate request/LLM output against the selected profile at runtime.

5. **Improve grounding sources.**
   Local file grounding is implemented. External terminology/ontology adapters, especially HTTP/API-backed sources, remain the most important grounding extension.

# Other recommended work

* Continue optional Ruff expansion with bugbear/logging/simplification rules.
* Consider deeper decomposition of larger modules such as `src/core/clustering/word_embedding.py`, `src/core/data/factory.py`, `src/storage/marqo/embedding_store.py`, and `src/pipeline/document_addition.py`.
* Decide whether `/graph/document/add` should also update the external document index server, not only graph/vector-store state.
* Remove obsolete generated caches/artifacts and old commented historical code when safe.
