---
type: Design Note
title: Query-expansion downstream query translation
description: Historical note about translating backend-neutral query-expansion semantics into search queries outside src.query_expansion.
tags: [query-expansion, ontology, search, roadmap, superseded]
status: superseded
superseded_by: /okf/concept-graphs-architecture/modules/query-expansion.md
generated: { by: pi-agent/gpt-5.5, at: 2026-09-02T00:00:00Z }
sources:
  - id: query-expansion
    resource: /okf/concept-graphs-architecture/modules/query-expansion.md
    title: Query expansion module notes
  - id: future-work
    resource: /okf/concept-graphs-architecture/operations/future-work.md
    title: Future work notes
---

# Status

This page is retained as historical design context. It no longer describes code that should live inside `src.query_expansion`.

The current boundary is:

* `src.query_expansion` returns backend-neutral semantic JSON: categorized `expansions`, optional `concepts`, and optional semantic `relations`.
* `src.query_expansion` does not define search-engine methods, boosts, proximity windows, Elasticsearch DSL, Marqo behavior, or executable query plans.
* Any conversion from semantic relations into a concrete search prompt/query belongs to a downstream service or caller.

# Current semantic output shape

A downstream service should consume output such as:

```json
{
  "term": "Bauchschmerzen",
  "language": "de",
  "expansions": {
    "symptom": [],
    "diagnosis": []
  },
  "concepts": [
    {
      "id": "abdominal_pain",
      "label": "Bauchschmerzen",
      "category": "symptom",
      "terms": ["Bauchschmerzen", "Abdominalschmerz"]
    },
    {
      "id": "appendicitis",
      "label": "Appendizitis",
      "category": "diagnosis",
      "terms": ["Appendizitis", "Blinddarmentzündung"]
    }
  ],
  "relations": [
    {
      "source_concept_id": "abdominal_pain",
      "relation": "may_indicate",
      "target_concept_id": "appendicitis",
      "confidence": 0.8
    }
  ]
}
```

# Current relation vocabulary

The default backend-neutral medical relations are defined in `src/query_expansion/relations.py`:

```text
equivalent_to
related_to
may_indicate
treated_by
investigated_by
confirmed_by
broader_than
narrower_than
```

Important current default directions include:

```text
symptom --may_indicate--> diagnosis
diagnosis --treated_by--> medication/procedure
symptom --investigated_by--> procedure
diagnosis --confirmed_by--> procedure
```

# Downstream translation examples

These examples are illustrative only and should not be implemented inside `src.query_expansion`.

A search/RAG caller might translate semantic output as follows:

* `equivalent_to`: treat concept terms as alternatives/synonyms.
* `may_indicate`: combine symptom and diagnosis concepts when looking for diagnostically relevant documents.
* `treated_by`: include treatment concepts as optional or required context depending on the use case.
* `investigated_by`: include diagnostic procedure/test concepts for symptom workups.
* `confirmed_by`: include procedure/test concepts as evidence for diagnosis confirmation.
* `broader_than` / `narrower_than`: use broader terms for recall or narrower terms for specificity.

Those choices are retrieval-policy decisions and belong to the consuming service.

# Recommended evaluation

Compare at least:

1. category-only expansion,
2. category + concept grouping,
3. category + concept grouping + semantic relations.

Track precision, recall/coverage, duplicate/noisy expansions, hallucination rate, grounding rate, concept grouping quality, relation validity, and downstream search/RAG usefulness.
