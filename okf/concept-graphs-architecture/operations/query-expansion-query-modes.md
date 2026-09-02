---
type: Design Note
title: Query-expansion query modes
description: Proposed mapping from query-expansion categories/relations to Elasticsearch-style classical queries.
tags: [query-expansion, ontology, elasticsearch, search, roadmap]
status: draft
generated: { by: pi-agent/gpt-5.5, at: 2026-09-02T00:00:00Z }
sources:
  - id: query-expansion
    resource: /okf/concept-graphs-architecture/modules/query-expansion.md
    title: Query expansion module notes
  - id: future-work
    resource: /okf/concept-graphs-architecture/operations/future-work.md
    title: Future work notes
---

# Purpose

For ontology-style query expansion, categories produce term groups and relations decide how those groups should be combined into a downstream Elasticsearch-like query.

This is proposed future work; the current implementation returns categorized expansion candidates but does not yet emit relation-aware query plans.

# Core idea

A good expanded query should not collapse every term into one large `OR` list. Instead:

* terms within the same concept/synonym group are combined with `OR`,
* related concept groups are combined using relation-specific query modes,
* weaker or indirect expansions are usually `should`/boost clauses,
* high-confidence relational expansions can become `must`, `near`, or boosted proximity clauses.

# Initial query modes

## `same_or_group`

Use for synonyms, spelling variants, abbreviations, and lay terms.

Semantics:

```text
(term_1 OR term_2 OR term_3 ...)
```

Elasticsearch-style implementation: `bool.should` with `minimum_should_match: 1`, or a configured synonym analyzer if available.

## `should_boost`

Use for weak `related_to` / associative expansions.

Semantics:

```text
original_query MUST match; related terms SHOULD match and increase score
```

This improves recall without making loosely related terms mandatory.

## `must_both_boost_near`

Use for strong relation pairs such as:

```text
symptom may_indicate diagnosis
```

Semantics:

```text
source concept group MUST match
AND target concept group MUST match
BOOST if source and target occur near each other
```

Elasticsearch-style implementation: `bool.must` for both groups plus `match_phrase`/span/proximity query in `should` with a boost.

## `must_source_should_target`

Use for directional but optional target relations, for example:

```text
diagnosis treated_by medication
diagnosis investigated_by procedure
```

Semantics:

```text
source concept group MUST match
target concept group SHOULD match and increase score
```

This avoids losing relevant documents that mention the diagnosis without explicitly mentioning treatment/procedure terms.

## `should_low_boost`

Use for broader terms and high-recall expansions.

Semantics:

```text
broader terms SHOULD match with lower boost
```

Broader terms improve recall but can reduce precision, so they should usually not be required.

## `should_high_boost`

Use for narrower terms and highly specific descendants.

Semantics:

```text
narrower terms SHOULD match with higher boost
```

Narrower terms are often useful precision signals, especially when the input concept is broad.

# Suggested relation mapping

```yaml
relations:
  equivalent_to:
    applies_to: [synonym, abbreviation]
    query_mode: same_or_group

  related_to:
    applies_to: [related_term]
    query_mode: should_boost

  symptom_may_indicate_diagnosis:
    from: symptom
    to: diagnosis
    label: may_indicate
    query_mode: must_both_boost_near

  diagnosis_treated_by_medication:
    from: diagnosis
    to: medication
    label: treated_by
    query_mode: must_source_should_target

  diagnosis_investigated_by_procedure:
    from: diagnosis
    to: procedure
    label: investigated_by
    query_mode: must_source_should_target

  broader_term:
    query_mode: should_low_boost

  narrower_term:
    query_mode: should_high_boost
```

# Example

Input term:

```text
Bauchschmerzen
```

Expanded concept groups:

```yaml
concepts:
  abdominal_pain:
    category: symptom
    terms: [Bauchschmerzen, Bauchweh, Abdominalschmerz]
  appendicitis:
    category: diagnosis
    terms: [Appendizitis, Blinddarmentzündung]
relations:
  - subject: abdominal_pain
    predicate: may_indicate
    object: appendicitis
    query_mode: must_both_boost_near
```

Classical query shape:

```text
(Bauchschmerzen OR Bauchweh OR Abdominalschmerz)
AND
(Appendizitis OR Blinddarmentzündung)
BOOST if both groups occur near each other
```

# Recommended first experiment

Compare three strategies on the same query set:

1. flat synonym/category expansion,
2. category-only grouped expansion,
3. category + relation-aware query modes.

Track at least precision, recall/coverage, duplicate/noisy expansions, hallucination rate, and grounding rate.

# Implementation note

The current `src.query_expansion` schema would need an extension to emit relation/query-plan metadata, for example fields such as `concept_id`, `relation`, `source_concept_id`, `target_concept_id`, and `query_mode`. The existing categorized response can remain as the compatibility layer.
