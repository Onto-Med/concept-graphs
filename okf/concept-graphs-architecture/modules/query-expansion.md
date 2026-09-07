---
type: Module
title: src.query_expansion package
description: LLM-generated query expansion with optional grounding against terminology sources.
tags: [module, query-expansion, llm, grounding]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: service
    resource: /src/query_expansion/service.py
    title: QueryExpansionService
  - id: models
    resource: /src/query_expansion/models.py
    title: Query expansion models
  - id: generator
    resource: /src/query_expansion/generator.py
    title: Expansion generators
  - id: prompts
    resource: /src/query_expansion/prompts.py
    title: Query-expansion prompt profile loader
  - id: route
    resource: /src/api/routes/query_expansion.py
    title: Query-expansion API route
  - id: grounding
    resource: /src/query_expansion/grounding.py
    title: Grounding helpers
  - id: relations
    resource: /src/query_expansion/relations.py
    title: Backend-neutral semantic relation vocabulary
  - id: sources
    resource: /src/query_expansion/sources/
    title: Grounding source adapters
---

# Responsibility

`src.query_expansion` generates related terms for a query and grounds generated candidates against configured terminology sources. It returns backend-neutral JSON only; search engines or RAG callers translate that semantic output into their own query behavior outside this package.

# Flow

`QueryExpansionService.expand()`:

1. builds source adapters from `SourceConfig` objects unless adapters were injected,
2. asks an `ExpansionGenerator` to generate candidate expansions from a localized/custom prompt profile,
3. filters candidates to requested categories,
4. grounds candidates against sources using configured grounding options, and
5. returns a `QueryExpansionResponse` grouped by category.[^service]

# Generator abstraction

`generator.py` defines the `ExpansionGenerator` protocol plus LangChain- and PydanticAI-backed implementations. The API route now uses `LangChainExpansionGenerator` by default; the PydanticAI generator remains available for future/custom use. Tests or deployments can inject deterministic or alternate generators.

# Prompt profiles

Prompt profiles live under `conf/query-expansion/localization/` and are loaded by `prompts.py`. The profile is selected from `request.prompt.profile` or the request language, with English fallback. Requests can override the template or per-category descriptions while keeping stable category IDs. See [Prompt profiles](/operations/prompt-profiles.md).

# Categories and relations

Built-in stable category IDs include `synonym`, `medication`, `diagnosis`, `symptom`, `procedure`, `abbreviation`, `broader_term`, `narrower_term`, and `related_term`.

`relations.py` defines backend-neutral medical relation IDs such as `equivalent_to`, `related_to`, `may_indicate`, `treated_by`, `investigated_by`, `broader_than`, and `narrower_than`. These relations describe semantic structure only and intentionally do not reference search-engine methods, backend DSLs, boosts, or proximity settings.

Requests may restrict the semantic mini-ontology with `relations` and `relation_definitions`. The LLM may propose candidates, concepts, and relations, but `QueryExpansionService` validates generated concepts/relations against requested categories, requested relation IDs, known concept IDs, and allowed source/target category connections before returning them.

# Source adapters

`source_from_config()` currently implements local YAML/JSON terminology sources. Local grounding exact-matches generated candidates against an entry's `term` and `synonyms`, and optional `category` / `categories` metadata can restrict which stable category IDs an entry grounds. HTTP source scaffolding exists under `sources/http.py`, but `service.py` raises `NotImplementedError` for unsupported source types.

# API integration

`src/api/routes/query_expansion.py` exposes `POST /query-expansion`. It validates `QueryExpansionRequest`, injects provider API keys from headers such as `Authorization: Bearer ...`, `X-LLM-API-Key`, or `X-API-Key` when not present in LLM options, runs `QueryExpansionService(generator=LangChainExpansionGenerator())`, and returns the Pydantic response as JSON. The response preserves categorized `expansions` and can also carry optional semantic `concepts` and `relations` for downstream query construction.[^route]

[^service]: QueryExpansionService
[^route]: Query-expansion API route
