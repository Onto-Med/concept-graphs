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

1. builds grounding source adapters from `SourceConfig` objects when request `sources` are provided, or uses injected adapters in tests/custom callers; an empty `sources` list means no grounding adapters are used,
2. asks an `ExpansionGenerator` to generate structured expansion output from a localized/custom prompt profile,
3. filters generated candidates to requested categories,
4. grounds candidates against sources using configured grounding options,
5. groups grounded/LLM-only candidates by category for the compatibility `expansions` response,
6. validates generated concepts against requested categories and known terms,
7. validates generated semantic relations against requested relation IDs, existing concept IDs, and allowed source/target category connections, and
8. returns a backend-neutral `QueryExpansionResponse` with categorized `expansions` plus optional semantic `concepts` and `relations`.[^service]

# Generator abstraction

`generator.py` defines the `ExpansionGenerator` protocol plus LangChain- and PydanticAI-backed implementations. The API route now uses `LangChainExpansionGenerator` by default; the PydanticAI generator remains available for future/custom use. Tests or deployments can inject deterministic or alternate generators.

# Prompt profiles

Domain prompt profiles live under `conf/query-expansion/profiles/` and are loaded by `prompts.py`; built-ins use names such as `medical-en` and `medical-de`. The profile is selected from `request.prompt.profile` or the request language; language shorthands such as `de` resolve to `medical-de` when present, with `medical-en` fallback. Profiles define localized prompt text plus the runtime category vocabulary via `category_descriptions` and optional `default_categories`. Requests can override the template or per-category descriptions for selected categories. The API exposes profile metadata through `GET /query-expansion/profiles` and `GET /query-expansion/profiles/{profile_name}` so remote clients do not need filesystem access. See [Prompt profiles](/operations/prompt-profiles.md).

# Categories and relations

Runtime category IDs come from the selected domain profile's `category_descriptions`. If a request omits `categories`, the selected profile's `default_categories` are used; if a profile does not define category metadata, `categories.py` provides built-in medical fallback helpers. The fallback medical IDs include `synonym`, `medication`, `diagnosis`, `symptom`, `procedure`, `abbreviation`, `broader_term`, `narrower_term`, and `related_term`. Request categories are validated against the selected profile/fallback vocabulary during prompt/service preparation rather than against a fixed code enum.

`relations.py` defines backend-neutral medical relation IDs such as `equivalent_to`, `related_to`, `may_indicate`, `treated_by`, `investigated_by`, `confirmed_by`, `broader_than`, and `narrower_than`. These relations describe semantic structure only and intentionally do not reference search-engine methods, backend DSLs, boosts, or proximity settings. The default `investigated_by` relation is symptom-to-procedure; diagnosis-to-procedure evidence uses `confirmed_by`.

Requests may restrict the semantic mini-ontology with `relations` and `relation_definitions`. Relation definitions may reference categories from the selected domain profile; prompt construction trims relation definitions to the effective selected categories so the LLM is not asked to use impossible category connections. The LLM may propose candidates, concepts, and relations, but `QueryExpansionService` validates generated concepts/relations against effective request categories, requested relation IDs, known concept IDs, and allowed source/target category connections before returning them.

# Source adapters

`source_from_config()` currently implements local YAML/JSON terminology sources when the request explicitly supplies `sources`. There is no automatic default grounding source. If no sources are supplied, generated candidates can still be returned as `llm_only` depending on grounding options. Local grounding exact-matches generated candidates against an entry's `term` and `synonyms`, and optional `category` / `categories` metadata can restrict which selected domain-profile category IDs an entry grounds. HTTP source scaffolding exists under `sources/http.py`, but `service.py` raises `NotImplementedError` for unsupported source types.

# API integration

`src/api/routes/query_expansion.py` exposes `GET /query-expansion/profiles`, `GET /query-expansion/profiles/{profile_name}`, and `POST /query-expansion`. The metadata endpoints return API-side domain profile names, category descriptions, and default categories for remote clients. The POST endpoint validates `QueryExpansionRequest`, injects provider API keys from headers such as `Authorization: Bearer ...`, `X-LLM-API-Key`, or `X-API-Key` when not present in LLM options, runs `QueryExpansionService(generator=LangChainExpansionGenerator())`, and returns the Pydantic response as JSON. The response preserves categorized `expansions` and can also carry optional semantic `concepts` and `relations` for downstream query construction.[^route]

[^service]: QueryExpansionService
[^route]: Query-expansion API route
