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
  - id: sources
    resource: /src/query_expansion/sources/
    title: Grounding source adapters
---

# Responsibility

`src.query_expansion` generates related terms for a query and grounds generated candidates against configured terminology sources.

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

# Categories

Built-in stable category IDs include `synonym`, `medication`, `diagnosis`, `symptom`, `procedure`, `abbreviation`, `broader_term`, `narrower_term`, and `related_term`.

# Source adapters

`source_from_config()` currently implements local YAML/JSON terminology sources. Local grounding exact-matches generated candidates against an entry's `term` and `synonyms`, and optional `category` / `categories` metadata can restrict which stable category IDs an entry grounds. HTTP source scaffolding exists under `sources/http.py`, but `service.py` raises `NotImplementedError` for unsupported source types.

# API integration

`src/api/routes/query_expansion.py` exposes `POST /query-expansion`. It validates `QueryExpansionRequest`, injects provider API keys from headers such as `Authorization: Bearer ...`, `X-LLM-API-Key`, or `X-API-Key` when not present in LLM options, runs `QueryExpansionService(generator=LangChainExpansionGenerator())`, and returns the Pydantic response as JSON.[^route]

[^service]: QueryExpansionService
[^route]: Query-expansion API route
