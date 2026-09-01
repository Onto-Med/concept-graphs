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
2. asks an `ExpansionGenerator` to generate candidate expansions,
3. filters candidates to requested categories,
4. grounds candidates against sources using configured grounding options, and
5. returns a `QueryExpansionResponse` grouped by category.[^service]

# Generator abstraction

`generator.py` defines the `ExpansionGenerator` protocol plus LangChain- and PydanticAI-backed implementations. This allows tests or deployments to inject deterministic or alternate generators.

# Source adapters

`source_from_config()` currently implements local terminology sources. HTTP source scaffolding exists under `sources/http.py`, but `service.py` raises `NotImplementedError` for unsupported source types.

# API integration note

The package is service-first and independent from Flask. The current source tree contains tests for the service, but no route module in `src/api/routes/` currently exposes query expansion as part of the documented HTTP API.

[^service]: QueryExpansionService
