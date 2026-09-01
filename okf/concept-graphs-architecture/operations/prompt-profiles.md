---
type: Operational Reference
title: Prompt profiles
description: File-based localized prompt profiles for RAG and query expansion.
tags: [operations, prompts, rag, query-expansion, localization]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: readme
    resource: /README.md
    title: README prompt profile sections
  - id: rag-prompts
    resource: /src/rag/prompts.py
    title: RAG prompt profile loader
  - id: qe-prompts
    resource: /src/query_expansion/prompts.py
    title: Query-expansion prompt profile loader
  - id: conf
    resource: /conf/
    title: Configuration tree
---

# RAG prompt profiles

RAG prompt profiles live in:

```text
conf/rag/localization/<profile>.yml
```

`src/rag/prompts.py` resolves profiles by normalized language/profile name, falling back to built-in English/German templates. Profiles contain a `template` and `input_variables`; request config can still use the older inline `templates`/`input_variables` shape or a direct `template` override.[^rag-prompts]

# Query-expansion prompt profiles

Query-expansion prompt profiles live in:

```text
conf/query-expansion/localization/<profile>.yml
```

`src/query_expansion/prompts.py` loads the requested profile or the English fallback and formats the generation prompt with `{term}`, `{language}`, `{language_name}`, `{limit_per_category}`, `{categories_json}`, and `{schema_instruction}`. Requests may override the full template or category descriptions while preserving stable category IDs.[^qe-prompts]

# Docker/runtime note

Production Docker usage should mount individual prompt profile files or profile directories into the matching `conf/` subdirectory rather than bind-mounting the whole project over `/rest_api`.[^readme]

[^readme]: README prompt profile sections
[^rag-prompts]: RAG prompt profile loader
[^qe-prompts]: Query-expansion prompt profile loader
