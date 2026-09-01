---
type: Module
title: src.rag package
description: Retrieval-augmented generation over process-scoped document chunks.
tags: [module, rag, llm, marqo, langchain]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: rag
    resource: /src/rag/rag.py
    title: RAG orchestration class
  - id: routes
    resource: /src/api/routes/rag.py
    title: RAG routes
  - id: rag-vectorstore
    resource: /src/api/services/rag_vectorstore.py
    title: RAG vector-store service
  - id: chatters
    resource: /src/rag/chatters/
    title: Chatter implementations
  - id: stores
    resource: /src/rag/embedding_stores/
    title: Chunk embedding stores
  - id: prompts
    resource: /src/rag/prompts.py
    title: RAG prompt profile loader
---

# Responsibility

`src.rag` provides question answering over retrieved document chunks. It is optional and process-scoped: each corpus/process can have one active RAG component stored in `RagContext.active_by_process`.

# Main class

`RAG` composes three pieces:

1. a `Chatter` implementation or import path,
2. a prompt template resolved from file-based profiles or request overrides, and
3. retrieved source documents converted into LangChain `Document` objects.

`build_and_invoke()` runs the prompt/chatter chain and returns `(success, answer_or_error)`.[^rag]

# API flow

`POST /rag/init` parses RAG config, creates or reuses a Marqo-backed chunk vector store named `<process>_rag`, initializes the configured chatter, resolves a prompt profile, and starts a background fill task when the chunk store is empty or force-reinitialized.[^routes]

`GET/POST /rag/question` requires a ready active RAG component. It retrieves chunks for the question, optionally filters by document ids, extracts highlighted source text, invokes the RAG chain, and returns an answer plus source metadata JSON.[^routes]

# Prompt profiles

`src/rag/prompts.py` resolves prompt profiles from `conf/rag/localization/<profile>.yml`, falling back to built-in English/German prompts. The request `prompt_template` can select a `profile`, provide a direct `template`, or use the older inline `templates`/`input_variables` shape. See [Prompt profiles](/operations/prompt-profiles.md).

# Extension points

* `src/rag/chatters/base.py` defines the chatter abstraction.
* `src/rag/chatters/blablador.py` and `ollama.py` provide concrete chatters; `BlabladorChatter` forwards provider-specific `extra_body` options to `ChatOpenAI`.
* `src/rag/embedding_stores/base.py` and `marqo.py` define chunk vector-store behavior.
* `src/rag/text_splitters.py` converts preprocessed spaCy documents into chunks.

[^rag]: RAG orchestration class
[^routes]: RAG routes
