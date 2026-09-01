---
type: Workflow
title: Concept graph pipeline
description: End-to-end flow that transforms input documents into graph and optional vector-store/RAG artifacts.
tags: [pipeline, workflow, preprocessing, embedding, clustering, graph]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: readme
    resource: /README.md
    title: Project README
  - id: api-pipeline
    resource: /src/api/pipeline.py
    title: Pipeline route implementation
  - id: step-support
    resource: /src/api/pipeline_support/steps.py
    title: Pipeline step preparation
  - id: utils
    resource: /src/pipeline/steps/
    title: Pipeline step utility modules
---

# Pipeline sequence

The main document-to-graph pipeline runs these ordered steps:[^readme]

1. **Data / preprocessing**: load documents from an uploaded zip or document server; run spaCy; extract noun/phrase chunks; optionally omit negated chunks.
2. **Embedding**: encode phrase chunks with a SentenceTransformer; optionally downscale; store embeddings as pickle or in Marqo.
3. **Clustering**: cluster phrase embeddings into candidate concepts with KMeans, MiniBatchKMeans, or AffinityPropagation; optionally estimate cluster count.
4. **Graph creation**: build NetworkX concept graphs from phrase clusters using string similarity, embedding cosine similarity, merge thresholds, and edge pruning/simplification.
5. **Integration**: optionally write graph-cluster metadata back to the external embedding store.
6. **RAG**: separately initialize chunk-level retrieval and question answering over processed documents.

# API orchestration

`POST /pipeline` is handled by `src/api/pipeline.py`. It:

* parses JSON or multipart requests,
* resolves query parameters such as `process`, `language`, `skip_present`, and `skip_steps`,
* normalizes vector-store configuration,
* loads documents from a document server when no upload is present,
* prepares step utility objects, and
* starts the pipeline in a background thread.[^api-pipeline]

# Step utility pattern

Each pipeline step is represented by a `BaseUtil` subclass under `src/pipeline/steps/`. The utility reads config, loads required predecessor artifacts, runs the domain factory/method, writes serialized outputs, and updates process status.

# Skip/reuse behavior

Pipeline query parameters support skipping named steps and reusing serialized step artifacts. This is implemented in `src/api/pipeline_support/steps.py`, which can load skipped predecessor steps from storage when downstream steps still need their objects.[^step-support]

# Outputs

Pipeline artifacts are process-scoped. See [Artifacts and storage](/operations/artifacts-and-storage.md).

[^readme]: Project README
[^api-pipeline]: Pipeline route implementation
[^step-support]: Pipeline step preparation
