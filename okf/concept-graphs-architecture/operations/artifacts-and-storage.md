---
type: Operational Reference
title: Artifacts and storage
description: How process-scoped runtime objects and serialized pipeline artifacts are stored and loaded.
tags: [operations, artifacts, storage, pickle, process]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: readme
    resource: /README.md
    title: README processes and storage
  - id: loader
    resource: /src/pipeline/load_utils.py
    title: FactoryLoader
  - id: base
    resource: /src/pipeline/base.py
    title: BaseUtil persistence helpers
  - id: context
    resource: /src/api/context.py
    title: AppContext storage state
---

# Process storage layout

The default file storage root is `tmp/`. Each corpus/process gets a directory:

```text
tmp/<process>/
```

Typical serialized artifacts are named with the process and step:

```text
tmp/<process>/<process>_data.pickle
tmp/<process>/<process>_data.spacy
tmp/<process>/<process>_embedding.pickle
tmp/<process>/<process>_clustering.pickle
tmp/<process>/<process>_graph.pickle
```

Some code paths save the graph through a path whose suffix is resolved by the pickle helper; operationally it is treated as the process graph artifact.[^readme]

# Active objects

Recently built or loaded artifacts may be held in memory under:

```python
AppContext.pipeline.active_objects[process][step]
```

Route handlers prefer active objects when available and fall back to `FactoryLoader` for persisted artifacts.

# Loader behavior

`FactoryLoader` reconstructs object relationships when loading:

* data loads from pickle plus `.spacy` DocBin,
* embeddings reattach data processing objects,
* clusters reattach embeddings,
* graphs load as a list of NetworkX graphs.[^loader]

# Vector-store artifacts

When embeddings are stored in Marqo instead of a local pickle, the embedding pickle may contain a small config dictionary (`client_url`, `index_name`, `model_name`, `dtype`) rather than the embedding matrix itself. Loading then reconnects to Marqo and retrieves embeddings.

# Document-addition provenance

Added documents are represented in graph node `documents` attributes and in Marqo/vector-store metadata. Deleting a document removes provenance from both sides when vector-store configuration is available.

[^readme]: README processes and storage
[^loader]: FactoryLoader
