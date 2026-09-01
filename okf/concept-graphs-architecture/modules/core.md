---
type: Module
title: src.core package
description: Domain logic for preprocessing data, embedding phrases, clustering concepts, creating graphs, graph algorithms, integration, and metrics.
tags: [module, core, nlp, embeddings, clustering, networkx]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: data
    resource: /src/core/data/factory.py
    title: DataProcessingFactory
  - id: embedding
    resource: /src/core/embedding_functions.py
    title: SentenceEmbeddingsFactory
  - id: clustering
    resource: /src/core/clustering/phrase.py
    title: PhraseClusterFactory
  - id: graph-creation
    resource: /src/core/graph/creation.py
    title: GraphCreator
  - id: word-embedding
    resource: /src/core/clustering/word_embedding.py
    title: WordEmbeddingClustering
  - id: graph-algorithms
    resource: /src/core/graph/algorithms.py
    title: Graph algorithms
---

# Responsibility

`src.core` contains most concept-graph domain logic. It is intentionally callable outside Flask through factories and classes used by [Pipeline package](/modules/pipeline.md).

# Major subdomains

## Data processing

`DataProcessingFactory` loads or creates `DataProcessing` objects. Creation reads file-like or iterable document entries, runs a spaCy pipeline, attaches document metadata as spaCy extensions, extracts noun chunks, optionally filters negated chunks, tracks document/chunk mappings, and persists a pickle plus `.spacy` DocBin.[^data]

## Embeddings

`SentenceEmbeddingsFactory` wraps SentenceTransformer phrase encoding. It can store embeddings as local pickles or in Marqo/vector store, and supports optional UMAP downscaling. The nested `SentenceEmbeddings` object can also encode external phrases during document addition.[^embedding]

## Phrase clustering

`PhraseClusterFactory` creates and loads phrase-cluster objects over embeddings. Supported algorithms include KMeans, MiniBatchKMeans, and AffinityPropagation. Cluster count can be provided or estimated through a Yellowbrick/k-elbow based `ClusterNumberDetector`.[^clustering]

## Graph creation

`GraphCreator` builds one NetworkX graph from one phrase cluster. It combines fuzzy string similarity and embedding cosine similarity, can merge highly similar phrases, cuts weak edges, and stores node labels plus document provenance on graph nodes.[^graph-creation]

## Concept graph clustering

`WordEmbeddingClustering._ConceptGraphClustering` selects meaningful phrase clusters, builds concept graphs with `GraphCreator`, optionally simplifies/unrolls/subclusters them, and can build document-concept matrices from graph connectivity.[^word-embedding]

## Graph algorithms and metrics

`src/core/graph/algorithms.py` provides node ranking, graph unrolling, naive simplification, and sub-clustering. `src/core/metrics.py` provides cluster-purity helpers.

# Compatibility modules

Top-level modules such as `src/core/data_functions.py`, `src/core/cluster_functions.py`, and `src/core/graph_functions.py` remain as compatibility exports for old imports and pickle paths.

[^data]: DataProcessingFactory
[^embedding]: SentenceEmbeddingsFactory
[^clustering]: PhraseClusterFactory
[^graph-creation]: GraphCreator
[^word-embedding]: WordEmbeddingClustering
