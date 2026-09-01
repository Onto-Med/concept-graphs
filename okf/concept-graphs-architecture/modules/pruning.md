---
type: Module
title: src.pruning package
description: NetworkX graph pruning support used by concept-graph simplification.
tags: [module, graph, pruning, networkx]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: pruning-readme
    resource: /src/pruning/README.md
    title: Pruning README
  - id: unimodal
    resource: /src/pruning/unimodal.py
    title: Marginal likelihood filter implementation
  - id: graph-algorithms
    resource: /src/core/graph/algorithms.py
    title: Graph simplification algorithms
---

# Responsibility

`src.pruning` provides graph-pruning algorithms used when concept graphs are simplified by edge significance rather than raw weight.

# Main implementation

`src/pruning/unimodal.py` defines `MLF`, a marginal-likelihood-filter style pruning class. `WordEmbeddingClustering._ConceptGraphClustering._graph_list()` uses `unimodal.MLF(directed=False)` when `graph_simplify_alg == "significance"` and graph simplification is enabled.

# Utility

`src/pruning/utils.py` exposes a generic `prune()` helper that removes graph edges according to a field and either a percentage or count.

# Operational note

The current project status states pruning support is explicitly NetworkX-only and tests cover NetworkX pruning behavior.
