"""Compatibility exports for clustering helpers.

New code should import from ``src.core.clustering`` or its focused submodules.
"""

from src.core.clustering import (
    ClusterNumberDetector,
    PhraseClusterFactory,
    WordEmbeddingClustering,
)

__all__ = [
    "ClusterNumberDetector",
    "PhraseClusterFactory",
    "WordEmbeddingClustering",
]
