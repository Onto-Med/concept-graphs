"""Compatibility exports for Marqo storage helpers.

New code should import from ``src.storage.marqo`` or its focused submodules.
"""

from src.storage.marqo import MarqoDocument, MarqoDocumentStore, MarqoEmbeddingStore

__all__ = [
    "MarqoDocument",
    "MarqoDocumentStore",
    "MarqoEmbeddingStore",
]
