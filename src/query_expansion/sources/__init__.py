"""Query-expansion source adapters."""

from src.query_expansion.sources.base import ExpansionSource
from src.query_expansion.sources.http import HTTPExpansionSource
from src.query_expansion.sources.local import LocalTerminologySource

__all__ = [
    "ExpansionSource",
    "HTTPExpansionSource",
    "LocalTerminologySource",
]
