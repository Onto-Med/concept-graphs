"""Placeholder HTTP source adapter.

Network-backed terminology adapters should subclass ``ExpansionSource`` and
implement source-specific authentication, lookup, and response mapping.
"""

from src.query_expansion.models import GeneratedExpansionCandidate, GroundingEvidence
from src.query_expansion.sources.base import ExpansionSource


class HTTPExpansionSource(ExpansionSource):
    def __init__(self, name: str, url: str, options: dict | None = None):
        self.name = name
        self.url = url
        self.options = options or {}

    def ground(self, candidate: GeneratedExpansionCandidate) -> list[GroundingEvidence]:
        raise NotImplementedError(
            "HTTP query-expansion sources are not implemented yet."
        )
