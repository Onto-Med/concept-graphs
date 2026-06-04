"""Source interfaces for grounding query-expansion candidates."""

from abc import ABC, abstractmethod

from src.query_expansion.models import GeneratedExpansionCandidate, GroundingEvidence


class ExpansionSource(ABC):
    """Base class for terminology/ontology/source adapters."""

    name: str

    @abstractmethod
    def ground(self, candidate: GeneratedExpansionCandidate) -> list[GroundingEvidence]:
        """Return grounding evidence for a generated candidate."""
        raise NotImplementedError
