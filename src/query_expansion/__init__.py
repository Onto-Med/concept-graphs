"""LLM-driven query expansion with optional source grounding."""

from src.query_expansion.generator import (
    LangChainExpansionGenerator,
    PydanticAIExpansionGenerator,
)
from src.query_expansion.models import (
    ExpansionGeneration,
    GeneratedExpansionCandidate,
    GroundedExpansionCandidate,
    GroundingEvidence,
    GroundingOptions,
    GroundingStatus,
    LLMConfig,
    QueryExpansionRequest,
    QueryExpansionResponse,
    SourceConfig,
)
from src.query_expansion.service import QueryExpansionService

__all__ = [
    "ExpansionGeneration",
    "GeneratedExpansionCandidate",
    "GroundedExpansionCandidate",
    "GroundingEvidence",
    "GroundingOptions",
    "GroundingStatus",
    "LangChainExpansionGenerator",
    "LLMConfig",
    "PydanticAIExpansionGenerator",
    "QueryExpansionRequest",
    "QueryExpansionResponse",
    "QueryExpansionService",
    "SourceConfig",
]
