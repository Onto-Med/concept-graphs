"""Query-expansion orchestration service."""

from src.query_expansion.generator import (
    ExpansionGenerator,
    LangChainExpansionGenerator,
)
from src.query_expansion.grounding import ground_candidate, group_grounded_candidates
from src.query_expansion.models import (
    QueryExpansionRequest,
    QueryExpansionResponse,
    SourceConfig,
)
from src.query_expansion.sources.base import ExpansionSource
from src.query_expansion.sources.local import LocalTerminologySource


def source_from_config(config: SourceConfig) -> ExpansionSource:
    """Create a grounding source adapter from request configuration."""
    if config.type == "local":
        if config.path is None:
            raise ValueError(
                f"Local query-expansion source '{config.name}' needs a path."
            )
        return LocalTerminologySource(config.name, config.path)
    raise NotImplementedError(
        f"Query-expansion source type '{config.type}' is not implemented yet."
    )


class QueryExpansionService:
    """Coordinate LLM generation and optional source grounding.

    The service is intentionally independent from Flask so it can be used from an
    API route, a CLI, tests, or future batch jobs. By default it uses the
    LangChain-backed generator, but tests or deployments can inject any object
    implementing the ``ExpansionGenerator`` protocol.
    """

    def __init__(self, generator: ExpansionGenerator | None = None):
        """Create a service with either a custom or default LLM generator."""
        self.generator = generator or LangChainExpansionGenerator()

    def expand(
        self,
        request: QueryExpansionRequest,
        sources: list[ExpansionSource] | None = None,
    ) -> QueryExpansionResponse:
        """Generate candidates, ground them, and group them by category.

        Args:
            request: User request containing the term, categories, LLM config,
                source config, and grounding options.
            sources: Optional pre-built source adapters. If omitted, adapters are
                created from ``request.sources``.

        Returns:
            A response containing grounded and/or LLM-only candidates grouped by
            category.
        """
        sources = (
            [source_from_config(source_config) for source_config in request.sources]
            if sources is None
            else sources
        )
        generated = self.generator.generate(request)
        grounded = [
            grounded_candidate
            for candidate in generated.candidates
            if candidate.category in request.categories
            if (
                grounded_candidate := ground_candidate(
                    candidate, sources, request.grounding
                )
            )
            is not None
        ]
        grouped = group_grounded_candidates(grounded)
        return QueryExpansionResponse(
            term=request.term,
            language=request.language,
            expansions={
                category: grouped.get(category, []) for category in request.categories
            },
        )
