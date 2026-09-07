"""Query-expansion orchestration service."""

from src.query_expansion.generator import (
    ExpansionGenerator,
    LangChainExpansionGenerator,
)
from src.query_expansion.grounding import ground_candidate, group_grounded_candidates
from src.query_expansion.models import (
    ExpansionConcept,
    ExpansionSemanticRelation,
    QueryExpansionRequest,
    QueryExpansionResponse,
    SourceConfig,
)
from src.query_expansion.relations import RelationDefinition
from src.query_expansion.sources.base import ExpansionSource
from src.query_expansion.sources.local import LocalTerminologySource


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


def _allowed_terms(
    request: QueryExpansionRequest,
    generated_terms: list[str],
) -> set[str]:
    return {_normalize(request.term), *(_normalize(term) for term in generated_terms)}


def _valid_concepts(
    request: QueryExpansionRequest,
    concepts: list[ExpansionConcept],
    generated_terms: list[str],
) -> list[ExpansionConcept]:
    allowed_terms = _allowed_terms(request, generated_terms)
    valid = []
    seen_ids = set()
    for concept in concepts:
        if concept.id in seen_ids or concept.category not in request.categories:
            continue
        concept_terms = [
            term for term in concept.terms if _normalize(term) in allowed_terms
        ]
        if not concept_terms:
            continue
        valid.append(concept.model_copy(update={"terms": concept_terms}))
        seen_ids.add(concept.id)
    return valid


def _relation_definitions_by_id(
    request: QueryExpansionRequest,
) -> dict[str, RelationDefinition]:
    return {
        definition.id: definition
        for definition in request.relation_definitions
        if definition.id in request.relations
    }


def _valid_relations(
    request: QueryExpansionRequest,
    concepts: list[ExpansionConcept],
    relations: list[ExpansionSemanticRelation],
) -> list[ExpansionSemanticRelation]:
    concepts_by_id = {concept.id: concept for concept in concepts}
    definitions_by_id = _relation_definitions_by_id(request)
    valid = []
    for relation in relations:
        definition = definitions_by_id.get(relation.relation)
        source = concepts_by_id.get(relation.source_concept_id)
        target = concepts_by_id.get(relation.target_concept_id)
        if definition is None or source is None or target is None:
            continue
        if source.category not in definition.source_categories:
            continue
        if target.category not in definition.target_categories:
            continue
        valid.append(relation)
    return valid


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
        concepts = _valid_concepts(
            request=request,
            concepts=generated.concepts,
            generated_terms=[
                candidate.term
                for candidate in generated.candidates
                if candidate.category in request.categories
            ],
        )
        relations = _valid_relations(
            request=request,
            concepts=concepts,
            relations=generated.relations,
        )
        return QueryExpansionResponse(
            term=request.term,
            language=request.language,
            expansions={
                category: grouped.get(category, []) for category in request.categories
            },
            concepts=concepts,
            relations=relations,
        )
