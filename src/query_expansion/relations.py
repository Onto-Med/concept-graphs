"""Backend-neutral semantic relation definitions for query expansion.

Relations describe meaning between expanded concepts. They intentionally do not
encode search-engine behavior, query DSL, boosts, proximity windows, or backend
methods. Downstream consumers can translate these semantic relations into their
own retrieval/query strategy.
"""

from typing import Literal

from pydantic import BaseModel, Field

from src.query_expansion.categories import ExpansionCategory

QueryExpansionRelation = Literal[
    "equivalent_to",
    "related_to",
    "may_indicate",
    "treated_by",
    "investigated_by",
    "broader_than",
    "narrower_than",
]


class RelationDefinition(BaseModel):
    """Domain relation allowed between expansion concept categories."""

    id: QueryExpansionRelation = Field(description="Stable relation identifier.")
    source_categories: tuple[ExpansionCategory, ...] = Field(
        description="Categories that can appear on the source side of the relation."
    )
    target_categories: tuple[ExpansionCategory, ...] = Field(
        description="Categories that can appear on the target side of the relation."
    )
    description: str = Field(
        description="Human-readable semantic meaning of the relation."
    )


MEDICAL_RELATION_DEFINITIONS: tuple[RelationDefinition, ...] = (
    RelationDefinition(
        id="equivalent_to",
        source_categories=("synonym", "abbreviation"),
        target_categories=("synonym", "abbreviation"),
        description=(
            "Concepts or terms are equivalent labels, variants, or abbreviations."
        ),
    ),
    RelationDefinition(
        id="related_to",
        source_categories=("related_term",),
        target_categories=("related_term",),
        description=(
            "Concepts are clinically or semantically associated without a more "
            "specific relation."
        ),
    ),
    RelationDefinition(
        id="may_indicate",
        source_categories=("symptom",),
        target_categories=("diagnosis",),
        description="A symptom, sign, or finding may indicate a diagnosis.",
    ),
    RelationDefinition(
        id="treated_by",
        source_categories=("diagnosis",),
        target_categories=("medication", "procedure"),
        description=(
            "A diagnosis or condition may be treated by a medication or procedure."
        ),
    ),
    RelationDefinition(
        id="investigated_by",
        source_categories=("diagnosis", "symptom"),
        target_categories=("procedure",),
        description=(
            "A diagnosis, condition, symptom, or finding may be investigated by a "
            "procedure or test."
        ),
    ),
    RelationDefinition(
        id="broader_than",
        source_categories=("broader_term",),
        target_categories=("narrower_term",),
        description="The source concept is broader than the target concept.",
    ),
    RelationDefinition(
        id="narrower_than",
        source_categories=("narrower_term",),
        target_categories=("broader_term",),
        description="The source concept is narrower than the target concept.",
    ),
)

RELATION_DESCRIPTIONS: dict[QueryExpansionRelation, str] = {
    relation.id: relation.description for relation in MEDICAL_RELATION_DEFINITIONS
}
