"""Grounding generated query-expansion candidates against sources."""

from collections import defaultdict

from src.query_expansion.models import (
    GeneratedExpansionCandidate,
    GroundedExpansionCandidate,
    GroundingOptions,
    GroundingStatus,
)
from src.query_expansion.sources.base import ExpansionSource


def ground_candidate(
    candidate: GeneratedExpansionCandidate,
    sources: list[ExpansionSource],
    options: GroundingOptions,
) -> GroundedExpansionCandidate | None:
    """Ground one LLM-generated candidate against all configured sources.

    Returns ``None`` when the candidate should be filtered out according to the
    grounding options.
    """
    evidence = [item for source in sources for item in source.ground(candidate)]
    confidence = max((item.score for item in evidence), default=0.0)
    if confidence >= options.minimum_score and evidence:
        status = GroundingStatus.GROUNDED
    elif options.reject_below_minimum and confidence < options.minimum_score:
        status = GroundingStatus.REJECTED
    else:
        status = GroundingStatus.LLM_ONLY

    if status == GroundingStatus.LLM_ONLY and not options.include_llm_only:
        return None
    if status == GroundingStatus.REJECTED:
        return None

    return GroundedExpansionCandidate(
        term=candidate.term,
        category=candidate.category,
        status=status,
        confidence=confidence,
        evidence=evidence,
        rationale=candidate.rationale,
    )


def group_grounded_candidates(
    candidates: list[GroundedExpansionCandidate],
) -> dict[str, list[GroundedExpansionCandidate]]:
    """Group final candidates by semantic expansion category."""
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.category].append(candidate)
    return dict(grouped)
