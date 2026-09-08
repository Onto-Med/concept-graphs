"""Domain prompt profile loading for query expansion."""

import json
from pathlib import Path
from typing import Any

import yaml

from src.query_expansion.categories import (
    CATEGORY_DESCRIPTIONS,
    DEFAULT_EXPANSION_CATEGORIES,
    ExpansionCategory,
)
from src.query_expansion.models import QueryExpansionRequest

DEFAULT_PROFILE_DIR = Path("conf/query-expansion/profiles")
DEFAULT_PROMPT_DIR = DEFAULT_PROFILE_DIR
DEFAULT_LANGUAGE = "en"
DEFAULT_DOMAIN_PROFILE = "medical-en"
SCHEMA_INSTRUCTION = (
    "Return JSON matching this schema exactly: "
    '{"candidates": [{"term": "...", "category": "...", "rationale": "..."}], '
    '"concepts": [{"id": "...", "label": "...", "category": "...", '
    '"terms": ["..."]}], "relations": [{"source_concept_id": "...", '
    '"relation": "...", "target_concept_id": "...", "confidence": 0.0}]}. '
    "The output is validated as an ExpansionGeneration Pydantic model."
)

FALLBACK_PROMPT_TEMPLATE = """Generate medical query-expansion candidates for the provided term.

Term: {term}
Candidate language: {language_name} ({language})
Limit per category: {limit_per_category}
Categories: {categories_json}
Relations and allowed category connections: {relations_json}

Use exactly one of the requested category IDs for each candidate and concept.
Use only the listed relation IDs, and only between concepts with allowed source
and target categories.
Concept terms must be the original term or generated candidate terms.
Return only structured output.

{schema_instruction}
"""


def build_generation_prompt_from_profile(request: QueryExpansionRequest) -> str:
    """Build a localized/customized prompt for query expansion."""
    profile = load_domain_profile(request.prompt.profile or request.language)
    request = request_with_profile_defaults(request, profile)

    language_name = profile.get("language_name", request.language)
    template = request.prompt.template or profile.get(
        "prompt_template", FALLBACK_PROMPT_TEMPLATE
    )
    category_descriptions = _category_descriptions(request, profile)
    return template.format(
        term=request.term,
        language=request.language,
        language_name=language_name,
        limit_per_category=request.limit_per_category,
        categories_json=json.dumps(category_descriptions, ensure_ascii=False),
        relations_json=json.dumps(_relation_definitions(request), ensure_ascii=False),
        schema_instruction=profile.get("schema_instruction", SCHEMA_INSTRUCTION),
    )


def list_domain_profiles() -> list[str]:
    """List available query-expansion domain profile names."""
    if not DEFAULT_PROMPT_DIR.exists():
        return []
    return sorted(path.stem for path in DEFAULT_PROMPT_DIR.glob("*.yml"))


def load_domain_profile(profile_name: str | None) -> dict[str, Any]:
    """Load a query-expansion domain profile with English medical fallback."""
    resolved = _resolve_profile_name(profile_name or DEFAULT_LANGUAGE)
    return _load_prompt_profile(resolved) or _load_prompt_profile(DEFAULT_DOMAIN_PROFILE)


def domain_profile_metadata(profile_name: str) -> dict[str, Any]:
    """Return API/GUI-safe metadata for a query-expansion domain profile."""
    resolved = _resolve_profile_name(profile_name)
    profile = _load_prompt_profile(resolved)
    if not profile:
        raise ValueError(f"Unknown query-expansion domain profile: {profile_name}")
    categories = profile_category_descriptions(profile)
    return {
        "name": resolved,
        "language_name": profile.get("language_name", resolved),
        "categories": [
            {"id": category, "description": description}
            for category, description in categories.items()
        ],
        "default_categories": profile_default_categories(profile),
    }


def profile_category_descriptions(profile: dict[str, Any]) -> dict[str, str]:
    """Return the category vocabulary defined by a profile or fallback defaults."""
    descriptions = profile.get("category_descriptions", {}) or {}
    return dict(descriptions) if descriptions else dict(CATEGORY_DESCRIPTIONS)


def profile_default_categories(profile: dict[str, Any]) -> list[str]:
    """Return profile default categories or built-in fallback defaults."""
    descriptions = profile_category_descriptions(profile)
    defaults = profile.get("default_categories") or list(DEFAULT_EXPANSION_CATEGORIES)
    unknown_defaults = [category for category in defaults if category not in descriptions]
    if unknown_defaults:
        raise ValueError(
            "Unknown query-expansion default category IDs in prompt profile: "
            + ", ".join(sorted(unknown_defaults))
        )
    return list(defaults)


def request_with_profile_defaults(
    request: QueryExpansionRequest, profile: dict[str, Any] | None = None
) -> QueryExpansionRequest:
    """Apply domain-profile category defaults and validation to a request."""
    profile = profile or load_domain_profile(request.prompt.profile or request.language)
    descriptions = profile_category_descriptions(profile)
    categories = list(request.categories) if request.categories else profile_default_categories(profile)
    unknown_categories = [category for category in categories if category not in descriptions]
    if unknown_categories:
        raise ValueError(
            "Unknown query-expansion category IDs for selected prompt profile: "
            + ", ".join(sorted(set(unknown_categories)))
        )
    if categories == request.categories:
        return request
    return request.model_copy(update={"categories": categories})


def _normalize_profile_name(profile_name: str) -> str:
    return profile_name.strip().lower().replace("_", "-")


def _resolve_profile_name(profile_name: str) -> str:
    normalized = _normalize_profile_name(profile_name)
    if _profile_path(normalized).exists():
        return normalized
    medical_profile = f"medical-{normalized}"
    if _profile_path(medical_profile).exists():
        return medical_profile
    return normalized


def _profile_path(profile_name: str) -> Path:
    return DEFAULT_PROMPT_DIR / f"{profile_name}.yml"


def _load_prompt_profile(profile_name: str) -> dict[str, Any]:
    path = _profile_path(profile_name)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _relation_definitions(request: QueryExpansionRequest) -> list[dict[str, Any]]:
    selected_categories = set(request.categories)
    definitions = []
    for definition in request.relation_definitions:
        if definition.id not in request.relations:
            continue
        source_categories = [
            category
            for category in definition.source_categories
            if category in selected_categories
        ]
        target_categories = [
            category
            for category in definition.target_categories
            if category in selected_categories
        ]
        if not source_categories or not target_categories:
            continue
        definitions.append(
            {
                "id": definition.id,
                "source_categories": source_categories,
                "target_categories": target_categories,
                "description": definition.description,
            }
        )
    return definitions


def _category_descriptions(
    request: QueryExpansionRequest, profile: dict[str, Any]
) -> dict[ExpansionCategory, str]:
    profile_descriptions = profile_category_descriptions(profile)
    descriptions = {
        category: profile_descriptions.get(
            category, CATEGORY_DESCRIPTIONS.get(category, category)
        )
        for category in request.categories
    }
    descriptions.update(
        {
            category: description
            for category, description in request.prompt.category_descriptions.items()
            if category in request.categories
        }
    )
    return {category: descriptions[category] for category in request.categories}
