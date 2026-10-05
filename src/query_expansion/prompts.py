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
from src.query_expansion.relations import (
    DEFAULT_PROFILE_RELATIONS,
    MEDICAL_RELATION_DEFINITIONS,
    RelationDefinition,
)

DEFAULT_PROFILE_DIR = Path("conf/query-expansion/profiles")
DEFAULT_ONTOLOGY_DIR = Path("conf/query-expansion/ontologies")
DEFAULT_PROMPT_DIR = DEFAULT_PROFILE_DIR
DEFAULT_LANGUAGE = "en"
DEFAULT_DOMAIN_PROFILE = "medical_en"
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
    normalized = _normalize_profile_name(profile_name or DEFAULT_DOMAIN_PROFILE)
    return _load_prompt_profile(normalized) or _load_prompt_profile(DEFAULT_DOMAIN_PROFILE)


def domain_profile_metadata(profile_name: str) -> dict[str, Any]:
    """Return API/GUI-safe metadata for a query-expansion domain profile."""
    normalized = _normalize_profile_name(profile_name)
    profile = _load_prompt_profile(normalized)
    if not profile:
        raise ValueError(f"Unknown query-expansion domain profile: {profile_name}")
    categories = profile_category_descriptions(profile)
    category_labels = profile_category_labels(profile)
    return {
        "name": normalized,
        "language_name": profile.get("language_name", normalized),
        "categories": [
            {
                "id": category,
                "label": category_labels.get(category, _category_label(category)),
                "description": description,
            }
            for category, description in categories.items()
        ],
        "default_categories": profile_default_categories(profile),
        "relations": profile_relation_metadata(profile),
        "default_relations": profile_default_relations(profile),
    }


def profile_relation_metadata(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Return semantic relation metadata applicable to a domain profile."""
    profile_categories = set(profile_category_descriptions(profile))
    relation_labels = profile_relation_labels(profile)
    relations = []
    for relation in profile_relation_definitions(profile):
        source_categories = [
            category
            for category in relation.source_categories
            if category in profile_categories
        ]
        target_categories = [
            category
            for category in relation.target_categories
            if category in profile_categories
        ]
        if not source_categories or not target_categories:
            continue
        relations.append(
            {
                "id": relation.id,
                "label": relation_labels.get(relation.id, _relation_label(relation)),
                "description": relation.description,
                "source_categories": source_categories,
                "target_categories": target_categories,
            }
        )
    return relations


def profile_relation_definitions(profile: dict[str, Any]) -> list[RelationDefinition]:
    """Return ontology/profile-owned semantic relation definitions or medical fallbacks."""
    configured_definitions = profile.get("relation_definitions") or profile.get("relations") or []
    if not configured_definitions:
        return [relation.model_copy(deep=True) for relation in MEDICAL_RELATION_DEFINITIONS]
    relation_descriptions = profile.get("relation_descriptions", {}) or {}
    return [
        RelationDefinition.model_validate(
            {
                **definition,
                "description": relation_descriptions.get(
                    definition["id"],
                    definition.get("description") or _relation_label_from_id(definition["id"]),
                ),
            }
        )
        for definition in configured_definitions
    ]


def profile_default_relations(profile: dict[str, Any]) -> list[str]:
    """Return default semantic relation IDs applicable to a domain profile."""
    available_relations = {relation["id"] for relation in profile_relation_metadata(profile)}
    configured_defaults = profile.get("default_relations") or list(
        DEFAULT_PROFILE_RELATIONS
    )
    return [
        relation_id
        for relation_id in configured_defaults
        if relation_id in available_relations
    ]


def profile_category_descriptions(profile: dict[str, Any]) -> dict[str, str]:
    """Return the category vocabulary defined by a profile/ontology or fallback defaults."""
    descriptions = profile.get("category_descriptions", {}) or {}
    if descriptions:
        return dict(descriptions)
    categories = profile.get("categories") or []
    if categories:
        return {
            _category_id(category): _category_label(_category_id(category))
            for category in categories
        }
    return dict(CATEGORY_DESCRIPTIONS)


def profile_category_labels(profile: dict[str, Any]) -> dict[str, str]:
    """Return user-facing category labels defined by a profile or generated from IDs."""
    descriptions = profile_category_descriptions(profile)
    configured_labels = profile.get("category_labels", {}) or {}
    return {
        category: configured_labels.get(category, _category_label(category))
        for category in descriptions
    }


def profile_relation_labels(profile: dict[str, Any]) -> dict[str, str]:
    """Return user-facing relation labels defined by a profile or generated from IDs."""
    configured_labels = profile.get("relation_labels", {}) or {}
    return {
        relation.id: configured_labels.get(relation.id, _relation_label(relation))
        for relation in profile_relation_definitions(profile)
    }


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
    categories = (
        list(request.categories)
        if request.categories
        else profile_default_categories(profile)
    )
    unknown_categories = [category for category in categories if category not in descriptions]
    if unknown_categories:
        raise ValueError(
            "Unknown query-expansion category IDs for selected prompt profile: "
            + ", ".join(sorted(set(unknown_categories)))
        )
    updates: dict[str, Any] = {}
    if categories != request.categories:
        updates["categories"] = categories
    profile_definitions = profile_relation_definitions(profile)
    if request.relation_definitions == list(MEDICAL_RELATION_DEFINITIONS):
        updates["relation_definitions"] = profile_definitions
    if not updates:
        return request
    return request.model_copy(update=updates)


def _normalize_profile_name(profile_name: str) -> str:
    hyphenated = "-".join(profile_name.strip().lower().replace("_", "-").split())
    return hyphenated.replace("-", "_")


def _category_label(category: str) -> str:
    return category.replace("_", " ").title()


def _relation_label(relation: RelationDefinition) -> str:
    return _relation_label_from_id(relation.id)


def _relation_label_from_id(relation_id: str) -> str:
    return relation_id.replace("_", " ").title()


def _profile_path(profile_name: str) -> Path:
    return DEFAULT_PROMPT_DIR / f"{profile_name}.yml"


def _load_prompt_profile(profile_name: str) -> dict[str, Any]:
    path = _profile_path(profile_name)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        profile = yaml.safe_load(file) or {}
    return _profile_with_ontology(profile)


def _profile_with_ontology(profile: dict[str, Any]) -> dict[str, Any]:
    ontology_name = profile.get("ontology")
    if not ontology_name:
        return profile
    ontology = _load_ontology(str(ontology_name))
    merged = {**ontology, **profile}
    for key in ("categories", "default_categories", "relations", "default_relations"):
        if key not in profile and key in ontology:
            merged[key] = ontology[key]
    return merged


def _load_ontology(ontology_name: str) -> dict[str, Any]:
    path = DEFAULT_ONTOLOGY_DIR / f"{_normalize_profile_name(ontology_name)}.yml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _category_id(category: str | dict[str, Any]) -> str:
    if isinstance(category, dict):
        return str(category["id"])
    return str(category)


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
