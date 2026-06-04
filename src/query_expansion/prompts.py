"""Prompt profile loading for query expansion."""

import json
from pathlib import Path
from typing import Any

import yaml

from src.query_expansion.categories import CATEGORY_DESCRIPTIONS, ExpansionCategory
from src.query_expansion.models import QueryExpansionRequest

DEFAULT_PROMPT_DIR = Path("conf/query-expansion")
DEFAULT_LANGUAGE = "en"
SCHEMA_INSTRUCTION = (
    "Return JSON matching this schema exactly: "
    '{"candidates": [{"term": "...", "category": "...", "rationale": "..."}]}. '
    "The output is validated as an ExpansionGeneration Pydantic model."
)

FALLBACK_PROMPT_TEMPLATE = """Generate medical query-expansion candidates for the provided term.

Term: {term}
Candidate language: {language_name} ({language})
Limit per category: {limit_per_category}
Categories: {categories_json}

Use exactly one of the requested category IDs for each candidate.
Return only structured output.

{schema_instruction}
"""


def build_generation_prompt_from_profile(request: QueryExpansionRequest) -> str:
    """Build a localized/customized prompt for query expansion."""
    profile_name = _normalize_profile_name(
        request.prompt.profile or request.language or DEFAULT_LANGUAGE
    )
    profile = _load_prompt_profile(profile_name) or _load_prompt_profile(
        DEFAULT_LANGUAGE
    )

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
        schema_instruction=SCHEMA_INSTRUCTION,
    )


def _normalize_profile_name(profile_name: str) -> str:
    return profile_name.lower().replace("_", "-").split("-", maxsplit=1)[0]


def _load_prompt_profile(profile_name: str) -> dict[str, Any]:
    path = DEFAULT_PROMPT_DIR / f"{profile_name}.yml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _category_descriptions(
    request: QueryExpansionRequest, profile: dict[str, Any]
) -> dict[ExpansionCategory, str]:
    profile_descriptions = profile.get("category_descriptions", {}) or {}
    descriptions = {
        category: profile_descriptions.get(
            category, CATEGORY_DESCRIPTIONS.get(category, category)
        )
        for category in request.categories
    }
    descriptions.update(request.prompt.category_descriptions)
    return {category: descriptions[category] for category in request.categories}
