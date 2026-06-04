"""RAG prompt profile loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_RAG_PROMPT_DIR = Path("conf/rag/localization")
DEFAULT_INPUT_VARIABLES = ["summaries", "question"]

FALLBACK_TEMPLATES = {
    "en": """
        Answer the question using only the provided SOURCES.
        The answer language is English. Always answer in English.

        Rules:
        - Use only information that is explicitly supported by the SOURCES.
        - Do not use outside knowledge.
        - If the SOURCES do not contain enough information to answer, say exactly: "No source I can find."
        - Keep the answer concise.
        - When possible, cite the supporting source markers such as [0], [1].
        - Do not output chain-of-thought, analysis, reasoning steps, or internal deliberation.
        - Do not repeat the question, sources, separators, or instructions.
        - Output only the final answer.

        QUESTION:
        {question}

        SOURCES:
        {summaries}

        FINAL ANSWER IN ENGLISH:
        """,
    "de": """
        Beantworte die Frage ausschließlich anhand der bereitgestellten QUELLEN.
        Die Antwortsprache ist Deutsch. Antworte immer auf Deutsch.

        Regeln:
        - Verwende nur Informationen, die ausdrücklich durch die QUELLEN gestützt werden.
        - Nutze kein externes Wissen.
        - Wenn die QUELLEN keine ausreichende Antwort enthalten, sage exakt: "Keine Quelle die ich finden kann."
        - Antworte kurz und präzise.
        - Verweise, wenn möglich, auf die stützenden Quellenmarker wie [0], [1].
        - Gib keine Gedankenkette, Analyse, Begründungsschritte oder internen Überlegungen aus.
        - Wiederhole nicht die Frage, Quellen, Trennzeichen oder Anweisungen.
        - Gib ausschließlich die finale Antwort aus.

        FRAGE:
        {question}

        QUELLEN:
        {summaries}

        FINALE ANTWORT AUF DEUTSCH:
        """,
}


@dataclass(frozen=True)
class RagPromptConfig:
    """Resolved RAG prompt config."""

    template: str
    input_variables: list[str]


def normalize_prompt_profile(language: str | None) -> str:
    """Normalize language/profile names such as ``de-DE`` to ``de``."""
    profile = (language or "en").lower().replace("_", "-")
    return profile.split("-", 1)[0]


def load_rag_prompt_profile(
    profile: str, prompt_dir: Path = DEFAULT_RAG_PROMPT_DIR
) -> dict[str, Any]:
    """Load a RAG prompt profile from YAML, falling back to in-code defaults."""
    normalized_profile = normalize_prompt_profile(profile)
    profile_path = prompt_dir / f"{normalized_profile}.yml"
    if not profile_path.exists():
        return {
            "template": FALLBACK_TEMPLATES.get(
                normalized_profile, FALLBACK_TEMPLATES["en"]
            ),
            "input_variables": DEFAULT_INPUT_VARIABLES,
        }
    profile_data = yaml.safe_load(profile_path.read_text()) or {}
    if not isinstance(profile_data, dict):
        return {}
    return profile_data


def resolve_rag_prompt_config(
    language: str = "en", prompt_template_config: dict[str, Any] | None = None
) -> RagPromptConfig:
    """Resolve file-based, legacy, and request-level RAG prompt config.

    ``prompt_template_config`` remains backwards compatible with the previous
    shape: ``{"templates": {"en": "..."}, "input_variables": [...]}``.
    It may also contain ``profile`` and/or a direct ``template`` override.
    """
    profile_name = normalize_prompt_profile(language)
    if prompt_template_config and prompt_template_config.get("profile"):
        profile_name = str(prompt_template_config["profile"])

    profile = load_rag_prompt_profile(profile_name)
    fallback_template = FALLBACK_TEMPLATES.get(
        normalize_prompt_profile(language), FALLBACK_TEMPLATES["en"]
    )
    template = profile.get("template", fallback_template)
    input_variables = profile.get("input_variables", DEFAULT_INPUT_VARIABLES)

    if prompt_template_config:
        templates = prompt_template_config.get("templates")
        if isinstance(templates, dict):
            template = templates.get(
                normalize_prompt_profile(language),
                templates.get(profile_name, template),
            )
        template = prompt_template_config.get("template", template)
        input_variables = prompt_template_config.get("input_variables", input_variables)

    return RagPromptConfig(template=template, input_variables=list(input_variables))
