"""LLM generation for query expansion."""

import json
import re
from collections.abc import Callable
from typing import Any, Protocol

from src.query_expansion.models import ExpansionGeneration, QueryExpansionRequest
from src.query_expansion.prompts import build_generation_prompt_from_profile


class ExpansionGenerator(Protocol):
    """Protocol for LLM-backed expansion generators."""

    def generate(self, request: QueryExpansionRequest) -> ExpansionGeneration:
        """Generate raw, ungrounded expansion candidates for a request."""


class LangChainExpansionGenerator:
    """LangChain-backed structured generator.

    The generator keeps LangChain as the default project LLM framework while
    still validating all LLM output with the Pydantic ``ExpansionGeneration``
    model. A concrete LangChain chat model/runnable can be injected for tests or
    custom deployments. If none is provided, a small provider factory supports
    ``ollama`` and OpenAI-compatible chat endpoints.
    """

    def __init__(
        self,
        llm: Any | None = None,
        llm_factory: Callable[[QueryExpansionRequest], Any] | None = None,
    ):
        self._llm = llm
        self._llm_factory = llm_factory

    def generate(self, request: QueryExpansionRequest) -> ExpansionGeneration:
        """Generate and Pydantic-validate structured LangChain output."""
        llm = self._llm or self._build_llm(request)
        prompt = build_generation_prompt(request)

        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(ExpansionGeneration)
            result = structured_llm.invoke(prompt)
            return _validate_generation(result)

        result = llm.invoke(prompt)
        return _validate_generation(_extract_json_payload(result))

    def _build_llm(self, request: QueryExpansionRequest) -> Any:
        if self._llm_factory is not None:
            return self._llm_factory(request)

        provider = request.llm.options.get("provider", "ollama")
        if provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "langchain-ollama is required for Ollama query expansion."
                ) from exc

            return ChatOllama(
                model=request.llm.model,
                base_url=request.llm.options.get("base_url", "http://localhost:11434"),
                temperature=request.llm.options.get("temperature", 0.0),
            )

        if provider in {"openai", "blablador"}:
            try:
                from langchain_openai import ChatOpenAI
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "langchain-openai is required for OpenAI-compatible query expansion."
                ) from exc

            return ChatOpenAI(
                model=request.llm.model,
                base_url=request.llm.options.get("base_url"),
                api_key=request.llm.options.get("api_key"),
                temperature=request.llm.options.get("temperature", 0.0),
            )

        raise ValueError(f"Unsupported LangChain query-expansion provider: {provider}")


class PydanticAIExpansionGenerator:
    """PydanticAI-backed structured generator.

    The import is intentionally lazy so the rest of the query-expansion package can
    be imported in environments where pydantic-ai is not installed yet.
    """

    def generate(self, request: QueryExpansionRequest) -> ExpansionGeneration:
        """Run a PydanticAI agent and return structured expansion candidates."""
        try:
            from pydantic_ai import Agent
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pydantic-ai is required for LLM query expansion. Install it or use "
                "a test/fake ExpansionGenerator implementation."
            ) from exc

        prompt = build_generation_prompt(request)
        agent = Agent(
            request.llm.model,
            result_type=ExpansionGeneration,
            system_prompt=request.llm.system_prompt,
            **request.llm.options,
        )
        result = agent.run_sync(prompt)
        return result.data


def _validate_generation(value: Any) -> ExpansionGeneration:
    if isinstance(value, ExpansionGeneration):
        return value
    return ExpansionGeneration.model_validate(value)


def _extract_json_payload(value: Any) -> dict[str, Any]:
    content = getattr(value, "content", value)
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        text = content.strip()
        fenced_json = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced_json:
            text = fenced_json.group(1).strip()
        return json.loads(text)
    raise TypeError(f"Cannot extract JSON query-expansion payload from {type(value)!r}")


def build_generation_prompt(request: QueryExpansionRequest) -> str:
    """Build the localized/customized prompt used by the LLM generator."""
    return build_generation_prompt_from_profile(request)
