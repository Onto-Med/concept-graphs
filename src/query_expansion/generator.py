"""LLM generation for query expansion."""

import json
import re
import urllib.error
import urllib.request
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

        no_structured_output = request.llm.options.get("no_structured_output", False)
        if no_structured_output and request.llm.options.get("provider") in {
            "openai",
            "blablador",
        }:
            return _validate_generation(_openai_compatible_json_completion(request, prompt))

        if not no_structured_output and hasattr(llm, "with_structured_output"):
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


def _openai_compatible_json_completion(
    request: QueryExpansionRequest, prompt: str
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat endpoint without SDK response parsing.

    Some OpenAI-compatible providers support normal chat completions but do not
    return objects that the OpenAI SDK/LangChain parser can handle. This fallback
    is intentionally used only when ``no_structured_output`` is set.
    """
    base_url = request.llm.options.get("base_url")
    if not base_url:
        raise ValueError("base_url is required for OpenAI-compatible fallback.")

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": request.llm.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": request.llm.options.get("temperature", 0.0),
    }
    headers = {"Content-Type": "application/json"}
    api_key = request.llm.options.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    http_request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=300) as response:
            response_body = response.read().decode("utf-8")
            response_status = response.status
            response_content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI-compatible query expansion request failed with "
            f"HTTP {exc.code}: {error_body}"
        ) from exc

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "OpenAI-compatible query expansion response was not JSON "
            f"(HTTP {response_status}, Content-Type: {response_content_type}). "
            f"Response preview: {_preview_text(response_body)}"
        ) from exc

    content = _extract_openai_compatible_content(decoded)
    try:
        return _extract_json_payload(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "OpenAI-compatible query expansion message content was not valid JSON. "
            f"Content preview: {_preview_text(str(content))}"
        ) from exc


def _extract_openai_compatible_content(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise TypeError(
            f"Cannot extract chat completion content from {type(value)!r}"
        )

    choices = value.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict) and "content" in message:
                return message["content"]
            if "text" in first_choice:
                return first_choice["text"]
    if "content" in value:
        return value["content"]
    return value


def _preview_text(value: str, limit: int = 500) -> str:
    text = value.replace("\n", " ").replace("\r", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


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
