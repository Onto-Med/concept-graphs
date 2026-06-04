"""Routes for LLM-first query expansion."""

import logging

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from src.api.responses import HTTPResponses
from src.query_expansion.generator import LangChainExpansionGenerator
from src.query_expansion.models import QueryExpansionRequest
from src.query_expansion.service import QueryExpansionService

logger = logging.getLogger(__name__)


def _token_from_authorization_value(value: str | None) -> str | None:
    authorization = (value or "").strip()
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token.strip()
    return authorization


def _provider_api_key_from_headers() -> str | None:
    """Read provider API keys from auth/proxy-friendly headers.

    Some reverse proxies need explicit configuration to forward the standard
    ``Authorization`` header, so we also accept dedicated API-key headers.
    """
    for header_name in (
        "Authorization",
        "X-Authorization",
        "X-Forwarded-Authorization",
    ):
        if token := _token_from_authorization_value(request.headers.get(header_name)):
            return token

    for environ_name in ("HTTP_AUTHORIZATION", "REDIRECT_HTTP_AUTHORIZATION"):
        if token := _token_from_authorization_value(request.environ.get(environ_name)):
            return token

    for header_name in ("X-API-Key", "X-LLM-API-Key", "X-Blablador-API-Key"):
        if token := request.headers.get(header_name, "").strip():
            return token

    return None


def _with_authorization_api_key(
    expansion_request: QueryExpansionRequest,
) -> QueryExpansionRequest:
    token = _provider_api_key_from_headers()
    if token is None or expansion_request.llm.options.get("api_key"):
        return expansion_request
    llm_options = dict(expansion_request.llm.options)
    llm_options["api_key"] = token
    return expansion_request.model_copy(
        update={
            "llm": expansion_request.llm.model_copy(update={"options": llm_options})
        }
    )


def create_query_expansion_blueprint():
    """Create the query-expansion blueprint."""
    blueprint = Blueprint("query_expansion_routes", __name__)

    @blueprint.route("/query-expansion", methods=["POST"])
    def expand_query():
        if request.headers.get("Content-Type") != "application/json":
            return jsonify("Wrong content type; need 'application/json'"), int(
                HTTPResponses.BAD_REQUEST
            )
        try:
            expansion_request = _with_authorization_api_key(
                QueryExpansionRequest.model_validate(request.json)
            )
            service = QueryExpansionService(generator=LangChainExpansionGenerator())
            response = service.expand(expansion_request)
        except ValidationError as exc:
            return jsonify(
                error="Invalid query-expansion request.", details=exc.errors()
            ), int(HTTPResponses.BAD_REQUEST)
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            logger.warning("Query expansion failed: %s", exc)
            return jsonify(error=str(exc)), int(HTTPResponses.BAD_REQUEST)
        except Exception as exc:
            logger.exception("Unexpected query-expansion error.")
            return jsonify(error=str(exc)), int(HTTPResponses.INTERNAL_SERVER_ERROR)

        return jsonify(response.model_dump(mode="json")), int(HTTPResponses.OK)

    return blueprint
