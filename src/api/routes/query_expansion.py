"""Routes for LLM-first query expansion."""

import logging

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from src.api.responses import HTTPResponses
from src.query_expansion.generator import LangChainExpansionGenerator
from src.query_expansion.models import QueryExpansionRequest
from src.query_expansion.service import QueryExpansionService

logger = logging.getLogger(__name__)


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
            expansion_request = QueryExpansionRequest.model_validate(request.json)
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
