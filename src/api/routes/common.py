"""Shared helpers for API route modules."""

from flask import jsonify

from main_utils import HTTPResponses


def unspecified_server_error():
    """Return a generic internal-server-error response."""
    return (
        jsonify(error="Something went wrong; please consult the logs."),
        HTTPResponses.INTERNAL_SERVER_ERROR,
    )


def path_arg_error(parent_endpoint: str, path_arg: str, possible_path_args: list[str]):
    """Return a standardized invalid path-argument response."""
    return (
        jsonify(
            error=f"No such path argument '{path_arg}' for '{parent_endpoint}' endpoint.",
            possible_path_args=[f"/{p}" for p in possible_path_args],
        ),
        HTTPResponses.BAD_REQUEST,
    )
