import argparse
import logging
import os
import pathlib

import flask

from src.api.context import (
    AppContext,
    PipelineContext,
    ProcessContext,
    RagContext,
    StorageContext,
)
from src.api.routes import register_routes
from src.pipeline.processes import populate_running_processes


def configure_logging(logging_setup_tuples: list[tuple] | None = None) -> None:
    """Configure application logging defaults.

    ``LOG_LEVEL`` controls application/root logging and defaults to ``INFO`` so
    operational messages such as RAG indexing progress are visible in container
    logs. Noisy dependency loggers are still kept at warning level by default.
    """
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    if logging_setup_tuples is None:
        logging_setup_tuples = [
            ("werkzeug", logging.WARN),
            ("marqo", logging.WARN),
        ]

    for logger_name, level in logging_setup_tuples:
        logging.getLogger(logger_name).setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.propagate = False
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    flask.logging.default_handler.setLevel(log_level)
    root_logger.addHandler(flask.logging.default_handler)


def create_app_context(
    app: flask.Flask,
    file_storage_dir: str = "tmp",
) -> AppContext:
    """Create shared application runtime state for a Flask app."""
    app_context = AppContext(
        app=app,
        processes=ProcessContext(running={}, threads={}),
        pipeline=PipelineContext(active_objects={}),
        storage=StorageContext(file_storage_dir=pathlib.Path(file_storage_dir)),
        rag=RagContext(active_by_process={}),
    )
    app_context.storage.file_storage_dir.mkdir(exist_ok=True)
    populate_running_processes(
        app_context.app,
        app_context.storage.file_storage_dir,
        app_context.processes.running,
    )
    return app_context


def create_app(
    static_folder: str = "api",
    static_url_path: str = "",
    file_storage_dir: str = "tmp",
    logging_setup_tuples: list[tuple] | None = None,
) -> flask.Flask:
    """Create and configure the Flask application."""
    configure_logging(logging_setup_tuples)
    app = flask.Flask(
        __name__, static_folder=static_folder, static_url_path=static_url_path
    )
    app_context = create_app_context(app=app, file_storage_dir=file_storage_dir)
    register_routes(app_context)
    app.extensions["concept_graphs_context"] = app_context
    return app


def parse_args() -> argparse.Namespace:
    """Parse development-server command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Concept Graphs Flask API.")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=9010,
        help="Port to bind the development server to. Defaults to 9010.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/interface to bind. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--storage-dir",
        default="tmp",
        help="Directory for process artifacts. Defaults to tmp.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask's development server in debug mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_app(file_storage_dir=args.storage_dir).run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )
