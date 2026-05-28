import logging
import pathlib
from typing import Optional

import flask

from main_utils import (
    AppContext,
    PipelineContext,
    ProcessContext,
    RagContext,
    StorageContext,
)
from src.api.routes import register_routes
from src.pipeline.processes import populate_running_processes


def setup(
    static_folder: str = "api",
    static_url_path: str = "",
    file_storage_dir: str = "tmp",
    logging_setup_tuples: Optional[list[tuple]] = None,
) -> AppContext:
    """Create the Flask app and initialize shared runtime state."""
    if logging_setup_tuples is None:
        logging_setup_tuples = [
            ("werkzeug", logging.WARN),
            ("marqo", logging.WARN),
        ]

    app = flask.Flask(
        __name__, static_folder=static_folder, static_url_path=static_url_path
    )
    for logger_name, level in logging_setup_tuples:
        logging.getLogger(logger_name).setLevel(level)

    root_logger = logging.getLogger()
    root_logger.propagate = False
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(flask.logging.default_handler)

    app_context = AppContext(
        app=app,
        processes=ProcessContext(running={}, threads={}),
        pipeline=PipelineContext(active_objects={}),
        storage=StorageContext(file_storage_dir=pathlib.Path(file_storage_dir)),
        rag=RagContext(),
    )
    app_context.storage.file_storage_dir.mkdir(exist_ok=True)
    populate_running_processes(
        app_context.app,
        app_context.storage.file_storage_dir,
        app_context.processes.running,
    )
    return app_context


app_context = setup(static_folder="api", static_url_path="", file_storage_dir="tmp")
register_routes(app_context)


if __name__ in ["__main__"]:
    app_context.app.run(host="0.0.0.0", port=9010)
