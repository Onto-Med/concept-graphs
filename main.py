import logging
import pathlib
from typing import Optional

import flask

from main_utils import PersistentObjects
from src.api.routes import register_routes
from src.pipeline.processes import populate_running_processes


def setup(
    static_folder: str = "api",
    static_url_path: str = "",
    file_storage_dir: str = "tmp",
    logging_setup_tuples: Optional[list[tuple]] = None,
) -> PersistentObjects:
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

    persistent_objects = PersistentObjects(
        app=app,
        running_processes={},
        pipeline_threads_store={},
        current_active_pipeline_objects={},
        file_storage_dir=pathlib.Path(file_storage_dir),
        active_rag=None,
    )
    persistent_objects.file_storage_dir.mkdir(exist_ok=True)
    populate_running_processes(
        persistent_objects.app,
        persistent_objects.file_storage_dir,
        persistent_objects.running_processes,
    )
    return persistent_objects


main_objects = setup(static_folder="api", static_url_path="", file_storage_dir="tmp")
register_routes(main_objects)


if __name__ in ["__main__"]:
    main_objects.app.run(host="0.0.0.0", port=9010)
