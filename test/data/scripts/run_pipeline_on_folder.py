#!/usr/bin/env python3
"""Run the complete concept-graphs pipeline on a folder of documents.

This helper is intended for regenerating local test fixtures and manual smoke
runs. It uses the Flask app factory and test client, so no HTTP server has to be
started separately.

Run from the repository root, for example:

    uv run python test/data/scripts/run_pipeline_on_folder.py \
     test/data/documents/grascco/ --process grascco --language de --file-storage-dir test/data/results \
     --pipeline-config test/data/configs/pipeline-config_de_test.json --skip-steps integration --skip-present

The input folder is zipped into a temporary archive and uploaded to the
``POST /pipeline`` endpoint as multipart form data.
You can use configuration files from test/data/configs
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import pathlib
import shutil
import sys
import zipfile
from collections.abc import Iterable
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main


def _iter_document_files(folder: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield regular, non-hidden files from ``folder`` recursively."""
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue
        yield path


def _zip_folder(folder: pathlib.Path) -> io.BytesIO:
    """Create an in-memory zip archive from ``folder``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _iter_document_files(folder):
            archive.write(path, arcname=path.relative_to(folder))
    buffer.seek(0)
    return buffer


def _file_upload(path: pathlib.Path) -> tuple[Any, str, str]:
    """Return a Flask-test-client-compatible multipart file tuple."""
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return path.open("rb"), path.name, content_type


def _json_upload(payload: dict, filename: str) -> tuple[io.BytesIO, str, str]:
    """Return an in-memory JSON upload for multipart form data."""
    return (
        io.BytesIO(json.dumps(payload).encode("utf-8")),
        filename,
        "application/json",
    )


def _load_pipeline_config(path: pathlib.Path | None) -> dict[str, Any]:
    """Load a complete pipeline JSON config, if one was provided."""
    if path is None:
        return {}
    with path.open("rb") as config_file:
        return json.load(config_file)


def _build_form_data(
    args: argparse.Namespace, archive: io.BytesIO, pipeline_config: dict[str, Any]
) -> dict[str, Any]:
    """Build multipart form fields for the pipeline endpoint."""
    data: dict[str, Any] = {
        "data": (archive, f"{args.process}.zip", "application/zip"),
    }

    step_configs = pipeline_config.get("config", {})
    config_sections = {
        "data_config": step_configs.get("data"),
        "embedding_config": step_configs.get("embedding"),
        "clustering_config": step_configs.get("clustering"),
        "graph_config": step_configs.get("graph"),
        "vectorstore_server_config": pipeline_config.get(
            "vectorstore_server", pipeline_config.get("vector_store_server")
        ),
    }
    for field_name, config_section in config_sections.items():
        if config_section is not None:
            data[field_name] = _json_upload(config_section, f"{field_name}.json")

    optional_files = {
        "data_config": args.data_config,
        "embedding_config": args.embedding_config,
        "clustering_config": args.clustering_config,
        "graph_config": args.graph_config,
        "labels": args.labels,
        "vectorstore_server_config": args.vectorstore_server_config,
    }
    for field_name, path in optional_files.items():
        if path is not None:
            # Explicit per-step files override values from --pipeline-config.
            data[field_name] = _file_upload(path)

    return data


def _query_string(args: argparse.Namespace) -> dict[str, str]:
    query = {
        "process": args.process,
        # The API currently reads the language from the `lang` query parameter.
        # Keep `language` too because it is the public name documented elsewhere.
        "lang": args.language,
        "language": args.language,
        "skip_present": str(args.skip_present).lower(),
        "return_statistics": str(args.return_statistics).lower(),
    }
    if args.skip_steps:
        query["skip_steps"] = ",".join(args.skip_steps)
    return query


def _cleanup_tmp_streams(file_storage_dir: pathlib.Path) -> None:
    """Remove temporary upload streams created by the pipeline endpoint."""
    tmp_streams = file_storage_dir / ".tmp_streams"
    if tmp_streams.exists():
        shutil.rmtree(tmp_streams)


def run_pipeline(args: argparse.Namespace) -> int:
    """Create the app, upload the folder archive, and print the response."""
    document_folder = args.document_folder.resolve()
    if not document_folder.is_dir():
        raise NotADirectoryError(document_folder)

    files = list(_iter_document_files(document_folder))
    if not files:
        raise ValueError(f"No document files found in {document_folder}")

    pipeline_config = _load_pipeline_config(args.pipeline_config)
    args.process = args.process or pipeline_config.get("name") or "test_corpus"
    args.language = args.language or pipeline_config.get("language") or "en"

    app = main.create_app(file_storage_dir=str(args.file_storage_dir))
    client = app.test_client()

    archive = _zip_folder(document_folder)
    form_data = _build_form_data(args, archive, pipeline_config)

    response = client.post(
        "/pipeline",
        query_string=_query_string(args),
        data=form_data,
        content_type="multipart/form-data",
    )
    if response.status_code < 400:
        _cleanup_tmp_streams(args.file_storage_dir)

    print(f"HTTP {response.status_code}")
    try:
        print(json.dumps(response.get_json(), indent=2, sort_keys=True))
    except TypeError:
        print(response.get_data(as_text=True))

    return 0 if response.status_code < 400 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full concept-graphs pipeline on a folder of documents."
    )
    parser.add_argument("document_folder", type=pathlib.Path)
    parser.add_argument("--process")
    parser.add_argument("--language")
    parser.add_argument(
        "--pipeline-config",
        type=pathlib.Path,
        help=(
            "Complete pipeline JSON config, e.g. conf/pipeline-config_de_test.json. "
            "The script extracts config.data/config.embedding/config.clustering/"
            "config.graph and vectorstore_server from it. Explicit per-step config "
            "arguments override these values."
        ),
    )
    parser.add_argument(
        "--file-storage-dir", type=pathlib.Path, default=pathlib.Path("tmp")
    )
    parser.add_argument(
        "--skip-present",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse already serialized pipeline steps when present. Enabled by "
            "default; pass --no-skip-present to force recomputation."
        ),
    )
    parser.add_argument("--return-statistics", action="store_true")
    parser.add_argument(
        "--skip-steps",
        nargs="*",
        default=[],
        choices=["data", "embedding", "clustering", "graph", "integration"],
    )

    parser.add_argument("--data-config", type=pathlib.Path)
    parser.add_argument("--embedding-config", type=pathlib.Path)
    parser.add_argument("--clustering-config", type=pathlib.Path)
    parser.add_argument("--graph-config", type=pathlib.Path)
    parser.add_argument("--labels", type=pathlib.Path)
    parser.add_argument("--vectorstore-server-config", type=pathlib.Path)

    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_pipeline(parse_args()))
