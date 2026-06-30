"""Synchronize the project/API/Docker version references.

Usage:
    uv run --no-sync python -m src.scripts.set_version 1.2.0
    uv run --no-sync python -m src.scripts.set_version 1.2.0 --check

The script intentionally updates only project-owned version locations and avoids
blind global replacement so dependency versions in ``uv.lock`` stay untouched.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
IMAGE_REF_PATTERN = re.compile(
    r"(ghcr\.io/onto-med/concept-graphs/concept-graphs-api:)[0-9A-Za-z][0-9A-Za-z._+-]*"
)


def _replace(pattern: str | re.Pattern[str], replacement: str, text: str) -> str:
    return re.sub(pattern, replacement, text, count=1, flags=0)


def _update_version_file(text: str, version: str) -> str:
    return f"{version}\n"


def _update_pyproject(text: str, version: str) -> str:
    return _replace(
        re.compile(r'(?m)^(version\s*=\s*")[^"]+("\s*)$'),
        rf"\g<1>{version}\2",
        text,
    )


def _update_uv_lock(text: str, version: str) -> str:
    return _replace(
        re.compile(r'(\[\[package\]\]\nname = "concept-graphs"\nversion = ")[^"]+(")'),
        rf"\g<1>{version}\2",
        text,
    )


def _update_openapi(text: str, version: str) -> str:
    return _replace(
        re.compile(r"(?m)^(  version:\s*)[^\n]+$"),
        rf"\g<1>{version}",
        text,
    )


def _update_image_refs(text: str, version: str) -> str:
    return IMAGE_REF_PATTERN.sub(rf"\g<1>{version}", text)


def _update_workflow_description(text: str, version: str) -> str:
    return re.sub(
        r"(Version/tag for the concept-graphs-api image, e\.g\. )[0-9A-Za-z][0-9A-Za-z._+-]*",
        rf"\g<1>{version}",
        text,
    )


def _candidate_files() -> dict[Path, list[Callable[[str, str], str]]]:
    files: dict[Path, list[Callable[[str, str], str]]] = {
        PROJECT_ROOT / "VERSION": [_update_version_file],
        PROJECT_ROOT / "pyproject.toml": [_update_pyproject],
        PROJECT_ROOT / "uv.lock": [_update_uv_lock],
        PROJECT_ROOT / "api" / "concept-graphs-api.yml": [_update_openapi],
        PROJECT_ROOT / "README.md": [_update_image_refs],
        PROJECT_ROOT / ".github" / "workflows" / "build-docker-image.yml": [
            _update_workflow_description
        ],
    }
    for path in PROJECT_ROOT.glob("docker-compose*.yml"):
        files[path] = [_update_image_refs]
    return files


def update_file(path: Path, version: str) -> bool:
    original = path.read_text()
    updated = original
    for updater in _candidate_files()[path]:
        updated = updater(updated, version)
    if updated == original:
        return False
    path.write_text(updated)
    return True


def planned_changes(version: str) -> list[Path]:
    changed = []
    for path, updaters in _candidate_files().items():
        if not path.exists():
            continue
        original = path.read_text()
        updated = original
        for updater in updaters:
            updated = updater(updated, version)
        if updated != original:
            changed.append(path)
    return changed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize project version references."
    )
    parser.add_argument("version", help="Target version, e.g. 1.2.0")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check whether files already contain the requested version.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would change without writing them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not VERSION_PATTERN.match(args.version):
        print(
            f"Invalid version '{args.version}'. Expected semantic version like 1.2.0.",
            file=sys.stderr,
        )
        return 2

    changes = planned_changes(args.version)
    relative_changes = [path.relative_to(PROJECT_ROOT) for path in changes]

    if args.check:
        if relative_changes:
            print("Version references are not synchronized:")
            for path in relative_changes:
                print(f"  would update {path}")
            return 1
        print("Version references are synchronized.")
        return 0

    if args.dry_run:
        if relative_changes:
            print("Would update:")
            for path in relative_changes:
                print(f"  {path}")
        else:
            print("No files would change.")
        return 0

    for path in changes:
        update_file(path, args.version)

    if relative_changes:
        print(f"Updated version references to {args.version}:")
        for path in relative_changes:
            print(f"  {path}")
    else:
        print(f"All version references already set to {args.version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
