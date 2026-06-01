import re
from pathlib import Path

import yaml

from main import create_app

HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
IGNORED_STATIC_RULES = {
    ("/", "GET"),
    ("/openapi", "GET"),
    ("/<path:filename>", "GET"),
}
# Registered in Flask but currently historical/non-business behavior. The OpenAPI
# spec intentionally documents only GET graph artifact retrieval.
ALLOWED_UNDOCUMENTED_RULES = {
    ("/graph/<path_arg>", "POST"),
}


def _openapi_path_to_regex(path: str) -> re.Pattern:
    escaped = re.escape(path)
    pattern = re.sub(r"\\\{[^/]+\\\}", r"[^/]+", escaped)
    return re.compile(f"^{pattern}$")


def _flask_rule_to_regex(rule: str) -> re.Pattern:
    escaped = re.escape(rule)
    pattern = re.sub(r"<path:[^/]+>", r".+", escaped)
    pattern = re.sub(r"<[^/]+>", r"[^/]+", pattern)
    return re.compile(f"^{pattern}$")


def _documented_operations(spec: dict) -> set[tuple[str, str]]:
    return {
        (path, method.upper())
        for path, path_item in spec["paths"].items()
        for method in path_item
        if method.upper() in HTTP_METHODS
    }


def _flask_operations() -> set[tuple[str, str]]:
    app = create_app(logging_setup_tuples=[])
    return {
        (rule.rule, method)
        for rule in app.url_map.iter_rules()
        for method in rule.methods
        if method in HTTP_METHODS
    }


def test_documented_openapi_operations_are_implemented_by_flask_routes():
    spec = yaml.safe_load(Path("api/concept-graphs-api.yml").read_text())
    flask_operations = _flask_operations()

    missing = []
    for documented_path, documented_method in _documented_operations(spec):
        documented_regex = _openapi_path_to_regex(documented_path)
        is_implemented = any(
            documented_method == flask_method
            and documented_regex.match(flask_rule.replace("<path_arg>", "example"))
            for flask_rule, flask_method in flask_operations
        )
        if not is_implemented:
            # Concrete OpenAPI paths such as /preprocessing/statistics are served
            # by generic Flask routes such as /preprocessing/<path_arg>.
            is_implemented = any(
                documented_method == flask_method
                and _flask_rule_to_regex(flask_rule).match(
                    documented_path.replace("{document_id}", "example")
                    .replace("{graph_id}", "0")
                    .replace("{process}", "example")
                )
                for flask_rule, flask_method in flask_operations
            )
        if not is_implemented:
            missing.append((documented_method, documented_path))

    assert missing == []


def test_business_flask_routes_are_documented_or_explicitly_ignored():
    spec = yaml.safe_load(Path("api/concept-graphs-api.yml").read_text())
    documented_operations = _documented_operations(spec)
    documented_paths = [path for path, _ in documented_operations]

    undocumented = []
    for flask_rule, flask_method in _flask_operations():
        if (flask_rule, flask_method) in IGNORED_STATIC_RULES:
            continue
        if (flask_rule, flask_method) in ALLOWED_UNDOCUMENTED_RULES:
            continue
        is_documented = any(
            flask_method == documented_method
            and _flask_rule_to_regex(flask_rule).match(
                documented_path.replace("{document_id}", "example")
                .replace("{graph_id}", "0")
                .replace("{process}", "example")
            )
            for documented_path, documented_method in documented_operations
        )
        if not is_documented:
            undocumented.append((flask_method, flask_rule, documented_paths))

    assert [(method, rule) for method, rule, _ in undocumented] == []
