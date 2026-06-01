"""Small parsing and normalization helpers."""

import re


def get_bool_expression(str_bool: str, default: bool | str = False) -> bool:
    if isinstance(str_bool, bool):
        return str_bool
    elif isinstance(str_bool, str):
        return {
            "true": True,
            "yes": True,
            "y": True,
            "ja": True,
            "j": True,
            "false": False,
            "no": False,
            "n": False,
            "nein": False,
        }.get(str_bool.lower(), default)
    else:
        return False


def string_conformity(s: str):
    if s is None:
        return None
    return re.sub(r"\s+|-+", "_", s.lower())
