"""Configuration file loading helpers."""

import json
import pickle
from collections.abc import Callable
from typing import Any

import yaml


class ConfigLoadMethods:
    """Map config file suffixes to loader functions."""

    @staticmethod
    def get(file_ending: str) -> Callable[..., Any]:
        suffix = file_ending[1:] if file_ending.startswith(".") else file_ending
        return {
            "pickle": pickle.load,
            "pckl": pickle.load,
            "json": json.load,
            "jsn": json.load,
            "yaml": yaml.safe_load,
            "yml": yaml.safe_load,
        }.get(suffix, json.load)
