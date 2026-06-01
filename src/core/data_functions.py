"""Compatibility exports for data preprocessing helpers.

New code should import from ``src.core.data`` or its focused submodules.
"""

from src.core.data import (
    DataProcessingFactory,
    clean_span,
    get_actual_str,
    validate_negspacy_config,
)

__all__ = [
    "DataProcessingFactory",
    "clean_span",
    "get_actual_str",
    "validate_negspacy_config",
]
