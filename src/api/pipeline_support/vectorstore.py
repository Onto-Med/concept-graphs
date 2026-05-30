"""Vector-store configuration helpers for the pipeline route."""

import logging
from typing import Optional

from src.storage.marqo import MarqoEmbeddingStore


def normalize_vector_store_config(
    vector_store_config: Optional[dict],
) -> Optional[dict]:
    """Convert vector-store settings to client_url form and verify accessibility."""
    if vector_store_config is None:
        return None

    vector_store_config = dict(vector_store_config)
    url = vector_store_config.pop("url", "http://localhost")
    port = str(vector_store_config.pop("port", 8882))
    vector_store_config["client_url"] = f"{url}:{port}"

    if MarqoEmbeddingStore.is_accessible(vector_store_config.copy()):
        return vector_store_config

    logging.warning(
        "Vector store doesn't seem to be accessible under '%s'. Using 'pickle' storage.",
        vector_store_config["client_url"],
    )
    return None
