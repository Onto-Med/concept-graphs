"""Download optional runtime NLP/tokenizer models.

This script is used by the runtime-download Docker image. Model downloads are
best-effort by default so a transient Hugging Face/spaCy download issue does not
prevent the API from starting. Set ``STRICT_MODEL_DOWNLOAD=true`` to fail the
container startup on model download errors.
"""

import logging
import os
from collections.abc import Callable
from typing import Any

import spacy
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

SPACY_MODELS = ["de_core_news_sm", "de_dep_news_trf"]
TOKENIZER_MODELS = ["gpt2", "dbmdz/german-gpt2"]


def _strict_model_download() -> bool:
    return os.getenv("STRICT_MODEL_DOWNLOAD", "false").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _run_download(name: str, loader: Callable, *args: Any, **kwargs: Any) -> bool:
    try:
        loader(*args, **kwargs)
        logger.info("Downloaded/verified model: %s", name)
        return True
    except Exception as exc:
        if _strict_model_download():
            raise
        logger.warning("Could not download/verify model '%s': %s", name, exc)
        return False


def download_spacy_models() -> None:
    for model in SPACY_MODELS:
        _run_download(model, spacy.cli.download, model)


def download_tokenizers() -> None:
    for model in TOKENIZER_MODELS:
        # Avoid force_download by default. It can leave a partially resolved cache in
        # some environments and produce a tokenizer init with a missing vocab path.
        if not _run_download(model, AutoTokenizer.from_pretrained, model):
            continue


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    download_spacy_models()
    download_tokenizers()


if __name__ == "__main__":
    main()
