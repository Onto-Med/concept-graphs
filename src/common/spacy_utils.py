"""spaCy model loading helpers."""

import logging
from dataclasses import dataclass

import spacy
from dataclass_wizard import JSONWizard
from waiting import TimeoutExpired, wait


@dataclass
class NegspacyConfig(JSONWizard):
    chunk_prefix: str | list[str] | None = None
    neg_termset_file: str | None = None
    scope: int | None = None
    language: str | None = None
    feat_of_interest: str | None = None


def load_spacy_model(spacy_model: str, logger: logging.Logger, default_model: str):
    def is_valid_spacy_model(model: str):
        from spacy.cli.download import get_compatibility

        if model in get_compatibility():
            return True
        logger.error(f"'{model}' is not a valid model name.")
        return False

    def wait_for_download(model: str, time_out: int = 30):
        spacy.cli.download(model)

        def wait_pred():
            return model in spacy.util.get_installed_models()

        try:
            wait(wait_pred, timeout_seconds=time_out)
        except TimeoutExpired:
            logger.warning(
                f"TimeOut while waiting >{time_out} seconds for download to finish."
                f" Hopefully this is just due to installed models not refreshing."
            )

    spacy_language = None
    try:
        spacy_language = spacy.load(spacy_model)
    except OSError as e:
        if spacy_model != default_model:
            if is_valid_spacy_model(spacy_model):
                logger.info(
                    f"Model '{spacy_model}' doesn't seem to be installed; trying to download model."
                )
                wait_for_download(spacy_model)
                spacy_language = spacy.load(spacy_model)
            else:
                logger.error(f"{e}\nUsing default model {default_model}.")
                try:
                    spacy_language = spacy.load(default_model)
                except OSError as e:
                    logger.error(
                        f"{e}\ntrying to download default model {default_model}."
                    )
                    wait_for_download(default_model)
                    spacy_language = spacy.load(default_model)
        else:
            logger.error(f"{e}\ntrying to download default model {default_model}.")
            wait_for_download(default_model)
            spacy_language = spacy.load(default_model)
    return spacy_language


def get_default_spacy_model():
    return "en_core_web_trf"
