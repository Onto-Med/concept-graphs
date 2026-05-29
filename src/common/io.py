"""Pickle-based persistence helpers."""

import io
import logging
import pathlib
import pickle
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


def load_pickle(
    file_path: Union[pathlib.Path, str, io.IOBase], logger: logging.Logger = None
) -> Any:
    if logger is None:
        logger = logging.getLogger()
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path).absolute()
    elif isinstance(file_path, pathlib.Path):
        file_path = file_path.absolute()

    if not file_path.suffix == ".pickle":
        file_path = pathlib.Path(f"{file_path}.pickle")
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path.resolve()}")

    if not isinstance(file_path, io.IOBase):
        with file_path.open("rb") as f:
            try:
                return pickle.load(f)
            except EOFError as e:
                logger.error(f"Unable to load {file_path}: {e}")
                return None
    else:
        loaded_object = pickle.load(file_path)
        file_path.close()
        return loaded_object


def save_pickle(dump_obj: Any, file_path: pathlib.Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.suffix == ".pickle":
        file_path = pathlib.Path(f"{file_path}.pickle")
    with file_path.open("wb") as f:
        pickle.dump(dump_obj, f)
        logger.info("Saved pickle under: %s", file_path.resolve())


def unpickle_or_run(
    base_path: pathlib.Path,
    filename: str,
    overwrite: bool = False,
    run: Optional[Callable] = None,
    args: Optional[list] = None,
    kwargs: Optional[dict] = None,
) -> Any:
    args = [] if args is None else args
    kwargs = {} if kwargs is None else kwargs
    _path = pathlib.Path(base_path / f"{filename}.pickle")
    if overwrite or (not _path.exists()):
        if run is None:
            logger.warning("No Callable given for 'run' argument.")
            return None
        if not _path.exists():
            logger.info(
                "'%s' does not exist; executing function '%s' with given args & kwargs.",
                _path,
                run,
            )
        else:
            logger.info(
                "Chose to overwrite; executing function '%s' with given args & kwargs.",
                run,
            )
        _result = run(*args, **kwargs)
        with _path.open("wb") as f:
            pickle.dump(_result, f)
    else:
        logger.info("Unpickling '%s' ...", _path)
        with _path.open("rb") as f:
            _result = pickle.load(f)
    return _result
