import abc
import enum
import io
import itertools
import json
import logging
import pathlib
import pickle
from collections import Counter, defaultdict
from threading import Lock
from typing import Callable, Any, Optional, Union, Iterable

# from cvae import cvae
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans, AgglomerativeClustering


class ConfigLoadMethods:
    @staticmethod
    def get(file_ending: str):
        return {
            "pickle": pickle.load,
            "pckl": pickle.load,
            "json": json.load,
            "jsn": json.load,
            "yaml": yaml.load,
            "yml": yaml.load,
        }.get(file_ending[1:] if file_ending[0] == "." else file_ending, json.load)


class ClusterNumberDetection(enum.Enum):
    DISTORTION = 1
    SILHOUETTE = 2
    CALINSKI_HARABASZ = 3


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    https://refactoring.guru/design-patterns/singleton/
    """

    _instances = {}

    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


# ToDo: this needs to be called whenever a data_proc object is used/loaded by another class
def set_spacy_extensions() -> None:
    from spacy.tokens import Doc

    if not Doc.has_extension("doc_id"):
        Doc.set_extension("doc_id", default=None)
    if not Doc.has_extension("doc_index"):
        Doc.set_extension("doc_index", default=None)
    if not Doc.has_extension("doc_name"):
        Doc.set_extension("doc_name", default=None)
    if not Doc.has_extension("doc_topic"):
        Doc.set_extension("doc_topic", default=None)
    if not Doc.has_extension("offset_in_doc"):
        Doc.set_extension("offset_in_doc", default=None)


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


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
        print(f"Saved under: {file_path.resolve()}")


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
            print("No Callable given for 'run' argument.")
            return
        if not _path.exists():
            print(
                f"'{_path}' does not exist; executing function '{run}' with given args & kwargs."
            )
        else:
            print(
                f"Choose to overwrite; executing function '{run}' with given args & kwargs."
            )
        _result = run(*args, **kwargs)
        with _path.open("wb") as f:
            pickle.dump(_result, f)
    else:
        print(f"Unpickling '{_path}' ...")
        with _path.open("rb") as f:
            _result = pickle.load(f)
    return _result


def cluster_purity(
    cluster_obj: Union["KMeans", "AgglomerativeClustering"],
    targets: np.ndarray,
    print_df: bool = False,
) -> float:
    counter = {}
    for c in range(cluster_obj.n_clusters):
        counter[c] = Counter()
        # if isinstance(targets, np.ndarray):
        for i in targets[np.where(cluster_obj.labels_ == c)]:
            counter[c].update({i: 1})

    df = pd.DataFrame.from_records([counter[i] for i in range(len(counter))])
    df.fillna(0, inplace=True)
    if print_df:
        print(df)
    return df.max(axis=1).to_numpy().sum() / df.to_numpy().sum()


def add_offset_to_documents_dicts_by_id(
    documents: list[dict], doc_id: str, offset: tuple[int, int]
):
    # ---> _docs = [{"id": doc_id, "offsets": [offset_of_nc_in_doc]}, ...]
    _added_offset = False
    for doc in documents:
        if doc_id == doc.get("id", None):
            doc.get("offsets", []).append(offset)
            _added_offset = True
            break
    if not _added_offset:
        documents.append({"id": doc_id, "offsets": [offset]})


def pick_color(cname=None):
    cnames = {
        "aliceblue": "#F0F8FF",
        "antiquewhite": "#FAEBD7",
        "aqua": "#00FFFF",
        "aquamarine": "#7FFFD4",
        "azure": "#F0FFFF",
        "beige": "#F5F5DC",
        "bisque": "#FFE4C4",
        "black": "#000000",
        "blanchedalmond": "#FFEBCD",
        "blue": "#0000FF",
        "blueviolet": "#8A2BE2",
        "brown": "#A52A2A",
        "burlywood": "#DEB887",
        "cadetblue": "#5F9EA0",
        "chartreuse": "#7FFF00",
        "chocolate": "#D2691E",
        "coral": "#FF7F50",
        "cornflowerblue": "#6495ED",
        "cornsilk": "#FFF8DC",
        "crimson": "#DC143C",
        "cyan": "#00FFFF",
        "darkblue": "#00008B",
        "darkcyan": "#008B8B",
        "darkgoldenrod": "#B8860B",
        "darkgray": "#A9A9A9",
        "darkgreen": "#006400",
        "darkkhaki": "#BDB76B",
        "darkmagenta": "#8B008B",
        "darkolivegreen": "#556B2F",
        "darkorange": "#FF8C00",
        "darkorchid": "#9932CC",
        "darkred": "#8B0000",
        "darksalmon": "#E9967A",
        "darkseagreen": "#8FBC8F",
        "darkslateblue": "#483D8B",
        "darkslategray": "#2F4F4F",
        "darkturquoise": "#00CED1",
        "darkviolet": "#9400D3",
        "deeppink": "#FF1493",
        "deepskyblue": "#00BFFF",
        "dimgray": "#696969",
        "dodgerblue": "#1E90FF",
        "firebrick": "#B22222",
        "floralwhite": "#FFFAF0",
        "forestgreen": "#228B22",
        "fuchsia": "#FF00FF",
        "gainsboro": "#DCDCDC",
        "ghostwhite": "#F8F8FF",
        "gold": "#FFD700",
        "goldenrod": "#DAA520",
        "gray": "#808080",
        "green": "#008000",
        "greenyellow": "#ADFF2F",
        "honeydew": "#F0FFF0",
        "hotpink": "#FF69B4",
        "indianred": "#CD5C5C",
        "indigo": "#4B0082",
        "ivory": "#FFFFF0",
        "khaki": "#F0E68C",
        "lavender": "#E6E6FA",
        "lavenderblush": "#FFF0F5",
        "lawngreen": "#7CFC00",
        "lemonchiffon": "#FFFACD",
        "lightblue": "#ADD8E6",
        "lightcoral": "#F08080",
        "lightcyan": "#E0FFFF",
        "lightgoldenrodyellow": "#FAFAD2",
        "lightgreen": "#90EE90",
        "lightgray": "#D3D3D3",
        "lightpink": "#FFB6C1",
        "lightsalmon": "#FFA07A",
        "lightseagreen": "#20B2AA",
        "lightskyblue": "#87CEFA",
        "lightslategray": "#778899",
        "lightsteelblue": "#B0C4DE",
        "lightyellow": "#FFFFE0",
        "lime": "#00FF00",
        "limegreen": "#32CD32",
        "linen": "#FAF0E6",
        "magenta": "#FF00FF",
        "maroon": "#800000",
        "mediumaquamarine": "#66CDAA",
        "mediumblue": "#0000CD",
        "mediumorchid": "#BA55D3",
        "mediumpurple": "#9370DB",
        "mediumseagreen": "#3CB371",
        "mediumslateblue": "#7B68EE",
        "mediumspringgreen": "#00FA9A",
        "mediumturquoise": "#48D1CC",
        "mediumvioletred": "#C71585",
        "midnightblue": "#191970",
        "mintcream": "#F5FFFA",
        "mistyrose": "#FFE4E1",
        "moccasin": "#FFE4B5",
        "navajowhite": "#FFDEAD",
        "navy": "#000080",
        "oldlace": "#FDF5E6",
        "olive": "#808000",
        "olivedrab": "#6B8E23",
        "orange": "#FFA500",
        "orangered": "#FF4500",
        "orchid": "#DA70D6",
        "palegoldenrod": "#EEE8AA",
        "palegreen": "#98FB98",
        "paleturquoise": "#AFEEEE",
        "palevioletred": "#DB7093",
        "papayawhip": "#FFEFD5",
        "peachpuff": "#FFDAB9",
        "peru": "#CD853F",
        "pink": "#FFC0CB",
        "plum": "#DDA0DD",
        "powderblue": "#B0E0E6",
        "purple": "#800080",
        "red": "#FF0000",
        "rosybrown": "#BC8F8F",
        "royalblue": "#4169E1",
        "saddlebrown": "#8B4513",
        "salmon": "#FA8072",
        "sandybrown": "#FAA460",
        "seagreen": "#2E8B57",
        "seashell": "#FFF5EE",
        "sienna": "#A0522D",
        "silver": "#C0C0C0",
        "skyblue": "#87CEEB",
        "slateblue": "#6A5ACD",
        "slategray": "#708090",
        "snow": "#FFFAFA",
        "springgreen": "#00FF7F",
        "steelblue": "#4682B4",
        "tan": "#D2B48C",
        "teal": "#008080",
        "thistle": "#D8BFD8",
        "tomato": "#FF6347",
        "turquoise": "#40E0D0",
        "violet": "#EE82EE",
        "wheat": "#F5DEB3",
        "white": "#FFFFFF",
        "whitesmoke": "#F5F5F5",
        "yellow": "#FFFF00",
        "yellowgreen": "#9ACD32",
    }
    if cname is not None:
        return cnames[cname]
    return cnames


class NoneDownScaleObj:
    def __init__(self, **kwargs):
        pass

    def __str__(self):
        return "None"


class Document(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def embeddings(self) -> list[Union[np.ndarray, list]]:
        raise NotImplementedError

    @abc.abstractmethod
    def phrases(self) -> list[str]:
        raise NotImplementedError


class EmbeddingStore(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def existing_from_config(cls, config: Union[dict, pathlib.Path, str]):
        """Instantiates an object from a config"""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def is_accessible(config: Union[dict, pathlib.Path, str]) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def store_embedding(
        self, embedding: Any, check_for_same: bool, **kwargs
    ) -> dict[str, set]:
        """Store the embedding and return its id in a dictionary with the following keys:
        'added', 'retained', 'added_idx', 'retained_idx'
        """
        raise NotImplementedError

    @abc.abstractmethod
    def store_embeddings(
        self,
        embeddings: Iterable,
        embeddings_repr: Iterable,
        vector_name: str,
        check_for_same: bool,
    ) -> dict[str, set]:
        """Store the embeddings and return their ids (as well as their indices) in a dictionary with the following keys:
        'added', 'retained', 'added_idx', 'retained_idx'"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_embedding(self, embedding_id: str):
        raise NotImplementedError

    @abc.abstractmethod
    def get_embeddings(self, embedding_ids: Optional[Iterable]):
        raise NotImplementedError

    @abc.abstractmethod
    def update_embedding(self, embedding_id: str, **kwargs) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def update_embeddings(
        self, embedding_ids: list[str], values: list[dict]
    ) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def delete_embedding(self, embedding_id: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def delete_embeddings(self, embedding_ids: Iterable) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def best_hits_for_field(self, embedding: Union[str, np.ndarray, list[float], dict]):
        raise NotImplementedError


class DocumentStore(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def add_document(
        self, document: Union[Document, tuple[Document, dict]], as_tuple: bool = False
    ) -> dict[str, dict[str, dict[str, list[str]]]]:
        """
        Adds a document to the store

        :param document: an instance of a ::class::`Document`
        :param as_tuple:
        :return: {
            "with_graph": {
                "added": [],
                "incorporated": []
            },
            "without_graph": {
                "added": [],
                "incorporated": [],
            }
        }
        """
        raise NotImplementedError

    @abc.abstractmethod
    def add_documents(
        self,
        document: Union[Iterable[Document], Iterable[tuple[Document, dict]]],
        as_tuple: bool = False,
    ) -> dict[str, dict[str, dict[str, dict[str, list[str]]]]]:
        """
        Adds documents from the Iterable to the store

        :param document: ::class::`Iterable` of ::class::`Document`
        :param as_tuple:
        :return: dictionary of results (see :func:`~DocumentStore.add_document`) keyed by document id (or if not give their index in the given iterable)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def suggest_graph_cluster(self, document: Document):
        raise NotImplementedError


def harmonic_mean(scores: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    _avg = lambda x: sum(x) / len(x)
    scores_by_class = defaultdict(list)
    for cls, score in scores:
        scores_by_class[cls].append(score)
    return sorted(
        [
            (
                cls,
                (
                    2 * _avg(l) * len(l) / (_avg(l) + len(l))
                    if (_avg(l) + len(l)) != 0
                    else 0
                ),
            )
            for cls, l in scores_by_class.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )


# class CVAEMantle:
#     def __init__(self, **kwargs):
#         self._cvae = None
#         self._extract_args(kwargs)
#
#     def _extract_args(self, kwargs):
#         _default_init_args = inspect.getfullargspec(cvae.CompressionVAE.__init__).args
#         _default_train_args = inspect.getfullargspec(cvae.CompressionVAE.train).args
#         _train_kwargs = {}
#         _init_kwargs = {}
#         for k, v in kwargs.items():
#             if k in _default_init_args:
#                 _init_kwargs[k] = v
#             if k in _default_train_args:
#                 _train_kwargs[k] = v
#         self._init_kwargs = _init_kwargs
#         self._train_kwargs = _train_kwargs
#
#     def fit(self, x: np.ndarray):
#         self._cvae = cvae.CompressionVAE(
#             x,
#             **self._init_kwargs
#         )
#         self._cvae.train(**self._train_kwargs)
#
#     def fit_transform(self, x: np.ndarray) -> np.ndarray:
#         self._cvae = cvae.CompressionVAE(
#             x,
#             **self._init_kwargs
#         )
#         self._cvae.train(**self._train_kwargs)
#         return self._cvae.X
#
#     def inverse_transform(self, x: np.ndarray) -> Optional[np.ndarray]:
#         if self._cvae is None:
#             return None
#         return self._cvae.decode(x)
#
#     def get_params(self):
#         return {'init': self._init_kwargs, 'train': self._train_kwargs}
