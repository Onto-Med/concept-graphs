import io
import itertools
import pathlib
import pickle
from enum import IntEnum
from typing import Callable, Any, Optional, Union
from threading import Lock

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering


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


class HTTPResponses(IntEnum):
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    SERVICE_UNAVAILABLE = 503


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def load_pickle(
        file_path: Union[pathlib.Path, str, io.IOBase]
):
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path).absolute()
    elif isinstance(file_path, pathlib.Path):
        file_path = file_path.absolute()

    if not file_path.suffix == ".pickle":
        file_path = pathlib.Path(f"{file_path}.pickle")
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path.resolve()}")

    if not isinstance(file_path, io.IOBase):
        with file_path.open('rb') as f:
            return pickle.load(f)
    else:
        loaded_object = pickle.load(file_path)
        file_path.close()
        return loaded_object


def save_pickle(
        dump_obj: Any,
        file_path: pathlib.Path
):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.suffix == ".pickle":
        file_path = pathlib.Path(f"{file_path}.pickle")
    with file_path.open('wb') as f:
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
            print(f"'{_path}' does not exist; executing function '{run}' with given args & kwargs.")
        else:
            print(f"Choose to overwrite; executing function '{run}' with given args & kwargs.")
        _result = run(*args, **kwargs)
        with _path.open('wb') as f:
            pickle.dump(_result, f)
    else:
        print(f"Unpickling '{_path}' ...")
        with _path.open('rb') as f:
            _result = pickle.load(f)
    return _result


def cluster_purity(
        cluster_obj: Union['KMeans', 'AgglomerativeClustering'],
        targets: np.ndarray,
        print_df: bool = False
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


def pick_color(cname=None):
    cnames = {
        'aliceblue':            '#F0F8FF',
        'antiquewhite':         '#FAEBD7',
        'aqua':                 '#00FFFF',
        'aquamarine':           '#7FFFD4',
        'azure':                '#F0FFFF',
        'beige':                '#F5F5DC',
        'bisque':               '#FFE4C4',
        'black':                '#000000',
        'blanchedalmond':       '#FFEBCD',
        'blue':                 '#0000FF',
        'blueviolet':           '#8A2BE2',
        'brown':                '#A52A2A',
        'burlywood':            '#DEB887',
        'cadetblue':            '#5F9EA0',
        'chartreuse':           '#7FFF00',
        'chocolate':            '#D2691E',
        'coral':                '#FF7F50',
        'cornflowerblue':       '#6495ED',
        'cornsilk':             '#FFF8DC',
        'crimson':              '#DC143C',
        'cyan':                 '#00FFFF',
        'darkblue':             '#00008B',
        'darkcyan':             '#008B8B',
        'darkgoldenrod':        '#B8860B',
        'darkgray':             '#A9A9A9',
        'darkgreen':            '#006400',
        'darkkhaki':            '#BDB76B',
        'darkmagenta':          '#8B008B',
        'darkolivegreen':       '#556B2F',
        'darkorange':           '#FF8C00',
        'darkorchid':           '#9932CC',
        'darkred':              '#8B0000',
        'darksalmon':           '#E9967A',
        'darkseagreen':         '#8FBC8F',
        'darkslateblue':        '#483D8B',
        'darkslategray':        '#2F4F4F',
        'darkturquoise':        '#00CED1',
        'darkviolet':           '#9400D3',
        'deeppink':             '#FF1493',
        'deepskyblue':          '#00BFFF',
        'dimgray':              '#696969',
        'dodgerblue':           '#1E90FF',
        'firebrick':            '#B22222',
        'floralwhite':          '#FFFAF0',
        'forestgreen':          '#228B22',
        'fuchsia':              '#FF00FF',
        'gainsboro':            '#DCDCDC',
        'ghostwhite':           '#F8F8FF',
        'gold':                 '#FFD700',
        'goldenrod':            '#DAA520',
        'gray':                 '#808080',
        'green':                '#008000',
        'greenyellow':          '#ADFF2F',
        'honeydew':             '#F0FFF0',
        'hotpink':              '#FF69B4',
        'indianred':            '#CD5C5C',
        'indigo':               '#4B0082',
        'ivory':                '#FFFFF0',
        'khaki':                '#F0E68C',
        'lavender':             '#E6E6FA',
        'lavenderblush':        '#FFF0F5',
        'lawngreen':            '#7CFC00',
        'lemonchiffon':         '#FFFACD',
        'lightblue':            '#ADD8E6',
        'lightcoral':           '#F08080',
        'lightcyan':            '#E0FFFF',
        'lightgoldenrodyellow': '#FAFAD2',
        'lightgreen':           '#90EE90',
        'lightgray':            '#D3D3D3',
        'lightpink':            '#FFB6C1',
        'lightsalmon':          '#FFA07A',
        'lightseagreen':        '#20B2AA',
        'lightskyblue':         '#87CEFA',
        'lightslategray':       '#778899',
        'lightsteelblue':       '#B0C4DE',
        'lightyellow':          '#FFFFE0',
        'lime':                 '#00FF00',
        'limegreen':            '#32CD32',
        'linen':                '#FAF0E6',
        'magenta':              '#FF00FF',
        'maroon':               '#800000',
        'mediumaquamarine':     '#66CDAA',
        'mediumblue':           '#0000CD',
        'mediumorchid':         '#BA55D3',
        'mediumpurple':         '#9370DB',
        'mediumseagreen':       '#3CB371',
        'mediumslateblue':      '#7B68EE',
        'mediumspringgreen':    '#00FA9A',
        'mediumturquoise':      '#48D1CC',
        'mediumvioletred':      '#C71585',
        'midnightblue':         '#191970',
        'mintcream':            '#F5FFFA',
        'mistyrose':            '#FFE4E1',
        'moccasin':             '#FFE4B5',
        'navajowhite':          '#FFDEAD',
        'navy':                 '#000080',
        'oldlace':              '#FDF5E6',
        'olive':                '#808000',
        'olivedrab':            '#6B8E23',
        'orange':               '#FFA500',
        'orangered':            '#FF4500',
        'orchid':               '#DA70D6',
        'palegoldenrod':        '#EEE8AA',
        'palegreen':            '#98FB98',
        'paleturquoise':        '#AFEEEE',
        'palevioletred':        '#DB7093',
        'papayawhip':           '#FFEFD5',
        'peachpuff':            '#FFDAB9',
        'peru':                 '#CD853F',
        'pink':                 '#FFC0CB',
        'plum':                 '#DDA0DD',
        'powderblue':           '#B0E0E6',
        'purple':               '#800080',
        'red':                  '#FF0000',
        'rosybrown':            '#BC8F8F',
        'royalblue':            '#4169E1',
        'saddlebrown':          '#8B4513',
        'salmon':               '#FA8072',
        'sandybrown':           '#FAA460',
        'seagreen':             '#2E8B57',
        'seashell':             '#FFF5EE',
        'sienna':               '#A0522D',
        'silver':               '#C0C0C0',
        'skyblue':              '#87CEEB',
        'slateblue':            '#6A5ACD',
        'slategray':            '#708090',
        'snow':                 '#FFFAFA',
        'springgreen':          '#00FF7F',
        'steelblue':            '#4682B4',
        'tan':                  '#D2B48C',
        'teal':                 '#008080',
        'thistle':              '#D8BFD8',
        'tomato':               '#FF6347',
        'turquoise':            '#40E0D0',
        'violet':               '#EE82EE',
        'wheat':                '#F5DEB3',
        'white':                '#FFFFFF',
        'whitesmoke':           '#F5F5F5',
        'yellow':               '#FFFF00',
        'yellowgreen':          '#9ACD32'
    }
    if cname is not None:
        return cnames[cname]
    return cnames


class NoneDownScaleObj:
    def __init__(self, **kwargs):
        pass

    def __str__(self):
        return "None"
