"""Metric and scoring helpers used by core algorithms."""

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans

logger = logging.getLogger(__name__)


def cluster_purity(
    cluster_obj: KMeans | AgglomerativeClustering,
    targets: np.ndarray,
    print_df: bool = False,
) -> float:
    counter = {}
    for c in range(cluster_obj.n_clusters):
        counter[c] = Counter()
        for i in targets[np.where(cluster_obj.labels_ == c)]:
            counter[c].update({i: 1})

    df = pd.DataFrame.from_records([counter[i] for i in range(len(counter))])
    df.fillna(0, inplace=True)
    if print_df:
        logger.info("%s", df)
    return df.max(axis=1).to_numpy().sum() / df.to_numpy().sum()


def harmonic_mean(scores: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    def _avg(values):
        return sum(values) / len(values)

    scores_by_class = defaultdict(list)
    for cls, score in scores:
        scores_by_class[cls].append(score)
    return sorted(
        [
            (
                cls,
                (
                    2
                    * _avg(class_scores)
                    * len(class_scores)
                    / (_avg(class_scores) + len(class_scores))
                    if (_avg(class_scores) + len(class_scores)) != 0
                    else 0
                ),
            )
            for cls, class_scores in scores_by_class.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )
