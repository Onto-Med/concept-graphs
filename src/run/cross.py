import itertools
import logging
import pathlib
import pickle
import sys
from math import prod


sys.path.insert(0, "../run/")
from test import (
    load_embedding_clustering_factory,
    create_clusterings,
    calculate_scores,
    metrics_generator,
)

if __name__ == "__main__":
    params_dict = {
        # "cluster_distance": [.45, .5, .55],
        "cluster_distance": [0.5, 0.6, 0.7],
        # "cluster_distance": [.5, .7],
        # "cluster_distance": [.6, .7],
        # "cluster_distance": [.4],
        "cluster_min_size": [4],
        # "cluster_min_size": [2, 3],
        # "cluster_min_size": [5],
        # "graph_cosine_weight": [.5, .6, .7],
        # "graph_cosine_weight": [.5, .7, .9],
        # "graph_cosine_weight": [.5, .9],
        "graph_cosine_weight": [0.85, 0.95, 1.0],
        # "graph_merge_threshold": [.75, .8],
        # "graph_merge_threshold": [.8, .85, .9],
        # "graph_merge_threshold": [.85, .9, .95],
        "graph_merge_threshold": [0.85, 0.95, None],
        # "graph_weight_cut_off": [.4, .45, .5],
        # "graph_weight_cut_off": [.5, .6, .7],
        # "graph_weight_cut_off": [.6, .7],
        "graph_weight_cut_off": [0.5, 0.6, 0.7],
        # "graph_weight_cut_off": [.5],
        # "graph_unroll": [True],
        "graph_unroll": [True, False],
        "graph_simplify": [0.5],
        # "graph_simplify": [.5, .7],
        # "graph_distance_cutoff": [.4, .5, .6, .7],
        # "connection_distance": [2, 3, 4, 5],
        # "connection_distance": [4, 6, 8, 10],
        # "connection_distance": [8, 10, 12, 14],
        # "use_exclusion_ids": [False],
        "use_exclusion_ids": [True],
        # "use_exclusion_ids": [False, True],
        "graph_simplify_alg": ["significance"],
        # "graph_sub_clustering": [False, 1.5, 1.75, 2.0],
        # "graph_sub_clustering": [False, 1.5, 2.0],
        "graph_sub_clustering": [False],
        # "restrict_to_cluster": [True, False]
        "restrict_to_cluster": [True],
        # "graph_distance_cutoff": [.5, .75],
        "graph_distance_cutoff": [0.75],
    }

    prefix = "ng20-small" if len(sys.argv) <= 1 else sys.argv[1]
    exclusion_ids = (
        [] if len(sys.argv) <= 2 else [int(i) for i in sys.argv[2].split(",")]
    )
    suffix = None if len(sys.argv) <= 3 else sys.argv[3]

    if prefix.lower() == "medical":
        prefix = "schulz-life100part01"
    elif prefix.lower() == "1000pa":
        prefix = "confidential"

    cache_path = pathlib.Path("../../pickles/")

    logging.info(
        f"Loading data, embedding & cluster objects for '{prefix}{('_' + suffix) if suffix is not None else ''}' and exclusion ids '{exclusion_ids}' ..."
    )
    data_obj_path = pathlib.Path(
        cache_path
        / pathlib.Path(
            f"{prefix}_data-processed{('_' + suffix) if suffix is not None else ''}"
        )
    )
    embed_path = pathlib.Path(
        cache_path
        / pathlib.Path(
            f"{prefix}_phrase-embeddings{('_' + suffix) if suffix is not None else ''}"
        )
    )
    cluster_obj_path = pathlib.Path(
        cache_path
        / pathlib.Path(
            f"{prefix}_phrase-cluster-obj{('_' + suffix) if suffix is not None else ''}"
        )
    )

    factory = load_embedding_clustering_factory(
        data_obj_path, embed_path, cluster_obj_path, exclusion_ids
    )
    concept_clustering, _ = create_clusterings(
        embedding_clustering_factory=factory, algorithm="concept_graph"
    )

    scores_dict = {}

    total_iter = prod([len(i) for i in params_dict.values()])

    logging.info("Start parameter grid search ...")
    _params_name, _params_value = zip(*params_dict.items())
    for i, _params in enumerate(itertools.product(*_params_value)):
        _params_kwargs = {k: v for k, v in zip(_params_name, _params)}
        _key_string = "+".join(
            [f"{str(i)}:{str(j)}" for i, j in zip(_params_name, _params)]
        )
        print(f"({i}/{total_iter}) Calculating scores for:\n{_key_string}\n")

        if not _params_kwargs["use_exclusion_ids"]:
            _params_kwargs["cluster_exclusion_ids"] = []
        _params_kwargs.pop("use_exclusion_ids")
        concept_clustering.build_document_concept_matrix(**_params_kwargs)
        scores_dict[_key_string] = calculate_scores(
            factory, concept_clustering, metrics_generator()
        )

    pickle.dump(
        scores_dict,
        (
            cache_path
            / f"cross-concept_graph-{prefix}-scores{('_' + suffix) if suffix is not None else ''}.pickle"
        ).open("wb"),
    )
    # pickle.dump(scores_dict, pathlib.Path(f"C:/Users/fra3066mat/Nextcloud/transfer/cross-concept_graph-{prefix}-scores.pickle").open('wb'))
