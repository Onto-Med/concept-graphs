import logging
import sys
import pathlib
import dill as pickle
import pprint
from collections import defaultdict
from typing import Union, Iterable, Tuple

from sklearn.cluster import AgglomerativeClustering, KMeans
from itertools import product as product_iter

sys.path.insert(0, "../../src/")
import cluster_functions
import embedding_functions

logging.basicConfig()
logging.root.setLevel(logging.INFO)

metrics = {
    'KMeans': {
        'cluster_obj': KMeans,
        'kwargs': {},
        'kwargs_restr': {}
    },
    'Agglomerative': {
        'cluster_obj': AgglomerativeClustering,
        'kwargs': {
            'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
            'linkage': ['ward', 'complete', 'average', 'single']
        },
        'kwargs_restr': {
            ('linkage', 'ward',): ('affinity', 'euclidean',)
        }
    }
}


def metrics_generator():
    for _alg_name, _metr_dict in metrics.items():
        _kwargs_dict = _metr_dict['kwargs']
        if len(_kwargs_dict) == 0:
            yield _alg_name, _metr_dict['cluster_obj'], {}
            continue
        _kwargs_combs = [product_iter([_kwargs_key], _kwargs_value)
                         for _kwargs_key, _kwargs_value in _kwargs_dict.items()]
        for _comb in product_iter(*_kwargs_combs):
            _metrics = {c[0]: c[1] for c in _comb}
            if _metrics['linkage'] == 'ward' and not _metrics['affinity'] == 'euclidean':  # Todo: use kwargs_restr
                continue
            yield _alg_name, _metr_dict['cluster_obj'], _metrics


# ToDo: added feedback where we stand on loading
def load_embedding_clustering_factory(
        data_obj_path,
        embeddings_obj_path,
        cluster_obj_path,
        exclusion_ids,
        view_from_topics=None
) -> cluster_functions.WordEmbeddingClustering:
    sent_emb = embedding_functions.SentenceEmbeddingsFactory.load(
        data_obj_path=data_obj_path,
        embeddings_path=embeddings_obj_path,
        view_from_topics=view_from_topics
    )

    cluster_obj = cluster_functions.PhraseClusterFactory.load(
        data_obj_path=cluster_obj_path
    )

    embedding_clustering_factory = cluster_functions.WordEmbeddingClustering(
        sentence_embedding_obj=sent_emb,
        cluster_obj=cluster_obj.concept_cluster,
        cluster_exclusion_ids=exclusion_ids
    )

    return embedding_clustering_factory


# ToDo: see above
def create_clusterings(
        embedding_clustering_factory,
        algorithm='concept_graph',
        stop_words=None,
        **kwargs
) -> Tuple[Union[cluster_functions.WordEmbeddingClustering._WEClustering,
                 cluster_functions.WordEmbeddingClustering._ConceptGraphClustering], dict]:
    assert algorithm.lower() in {'concept_graph', 'we_clustering'}
    clustering = None
    _kwargs = {}
    if algorithm.lower() == 'we_clustering':
        clustering = embedding_clustering_factory.create_we_clustering(use_lemma=kwargs.get("use_lemma", False))
        _kwargs = {"stop_words": stop_words, "n_process": 4}
    elif algorithm.lower() == 'concept_graph':
        clustering = embedding_clustering_factory.create_concept_graph_clustering()
        # life
        # cluster_distance: 0.4 + cluster_min_size:4 + graph_cosine_weight: 0.95 + graph_merge_threshold:0.95
        # graph_weight_cut_off: 0.5 + graph_unroll:True + graph_simplify: 0.5 + use_exclusion_ids:True
        # graph_simplify_alg: significance + graph_sub_clustering:False + restrict_to_cluster: True + graph_distance_cutoff:0.5

        # grassco
        # cluster_distance:0.4+cluster_min_size:4+graph_cosine_weight:0.95+graph_merge_threshold:0.95
        # graph_weight_cut_off:0.6+graph_unroll:True+graph_simplify:0.5+use_exclusion_ids:True
        # graph_simplify_alg:significance+graph_sub_clustering:False+restrict_to_cluster:True+graph_distance_cutoff:0.5

        # 1000pa
        # cluster_distance:0.5+cluster_min_size:4+graph_cosine_weight:0.85+graph_merge_threshold:None
        # graph_weight_cut_off:0.5+graph_unroll:True+graph_simplify:0.5+use_exclusion_ids:True
        # graph_simplify_alg:significance+graph_sub_clustering:1.5+restrict_to_cluster:True+graph_distance_cutoff:0.75

        _kwargs = {
            "cluster_distance": .6,                 # for each cluster center found, how far away can the corresponding phrases be at max (cosine: 0 - 1; the higher the more restricitve)
            "cluster_min_size": 4,                  # how many phrases shall the cluster have at min
            "graph_cosine_weight": .85,             # how much influence shall the cosine distance between phrases have, when calculating edge weights for the graph (0 - 1)
            "graph_merge_threshold": .95,            # at what string similarity shall phrases be merged to one phrase (0 - 1; 1 is the same string)
            "graph_weight_cut_off": .6,             # below what weight values shall edges be removed between phrases (0 - 1)
            "graph_unroll": True,                   # whether the graph shall be transformed to tree-like
            "graph_simplify": .5,                   # proportion on how many edges shall be cut from the graph (0 - 1)
            "graph_simplify_alg": 'significance',   # which edge value to use for cutting (weight, significance)
            "graph_sub_clustering": True,           # for some algorithms denotes the reward when phrases are part of a sub cluster (> 1.0)
            "restrict_to_cluster": True,
            # "filter_min_df": 2,
            # "filter_max_df": .8,
            # "filter_stop":
            "graph_distance_cutoff": 2.0,            # (for alg3)
            "connection_distance": 8,              # (for alg2) when calculating
        }
        _kwargs.update(kwargs)
    else:
        pass
    return clustering, _kwargs


def calculate_scores(
        embedding_clustering_factory,
        clustering_alg_obj,
        metrics_iterator: Iterable
):
    min_concepts = 10  # ToDo: need to set this individually
    results_dict = defaultdict(dict)
    for alg_name, alg_impl, alg_metrics in metrics_iterator:
        _key = "None"
        if len(alg_metrics) != 0:
            _key = "-".join([v for _, v in sorted(alg_metrics.items(), key=lambda item: item[0])])
        _ari = embedding_clustering_factory.ari_score(embeddings_cluster_obj=clustering_alg_obj,
                                                      clustering_obj=alg_impl,
                                                      min_concepts=min_concepts,
                                                      **{f"cluster_{k}": v for k, v in alg_metrics.items()})

        _purity = embedding_clustering_factory.purity_score(embeddings_cluster_obj=clustering_alg_obj,
                                                            clustering_obj=alg_impl,
                                                            min_concepts=min_concepts,
                                                            **{f"cluster_{k}": v for k, v in alg_metrics.items()})
        results_dict[alg_name][_key] = {'ari': _ari, 'purity': _purity}
    return results_dict


if __name__ == "__main__":
    prefix = "articles-253" if len(sys.argv) <= 1 else sys.argv[1]
    suffix = None if len(sys.argv) <= 4 else sys.argv[4]

    _excl = []
    #_excl.extend([2, 3, 8, 10, 11, 12, 15, 17, 24])
    exclusion_ids = _excl if len(sys.argv) <= 2 else (
        [int(i.strip()) for i in sys.argv[2].split(",")] if sys.argv[2] != "None" else [])
    fix = {'concept_graph': False, 'we_clustering': True} if len(sys.argv) <= 3 else eval(sys.argv[3])

    if prefix.lower() == "medical":
        prefix = "schulz-life100part01"
    elif prefix.lower() == "1000pa":
        prefix = "confidential"

    cache_path = pathlib.Path("../../pickles/")

    data_obj_path = pathlib.Path(cache_path / pathlib.Path(f"{prefix}_data-processed{('_' + suffix) if suffix is not None else ''}"))
    embed_path = pathlib.Path(cache_path / pathlib.Path(f"{prefix}_phrase-embeddings{('_' + suffix) if suffix is not None else ''}"))
    cluster_obj_path = pathlib.Path(cache_path / pathlib.Path(f"{prefix}_phrase-cluster-obj{('_' + suffix) if suffix is not None else ''}"))

    logging.info(f"Loading Cluster Factory with exclusion ids {exclusion_ids}")
    factory = load_embedding_clustering_factory(data_obj_path, embed_path, cluster_obj_path, exclusion_ids)

    scores_dict = {}
    for alg in ['concept_graph', 'we_clustering']:
        alg_obj_path = pathlib.Path(cache_path / pathlib.Path(f"{alg}_{prefix}{('_' + suffix) if suffix is not None else ''}.pickle"))
        logging.info(f"Searching pickle for {alg_obj_path.resolve()} ...")
        if alg_obj_path.exists() and fix[alg]:
            logging.info(f"Loading Cluster Algorithm Pickle for {alg} ...")
            _impl = pickle.load(alg_obj_path.open('rb'))
        else:
            logging.info(f"Calculating for {alg} ...")
            _impl, _kwargs = create_clusterings(embedding_clustering_factory=factory, algorithm=alg)
            _impl.build_document_concept_matrix(**_kwargs)
            if fix[alg]:
                pickle.dump(_impl, alg_obj_path.open('wb'))
        scores_dict[alg] = calculate_scores(factory, _impl, metrics_generator())

    with (cache_path / pathlib.Path(f"{prefix}_scores_dict{('_' + suffix) if suffix is not None else ''}.pickle")).open(mode='wb') as dump:
        pickle.dump(scores_dict, dump)

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(scores_dict)
