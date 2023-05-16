import itertools
import logging
import random
import re
import sys
import time
import os
from pathlib import Path
from typing import Optional, Callable, Union
from collections import Counter, defaultdict

import pandas as pd
import spacy
import ir_datasets
from sklearn.datasets import fetch_20newsgroups

sys.path.insert(0, "../../src")
import cluster_functions
import data_functions
import embedding_functions
from util_functions import load_pickle

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def pairwise(iterable):
    _iter_list = list(iterable) if not isinstance(iterable, list) else iterable
    _len = len(_iter_list)

    return zip(_iter_list[:int(_len / 2)], _iter_list[-int(_len / 2):])


def data_processing_medical(cache_path: Path, name_prefix: str, labels: Union[dict, Callable], n_process: int = 1,
                            use_lemma: bool = False, prepend_root: bool = False, head_only: bool = False,
                            sub_path: Optional[list] = None, suffix: Optional[str] = None):
    data_functions.DataProcessingFactory.create(
        pipeline=spacy.load("de_dep_news_trf"),
        base_data=Path("../../data/"),
        labels=labels,
        sub_paths=["Schulz-Arztbriefe", "life-100-part01"] if sub_path is None else sub_path,
        cache_path=cache_path,
        cache_name=f"{name_prefix}_data-processed{('_' + suffix) if suffix is not None else ''}",
        n_process=n_process,
        use_lemma=use_lemma,
        prepend_head=prepend_root,
        head_only=head_only
    )


def data_processing_1000PA(cache_path: Path, name_prefix: str, n_process: int = 1,
                           use_lemma: bool = False, prepend_root: bool = False, head_only: bool = False,
                           suffix: Optional[str] = None, sample_size: Union[float, int] = 1.0, category_count: int = 20,
                           min_cat_num: int = 10):

    def _clean_1000pa(txt: str):
        _lines = []
        for line in txt.split("\n"):
            if line.strip().startswith("|"):
                continue
            _lines.append(re.sub(r"^\W*\s*", "", line))
        return "\n".join(_lines)

    def _get_label_1000pa(name: str):
        _ward = "OTHER"
        _parts = name.split("_")
        _abr_pos = -1
        for i in range(3):
            if _parts[i].lower() == "abr":
                _abr_pos = i
                break
        if _abr_pos != -1:
            if _abr_pos == 0:
                _ward = _parts[1]
            elif _abr_pos == 1:
                _ward = _parts[0]
        else:
            _ward = _parts[0]
        return _ward.upper()

    def _base_data(data_path):
        for d in data_path.glob("*.txt"):
            yield {"name": d.stem, "content": _clean_1000pa(d.read_text(encoding="utf-8")),
                   "label": _get_label_1000pa(d.stem)}

    _data = Path(Path("../../data/") / "confidential")
    _max_data = len(list(_data.glob("*.txt")))

    if isinstance(sample_size, float):
        _data_len = int(_max_data * min(sample_size, 1.0))
    elif isinstance(sample_size, int):
        _data_len = min(sample_size, _max_data)
    else:
        _data_len = _max_data

    _label_count = Counter(i['label'] for i in _base_data(_data))
    _max_labels = [i[0] for i in _label_count.items() if i[1] > min_cat_num]
    _cats = random.sample([i[0] for i in _label_count.items() if i[1] > min_cat_num],
                          min(category_count, len(_max_labels)))

    _sorted_cats = [x for x in sorted(_label_count.items(), key=lambda x: x[1], reverse=False) if x[0] in _cats]
    _points_per_cat = int(_data_len / len(_sorted_cats))

    _harmonized_data = []
    for _cat in _sorted_cats:
        _docs_for_cat = [d for d in _base_data(_data) if d['label'] == _cat[0]]
        _dp = min(_cat[1], _points_per_cat)
        if _dp == _docs_for_cat:
            _harmonized_data.extend(_docs_for_cat)
        else:
            _harmonized_data.extend(random.sample(_docs_for_cat, _dp))

    logging.info(f"Using {len(_harmonized_data)} data points")

    data_functions.DataProcessingFactory.create(
        pipeline=spacy.load("de_dep_news_trf"),
        base_data=_harmonized_data,
        # labels=labels,
        # sub_paths=["Schulz-Arztbriefe", "life-100-part01"] if sub_path is None else sub_path,
        cache_path=cache_path,
        cache_name=f"{name_prefix}_data-processed{('_' + suffix) if suffix is not None else ''}",
        n_process=n_process,
        use_lemma=use_lemma,
        prepend_head=prepend_root,
        head_only=head_only
    )


def data_processing_ng20(cache_path, name_prefix: str, categories: Optional[list] = None, subset: str = 'test',
                         n_process: int = 1, use_lemma: bool = False, prepend_root: bool = False,
                         head_only: bool = False,
                         clean: Optional[set] = None, sample_size: Union[float, int] = 1.0, random_state: int = 42):
    assert subset in ["train", "test", "all"]
    logging.info(f"Using subset {subset}")
    if clean is not None and isinstance(clean, set):
        assert clean.issubset(('headers', 'footers', 'quotes'))
    else:
        clean = set()
    logging.info(f"Cleaning {clean}")

    _data = fetch_20newsgroups(subset=subset, categories=categories, remove=clean, random_state=random_state)
    _max_data = len(_data.data)

    if isinstance(sample_size, float):
        _data_len = int(_max_data * min(sample_size, 1.0))
    elif isinstance(sample_size, int):
        _data_len = min(sample_size, _max_data)
    else:
        _data_len = _max_data
    logging.info(f"Using {_data_len} data points")

    _base_data = ({"name": n.split(os.sep)[-1], "content": d, "label": n.split(os.sep)[-2]}
                  for n, d in zip(_data.filenames, _data.data))
    data_functions.DataProcessingFactory.create(
        pipeline=spacy.load("en_core_web_trf"),
        # spacy.load("../../ext_models/en_core_web_trf-3.2.0/en_core_web_trf/en_core_web_trf-3.2.0/"),
        base_data=itertools.islice(_base_data, _data_len),
        cache_path=cache_path,
        cache_name=f"{name_prefix}_data-processed",
        n_process=n_process,
        use_lemma=use_lemma,
        prepend_head=prepend_root,
        head_only=head_only
    )


def data_processing_articles(cache_path: Path, name_prefix: str, labels: Union[dict, Callable], n_process: int = 1,
                             use_lemma: bool = False, prepend_root: bool = False, head_only: bool = False):
    data_functions.DataProcessingFactory.create(
        pipeline=spacy.load("en_core_web_trf"),
        base_data=Path("../../data/"),
        labels=labels,
        sub_paths=["articles-253"],
        cache_path=cache_path,
        cache_name=f"{name_prefix}_data-processed",
        n_process=n_process,
        use_lemma=use_lemma,
        prepend_head=prepend_root,
        head_only=head_only,
        filter_min_df=2,
        filter_max_df=.9,
        filter_stop=None
    )


def data_processing_classic4(cache_path: Path, name_prefix: str, labels: Union[dict, Callable], n_process: int = 1,
                             subset: Optional[int] = None, use_lemma: bool = False, prepend_root: bool = False,
                             head_only: bool = False):
    data_functions.DataProcessingFactory.create(
        pipeline=spacy.load("en_core_web_trf"),
        # spacy.load("../../ext_models/en_core_web_trf-3.2.0/en_core_web_trf/en_core_web_trf-3.2.0/"),
        base_data=Path("../../data/"),
        file_extension='*',
        labels=labels,
        sub_paths=["classic4"],
        cache_path=cache_path,
        cache_name=f"{name_prefix}_data-processed",
        n_process=n_process,
        subset=subset,
        use_lemma=use_lemma,
        prepend_head=prepend_root,
        head_only=head_only,
        filter_min_df=2,
        filter_max_df=.9,
        filter_stop=None
    )


def data_processing_scopus(cache_path: Path, name_prefix: str, labels: Union[dict, Callable], n_process: int = 1,
                           subset: Optional[int] = None, use_lemma: bool = False, prepend_root: bool = False,
                           head_only: bool = False, categories: Optional[list] = None):
    data_functions.DataProcessingFactory.create(
        pipeline=spacy.load("en_core_web_trf"),
        # spacy.load("../../ext_models/en_core_web_trf-3.2.0/en_core_web_trf/en_core_web_trf-3.2.0/"),
        base_data=Path("../../data/"),
        file_extension=None,
        labels=labels,
        sub_paths=["scopus2800"],
        cache_path=cache_path,
        cache_name=f"{name_prefix}_data-processed",
        n_process=n_process,
        subset=subset,
        use_lemma=use_lemma,
        prepend_head=prepend_root,
        head_only=head_only,
        filter_min_df=1,
        filter_max_df=1.,
        filter_stop=None,
        categories=categories
    )


def data_processing_ncbi(cache_path: Path, name_prefix: str, labels: Union[dict, Callable], n_process: int = 1,
                         subset: Optional[int] = None, use_lemma: bool = False, prepend_root: bool = False,
                         head_only: bool = False):
    data_functions.DataProcessingFactory.create(
        pipeline=spacy.load("en_core_web_trf"),
        # spacy.load("../../ext_models/en_core_web_trf-3.2.0/en_core_web_trf/en_core_web_trf-3.2.0/"),
        base_data=Path("../../data/"),
        file_extension=None,
        labels=labels,
        sub_paths=["ncbi_disease"],
        cache_path=cache_path,
        cache_name=f"{name_prefix}_data-processed",
        n_process=n_process,
        subset=subset,
        use_lemma=use_lemma,
        prepend_head=prepend_root,
        head_only=head_only,
        filter_min_df=2,
        filter_max_df=.9,
        filter_stop=None
    )


def data_processing_ohsumed(cache_path, name_prefix: str,
                            n_process: int = 1, use_lemma: bool = False, prepend_root: bool = False,
                            head_only: bool = False, sample_size: Union[float, int] = 1.0, random_state: int = 42):
    _data_path = Path("../../data/ohsumed").rglob("*")
    # _max_data = len(_data)
    #
    # if isinstance(sample_size, float):
    #     _data_len = int(_max_data * min(sample_size, 1.0))
    # elif isinstance(sample_size, int):
    #     _data_len = min(sample_size, _max_data)
    # else:
    #     _data_len = _max_data
    # logging.info(f"Using {_data_len} data points")

    _base_data = [{"name": p.name, "content": p.read_text(), "label": p.parent.name}
                  for p in _data_path if p.is_file()]
    _counter = Counter([_class['label'] for _class in _base_data])
    _least = _counter.most_common()[-1]

    _act_counter = defaultdict(int)
    _max_data = 50
    _harmonized_data = []
    for _b in _base_data:
        _act_counter[_b['label']] += 1
        if _act_counter[_b['label']] > _max_data:
            continue
        _harmonized_data.append(_b)

    data_functions.DataProcessingFactory.create(
        pipeline=spacy.load("en_core_web_trf"),
        # spacy.load("../../ext_models/en_core_web_trf-3.2.0/en_core_web_trf/en_core_web_trf-3.2.0/"),
        base_data=_harmonized_data,  # itertools.islice(_base_data, _data_len),
        cache_path=cache_path,
        cache_name=f"{name_prefix}_data-processed",
        n_process=n_process,
        use_lemma=use_lemma,
        prepend_head=prepend_root,
        head_only=head_only
    )


def data_processing_medline(cache_path, name_prefix: str,
                            n_process: int = 1, use_lemma: bool = False, prepend_root: bool = False,
                            head_only: bool = False, class_count: int = 15, merge_consecutive: bool = False,
                            random_state: int = 42):
    if merge_consecutive:
        _base_data = []
        for _cat in Path("../../data/medline_sample").glob("*"):
            _cat = _cat.name
            _files = Path(f"../../data/medline_sample/{_cat}").glob("*.txt")
            _base_data += [{"name": f"{p1.name}_{p2.name}", "content": f"{p1.read_text()} {p2.read_text()}",
                            "label": p1.parent.name} for p1, p2 in pairwise(_files)]
    else:
        _data_path = Path("../../data/medline_sample").rglob("*.txt")
        _base_data = [{"name": p.name, "content": p.read_text(), "label": p.parent.name}
                      for p in _data_path if p.is_file()]

    _classes = random.sample(list(set([c["label"] for c in _base_data])), k=class_count)

    data_functions.DataProcessingFactory.create(
        pipeline=spacy.load("en_core_web_trf"),
        # spacy.load("../../ext_models/en_core_web_trf-3.2.0/en_core_web_trf/en_core_web_trf-3.2.0/"),
        base_data=(d for d in _base_data if d["label"] in _classes),  # itertools.islice(_base_data, _data_len),
        cache_path=cache_path,
        cache_name=f"{name_prefix}_data-processed",
        n_process=n_process,
        use_lemma=use_lemma,
        prepend_head=prepend_root,
        head_only=head_only
    )


def data_processing_mtsamples(cache_path, name_prefix: str,
                              n_process: int = 1, use_lemma: bool = False, prepend_root: bool = False,
                              head_only: bool = False, class_count: int = 10, min_class_samples: int = 20,
                              sample_spread: int = 5, random_state: int = 42):
    _class_col = "medical_specialty"
    _text_col = "transcription"
    _base_classes = [' Surgery',
                     ' Cardiovascular / Pulmonary',
                     ' Orthopedic',
                     ' Radiology',
                     ' General Medicine',
                     ' Gastroenterology',
                     ' Neurology',
                     ' Obstetrics / Gynecology',
                     ' Urology',
                     ' ENT - Otolaryngology',
                     ' Neurosurgery',
                     ' Hematology - Oncology',
                     ' Ophthalmology',
                     ' Nephrology',
                     ' Pediatrics - Neonatal',
                     ' Pain Management',
                     ' Psychiatry / Psychology',
                     ' Podiatry',
                     ' Dermatology',
                     ' Cosmetic / Plastic Surgery',
                     ' Dentistry',
                     ' Physical Medicine - Rehab',
                     ' Sleep Medicine',
                     ' Endocrinology',
                     ' Bariatrics',
                     ' Chiropractic',
                     ' Rheumatology',
                     ' Diets and Nutritions',
                     ' Allergy / Immunology',
                     ' Hospice - Palliative Care']

    _data = pd.read_csv("../../data/mtsamples.csv")
    _classes = random.sample(_base_classes, k=class_count)
    _base_data = []
    for _c in _classes:
        _data_points = _data[_data[_class_col] == _c]
        _sample_data = _data_points.sample(
            min(min_class_samples + random.randint(0, sample_spread), len(_data_points.index)))
        for _, _row in _sample_data.iterrows():
            if not isinstance(_row[_text_col], str):
                continue
            _base_data.append({'name': str(_row[0]), 'content': _row[_text_col], 'label': _row[_class_col]})

    random.shuffle(_base_data)
    data_functions.DataProcessingFactory.create(
        pipeline=spacy.load("en_core_web_trf"),
        # spacy.load("../../ext_models/en_core_web_trf-3.2.0/en_core_web_trf/en_core_web_trf-3.2.0/"),
        base_data=(d for d in _base_data),  # itertools.islice(_base_data, _data_len),
        cache_path=cache_path,
        cache_name=f"{name_prefix}_data-processed",
        n_process=n_process,
        use_lemma=use_lemma,
        prepend_head=prepend_root,
        head_only=head_only,
        disable=['ner']
    )


def phrase_embedding_german(cache_path, name_prefix: str, down_scale: bool = False, n_process: int = 2,
                            suffix: Optional[str] = None):
    _scaling_kwargs = {"scaling_n_neighbors": 10, "scaling_min_dist": 0.1, "scaling_n_components": 100,
                       "scaling_metric": 'euclidean', "scaling_random_state": 42, }
    embedding_functions.SentenceEmbeddingsFactory.create(
        data_obj=data_functions.DataProcessingFactory.load(
            file_path=(cache_path / Path(f"{name_prefix}_data-processed{('_' + suffix) if suffix is not None else ''}"))
        ),
        cache_path=cache_path,
        cache_name=f"{name_prefix}_phrase-embeddings{('_' + suffix) if suffix is not None else ''}",
        model_name="Sahajtomar/German-semantic",
        # model_name="/home/fmatthies/Workspaces/polar-mmi/data/trained_models/german_semantic_medication",
        show_progress_bar=True,
        n_process=n_process,
        down_scale_algorithm='umap' if down_scale else None,
        **_scaling_kwargs
    )


def phrase_embedding_english(cache_path, name_prefix: str, view_from_topics=None, ref_data_str=None,
                             n_process: int = 1, down_scale: bool = False, model_name: Optional[str] = None):
    _scaling_kwargs = {"scaling_n_neighbors": 10, "scaling_min_dist": 0.1, "scaling_n_components": 100,
                       "scaling_metric": 'euclidean', "scaling_random_state": 42, }

    if ref_data_str is None:
        data_obj = data_functions.DataProcessingFactory.load(
            file_path=(cache_path / Path(f"{name_prefix}_data-processed"))
        )
    else:
        data_obj = data_functions.DataProcessingFactory.load(
            file_path=(cache_path / f"{ref_data_str}_data-processed")
        )
    embedding_functions.SentenceEmbeddingsFactory.create(
        data_obj=data_obj,
        cache_path=cache_path,
        cache_name=f"{name_prefix}_phrase-embeddings",
        # model_name="sentence-transformers/all-mpnet-base-v2",
        model_name="sentence-transformers/paraphrase-albert-small-v2" if model_name is None else model_name,
        # model_name="sentence-transformers/stsb-mpnet-base-v2",
        show_progress_bar=True,
        n_process=n_process,
        view_from_topics=view_from_topics,
        down_scale_algorithm='umap' if down_scale else None,
        **_scaling_kwargs
    )


def phrase_clustering_various(cache_path, name_prefix: str, view_from_topics=None, ref_data_str=None,
                              cluster_by_down_scale: bool = True, cluster_algorithm: str = "kmeans",
                              scaling_n_neighbors=10, scaling_min_dist=0.1, scaling_n_components=100,
                              suffix: Optional[str] = None, **kwargs):
    if ref_data_str is None:
        data_obj = cache_path / Path(f"{name_prefix}_data-processed{('_' + suffix) if suffix is not None else ''}")
    else:
        data_obj = cache_path / Path(f"{ref_data_str}_data-processed{('_' + suffix) if suffix is not None else ''}")
    cluster_functions.PhraseClusterFactory.create(
        sentence_embeddings=embedding_functions.SentenceEmbeddingsFactory.load(
            data_obj_path=data_obj,
            embeddings_path=(cache_path / Path(f"{name_prefix}_phrase-embeddings{('_' + suffix) if suffix is not None else ''}")),
            view_from_topics=view_from_topics
        ),
        cache_path=cache_path,
        cache_name=f"{name_prefix}_phrase-cluster-obj{('_' + suffix) if suffix is not None else ''}",
        cluster_algorithm=cluster_algorithm,
        scaling_n_neighbors=scaling_n_neighbors, scaling_min_dist=scaling_min_dist, scaling_n_components=scaling_n_components,
        scaling_metric='euclidean', scaling_random_state=42,
        kelbow_k=(10, 100), kelbow_show=False,
        cluster_by_down_scale=cluster_by_down_scale,
        **kwargs
    )


if __name__ == "__main__":
    _cache_path = Path("../../pickles/")
    _pipe = "mtsamples" if len(sys.argv) == 1 else sys.argv[1]
    _components = (['data-processing', 'phrase-embedding', 'clustering']
                   if len(sys.argv) <= 2 else [a.strip() for a in sys.argv[2].split(",")])

    if _pipe == "medical":
        _suffix = 'life'
        _name_prefix = "schulz-life100part01"
        med_labels = {k: v[0] for k, v in load_pickle(_cache_path / Path("file2class")).items()}
        if 'data-processing' in _components:
            data_processing_medical(cache_path=_cache_path, name_prefix=_name_prefix, labels=med_labels,
                                    use_lemma=False, prepend_root=False, n_process=4, head_only=False,
                                    sub_path=["life-100-part01"], suffix=_suffix)
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_german(_cache_path, name_prefix=_name_prefix, down_scale=False, n_process=4, suffix=_suffix)
        time.sleep(1.0)
        #scaling_n_neighbors=10, scaling_min_dist=0.1, scaling_n_components=100,
        #scaling_metric='euclidean', scaling_random_state=42,
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix, cluster_by_down_scale=True,
                                      scaling_n_neighbors=.05, suffix=_suffix)
            # cluster_algorithm="affinity-prop", cluster_damping=.9, cluster_max_iter=500)

    elif _pipe.lower() == "1000pa":
        _name_prefix = "confidential"
        _suffix = "10th_harmonized"

        if 'data-processing' in _components:
            data_processing_1000PA(cache_path=_cache_path, name_prefix=_name_prefix, suffix=_suffix,
                                   use_lemma=False, prepend_root=False, n_process=4, head_only=False,
                                   sample_size=.1)
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_german(_cache_path, name_prefix=_name_prefix, down_scale=False, n_process=4,
                                    suffix=_suffix)
        time.sleep(1.0)
        # scaling_n_neighbors=10, scaling_min_dist=0.1, scaling_n_components=100,
        # scaling_metric='euclidean', scaling_random_state=42,
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix, cluster_by_down_scale=True, suffix=_suffix)
            # cluster_algorithm="affinity-prop", cluster_damping=.9, cluster_max_iter=500)

    elif _pipe == "ng20-small":
        _name_prefix = "ng20-small"
        if 'data-processing' in _components:
            data_processing_ng20(_cache_path, name_prefix=_name_prefix, sample_size=700, n_process=4, random_state=123,
                                 subset='all', use_lemma=False, prepend_root=False, head_only=False,
                                 categories=["alt.atheism", "talk.religion.misc", "comp.graphics", "sci.space"])
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_english(_cache_path, name_prefix=_name_prefix)
        time.sleep(1.0)
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix)

    elif _pipe == "ng20-long":
        _name_prefix = "ng20-long"
        _topics = ["alt.atheism", "talk.religion.misc", "comp.graphics", "sci.space", "sci.med",
                   "rec.motorcycles", "rec.sport.hockey", "sci.electronics", "talk.politics.misc"]
        if 'data-processing' in _components:
            data_processing_ng20(_cache_path, name_prefix=_name_prefix, n_process=3, subset="all", categories=_topics)
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_english(_cache_path, name_prefix=_name_prefix, n_process=3)
        time.sleep(1.0)
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix)

    elif _pipe == "ng20-all":
        _name_prefix = "ng20-all"
        if 'data-processing' in _components:
            data_processing_ng20(_cache_path, name_prefix=_name_prefix, n_process=4, subset="all")
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_english(_cache_path, name_prefix=_name_prefix)
        time.sleep(1.0)
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix)

    elif _pipe == "articles-253":
        _name_prefix = "articles-253"
        if 'data-processing' in _components:
            data_processing_articles(_cache_path, name_prefix=_name_prefix, labels=lambda x: x.split("-")[0],
                                     n_process=4)
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_english(_cache_path, name_prefix=_name_prefix)
        time.sleep(1.0)
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix)

    elif _pipe == "classic4-long":
        _name_prefix = "classic4-long"
        if 'data-processing' in _components:
            data_processing_classic4(_cache_path, name_prefix=_name_prefix, labels=lambda x: x.split(".")[0],
                                     n_process=2)
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_english(_cache_path, name_prefix=_name_prefix)
        time.sleep(1.0)
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix)

    elif _pipe == "classic4":
        _name_prefix = "classic4"
        if 'data-processing' in _components:
            data_processing_classic4(_cache_path, name_prefix=_name_prefix, labels=lambda x: x.split(".")[0],
                                     n_process=4, subset=800, use_lemma=False, prepend_root=False, head_only=False)
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_english(_cache_path, name_prefix=_name_prefix, n_process=4)
        time.sleep(1.0)
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix)

    elif _pipe == "scopus-long":
        _name_prefix = "scopus-long"
        if 'data-processing' in _components:
            data_processing_scopus(_cache_path, name_prefix=_name_prefix, labels=lambda x: x.split("-")[0], n_process=4)
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_english(_cache_path, name_prefix=_name_prefix, n_process=2)
        time.sleep(1.0)
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix)

    elif _pipe == "scopus":
        _name_prefix = "scopus"
        if 'data-processing' in _components:
            data_processing_scopus(_cache_path, name_prefix=_name_prefix, labels=lambda x: x.split("-")[0],
                                   n_process=4, subset=500,
                                   categories=["concrete", "hyperactivity", "investment", "photosynthesis", "tectonicplates"])
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_english(_cache_path, name_prefix=_name_prefix, n_process=4)
        time.sleep(1.0)
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix)

    elif _pipe == "ncbi":
        _name_prefix = "ncbi"
        if 'data-processing' in _components:
            data_processing_ncbi(_cache_path, name_prefix=_name_prefix, labels=lambda x: x.split("_")[0],
                                 n_process=4)
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_english(_cache_path, name_prefix=_name_prefix, n_process=4)
        time.sleep(1.0)
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix)

    elif _pipe == "ohsumed":
        _name_prefix = "ohsumed"
        if 'data-processing' in _components:
            data_processing_ohsumed(_cache_path, name_prefix=_name_prefix, n_process=4)
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_english(_cache_path, name_prefix=_name_prefix, n_process=4)
        time.sleep(1.0)
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix)

    elif _pipe == "medline":
        _name_prefix = "medline"
        if 'data-processing' in _components:
            data_processing_medline(_cache_path, name_prefix=_name_prefix, n_process=4, class_count=15,
                                    merge_consecutive=False)
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_english(_cache_path, name_prefix=_name_prefix, n_process=4,
                                     model_name="FremyCompany/BioLORD-STAMB2-v1")
        time.sleep(1.0)
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix)

    elif _pipe == "mtsamples":
        _name_prefix = "mtsamples"
        if 'data-processing' in _components:
            data_processing_mtsamples(_cache_path, name_prefix=_name_prefix, n_process=4, class_count=15,
                                      min_class_samples=15, sample_spread=10, use_lemma=False, head_only=False)
        time.sleep(1.0)
        if 'phrase-embedding' in _components:
            phrase_embedding_english(_cache_path, name_prefix=_name_prefix, n_process=4,
                                     model_name="FremyCompany/BioLORD-STAMB2-v1")
        time.sleep(1.0)
        if 'clustering' in _components:
            phrase_clustering_various(_cache_path, name_prefix=_name_prefix, kelbow_metric='distortion')

    else:
        logging.warning(f"pipeline name '{_pipe}' not declared")
