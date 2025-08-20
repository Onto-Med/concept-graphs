[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Concept Graphs

This is the implementation as described in [1].
WARNING: doesn't work for small number of phrases. You need a document corpus with at least 100 (?) different phrases

## Docker Image

1. `docker compose build` (the resulting image is appr. 5GB)
2. `docker compose up -d`

If you start the container as described, the address for the `curl` command would be `http://localhost:9007`.
All results (processed documents, the embeddings, etc.) are stored in the Docker volume `results` (mounted to
`/rest_api/tmp` in the container).
However, they are serialized as Python Objects and need (if necessary) to be loaded with the `FactoryLoader` class:

1. processed documents: `load_utils/FactoryLoader.load_data(PATH/TO/OBJECT/FOLDER, PROCESS_NAME)`
2. phrase embeddings: `load_utils/FactoryLoader.load_embedding(PATH/TO/OBJECT/FOLDER, PROCESS_NAME)`
3. phrase cluster: `load_utils/FactoryLoader.load_clustering(PATH/TO/OBJECT/FOLDER, PROCESS_NAME)`
4. concept graphs: `load_utils/FactoryLoader.load_graph(PATH/TO/OBJECT/FOLDER, PROCESS_NAME)` _(serialized networkx [2] graphs)_

## Endpoints

See concept-graphs-api.yml for the OpenAPI specification. When the service is started, on can reach the Swagger UI from `localhost:9007`.

## Pipeline

_Write out general functioning..._

## Example Config Files (YAML)

### `/preprocessing`

```
# Name of the corpus; can be chosen freely (but acts as reference point between the different endpoint actions)
corpus_name: default
# Name of the spaCy model; pre-installed: de_dep_news_trf, de_core_news_sm (German) & en_core_web_trf (English) [default]
spacy_model: en_core_web_trf
# Number of processes that will be spawned (how many cores will be utilized)
n_process: 1
# Only files in the data zip will be processed that have this file extension
file_extension: txt
# Encoding of the text files
file_encoding: utf-8
# Lemmatize the input data
use_lemma: False
# Prepend the head of a phrase to its beginning
prepend_head: False
# Use only the head of a phrase
head_only: False
# Discriminate between upper and lowercase tokens even if it's the same word
case_sensitive: False
# If a tf-idf pre-filter for phrases should be used; 1, 1.0, None takes phrases as is; see e.g. scikit-learn's tf-idf vectorizer for a description
# (int or float)
filter_min_df: 1
# (int or float)
filter_max_df: 1.0
# (None or name of POSTed stopwords file)
filter_stop: None
# Name of spaCy pipeline components to disable (None or array)
disable: None
# Configuration of NegSpacy (heuristic negation detection)
#   - chunk_prefix: str (path to file) or array
#   - neg_termset_file: str (path to file)
#   - scope: int or bool --> if >=1 it restricts the search for negated candidates to this in terms of distance in dependency parse
#   - feat_of_interest: string (NE | NC | BOTH) --> negation checked on which feature of interest: named entities (NE), noun chunks (NC), or both
#   - language: str
negspacy:
  - enabled: true
  - configuration:
    - chunk_prefix:
        - kein
        - keine
        - keinen
      neg_termset_file: ./conf/negex_files/negex_trigger_german_biotxtm_2016_extended.txt
      scope: 1
      language: de
      feat_of_interest: NC
```

### `/embedding`

```
# Name of the corpus; can be chosen freely (but acts as reference point between the different endpoint actions)
corpus_name: default
# (for German: Sahajtomar/German-semantic; English: sentence-transformers/paraphrase-albert-small-v2 [default])
model: sentence-transformers/paraphrase-albert-small-v2
# Number of processes that will be spawned (how many cores will be utilized)
n_process: 1
```

### `/clustering`

```
# Name of the corpus; can be chosen freely (but acts as reference point between the different endpoint actions)
corpus_name: default
# The clustering algorithm to be used. One of {kmeans (default), kmeans-mb, affinity-prop}
algorithm: kmeans
# The downscale algorithm to be used (to effectively cluster the high-dimensional embeddings); only "umap" supported.
downscale: umap
# Whether the recommended settings for 'clustering', 'downscaling' & 'cluster number deduction' shall be used if nothing else is stated
missing_as_recommended: True
# Arguments for deducing the number of clusters prefixed with 'deduction_'
deduction_enabled: true
deduction_k_min: 2
deduction_k_max: 100
deduction_n_samples: 15
deduction_sample_fraction: 25
deduction_regression_poly_degree: 5
# UMAP arguments prefixed with 'scaling_' (see umap-learn.readthedocs.io)
scaling_*
# Clustering arguments prefixed with 'clustering_' (dependent on algorithm used: see scikit-learn)
clustering_*
```

### `/graph/creation`

```
# Name of the corpus; can be chosen freely (but acts as reference point between the different endpoint actions)
corpus_name: default
# Phrases that are farther away (in cosinus) from the cluster center than this won't be used for graph creation
cluster_distance: 0.6
# A cluster needs to have at least this many phrases or else the whole cluster will be skipped
cluster_min_size: 1
# How important should the cosinus distance be when connecting phrases 
graph_cosine_weight: .5
# At which weight threshold between two phrase they will be merged into one
graph_merge_threshold: .95
# Edges where the weight is smaller than this value are cut
graph_weight_cut_off: .5
# Whether the graph will be transformed to treelike (each node has max one incoming and one outgoing edge) 
graph_unroll: True
# What proportion of edges shall be trimmed
graph_simplify: .5
# Shall the importance of an edge (when trimming) be measured by 'weight' or by 'significance' [3]
graph_simplify_alg: significance
# If true, sub clusters in a cluster will be formed that might be used downstream
graph_sub_clustering: False
# Whether phrases adhere strictly to their assigned cluster even if they might be nearer to another cluster center
restrict_to_cluster: True
# (Deprecated?)
filter_min_df: 1
filter_max_df: 1.
filter_stop: None
```

## References

**[1]** Matthies, F et al. *Concept Graphs: A Novel Approach for Textual Analysis of Medical Documents.* In: Röhrig, R
et al., editors. Studies in Health Technology and Informatics. IOS Press; 2023. Available
from: https://ebooks.iospress.nl/doi/10.3233/SHTI230710  
**[2]** https://networkx.org/  
**[3]** Dianati N. *Unwinding the hairball graph: Pruning algorithms for weighted complex networks.* Phys Rev E.
2016;93(1). Available from: https://link.aps.org/doi/10.1103/PhysRevE.93.012304  
