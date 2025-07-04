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
However, they are serialized as Python Objects and need to be loaded with:

1. processed documents: `src/data_functions/DataProcessingFactory.load(PATH/TO/DOCUMENT_OBJECT)`
2. phrase embeddings:
   `src/embedding_functions/SentenceEmbeddingsFactory.load(PATH/TO/DOCUMENT_OBJECT, PATH/TO/EMBEDDING_OBJECT)`
3. phrase cluster: `src/cluster_functions/PhraseClusterFactory.load(PATH/TO/CLUSTER_OBJECT)`
4. concept graphs: these are a list of serialized networkx [2] graphs

## Endpoints

### `/preprocessing`

upload text data to be preprocessed (i.e. extraction of phrases)

#### curl

`curl -X POST -F data=@"PATH/TO/DATA.zip" -F config=@"PATH/TO/CONFIG.yaml" -F labels=@"PATH/TO/LABELS.yaml" http://SOME_IP:SOME_PORT/preprocessing`

#### HTTP Requests

```
POST http://SOME_IP:SOME_PORT/preprocessing
Content-Type: multipart/form-data; boundary="boundary"

--boundary
Content-Disposition: form-data; name="data"; filename="DATA.zip"
Content-Type: application/zip

< PATH/TO/DATA.zip

--boundary
Content-Disposition: form-data; name="config"; filename="CONFIG.yaml"
Content-Type: application/x-yaml

< PATH/TO/CONFIG.yaml

--boundary
Content-Disposition: form-data; name="labels"; filename="LABELS.yaml"
Content-Type: application/x-yaml

< PATH/TO/LABELS.yaml
--boundary--
```

* `data`  (mandatory) : the text files provided as one zip file
* `config` (optional) : configurations for the preprocessing step provided as yaml file (if not provided, default values
  will be used)
* `labels` (optional) : the label (if any) of each data text file provided as yaml file (line-wise) (e.g. an entry looks
  like this: `file1: LABEL1`)

#### Path Parameters

* ``/preprocessing/statistics``: gets some basic statistics for the corpus
* ``/preprocessing/noun_chunks``: gets all noun chunks that were found (with the documents where)

#### Query Parameters

* ``process``: name of the process (e.g. ``corpus_name`` in config); if not provided, uses 'default'

### `/embedding`

embed the extracted phrases into a vector space

#### curl

`curl -X GET http://SOME_IP:SOME_PORT/embedding`  
or  
`curl -X POST -F config=@"PATH/TO/CONFIG.yaml" http://SOME_IP:SOME_PORT/embedding`

#### HTTP Requests

```
GET http://SOME_IP:SOME_PORT/embedding
```

or

```
POST http://SOME_IP:SOME_PORT/embedding
Content-Type: multipart/form-data; boundary="boundary"

--boundary
Content-Disposition: form-data; name="config"; filename="CONFIG.yaml"
Content-Type: application/x-yaml

< PATH/TO/CONFIG.yaml
--boundary--
```

* `config` (optional) : configurations for the embedding step provided as yaml file (if not provided, default values
  will be used)

The first time this endpoint is called, the respective model (as given in config or the default one) will be downloaded

#### Path Parameters

* ``/embedding/statistics``: gets some basic statistics for the embedding object

#### Query Parameters

* ``process``: name of the process (e.g. ``corpus_name`` in config); if not provided, uses 'default'

### `/clustering`

create the concept clusters from the embeddings which serve as the base for the concept graphs in the next step

#### curl

`curl -X GET http://SOME_IP:SOME_PORT/clustering`  
or  
`curl -X POST -F config=@"PATH/TO/CONFIG.yaml" http://SOME_IP:SOME_PORT/clustering`

#### HTTP Requests

```
GET http://SOME_IP:SOME_PORT/clustering
Content-Type: multipart/form-data; boundary="boundary"

--boundary
Content-Disposition: form-data; name="config"; filename="CONFIG.yaml"
Content-Type: application/x-yaml

< PATH/TO/CONFIG.yaml
--boundary--
```

* `config` (optional): configuration for the clustering step provided as yaml file (if not provided, default values will
  be used)

#### Path Parameters

* ``/clustering/concepts``: shows the concepts that were found

#### Query Parameters

* ``process``: name of the process (e.g. ``corpus_name`` in config); if not provided, uses 'default'
* ``top_k`` (only for `../concepts`): how many top k representatives of each cluster
* ``distance`` (only for `../concepts`): how far away (cosine) a representative can be

### `/graph`

**Attention**: the base endpoint has no funtion as such. To create the graphs you need to call the ``../creation`` path
argument
or call ``../statistics`` to get a simple overview of the graphs created for a specific ``process``.
Creates graph representations for each phrase cluster that was found during the 'clustering' step.  
You get a response with all graphs found and their respective edge/node count. To get a specific graph's adjacency
matrix use its id as path argument (the id is just the number given in the formerly mentioned response).

#### curl

`curl -X GET http://SOME_IP:SOME_PORT/graph/creation`  
or  
`curl -X POST -F config=@"PATH/TO/CONFIG.yaml" http://SOME_IP:SOME_PORT/graph/creation`

#### HTTP Requests

```
### POST with config
POST http://SOME_IP:SOME_PORT/graph/creation?exclusion_ids=[COMMA-SEPERATED LIST OF INTEGERS]
Content-Type: multipart/form-data; boundary="boundary"

--boundary
Content-Disposition: form-data; name="config"; filename="CONFIG.yaml"
Content-Type: application/x-yaml

< PATH/TO/CONFIG.yaml
--boundary--

### GET statistics
GET http://SOME_IP:SOME_PORT/graph/statistics

### GET specific graph
GET http://SOME_IP:SOME_PORT/graph/GRAPH_ID
```

* `config` (optional): configuration for the clustering step provided as yaml file (if not provided, default values will
  be used)

#### Path Parameters

* ``/graph/creation``: creates the graphs according to config
* ``/graph/statistics``: shows some basic statistics
* ``/graph/<GRAPH-ID>``: returns nodes and adjacency information for a specific graph

#### Query Parameters

* ``process``: name of the process (e.g. ``corpus_name`` in config); if not provided, uses 'default'
* ``exlusion_ids`` (only for creation): list of concept ids (as returned by e.g. ``/clustering/concepts``), that shall
  be excluded from the final graphs in the form of ``[ID1, ID2, etc.]``
* ``draw`` (only for querying a specific graph): `true` or `false` - if the response is a rendered graph or json

### `/pipeline`

starts a complete pipeline with all 4 sub steps.

* besides the configs for each subprocess (see further below) another `document_server_config` config might be provided
  to use documents from a server (e.g. elasticsearch instance) instead of uploading them.   
  If such a config is provided it overrides any uploaded documents.

```
url: BASE_URL to server (e.g.: 'http:\\localhost')
port: PORT (e.g.: '9008')
index: index in ES (e.g.: 'documents')
size: size of the document batch that is requested from the server
replace_keys: dictionary - 
    'key' is the servers response key for the actual text
    'value' is the concept-graphs-api internal name for the actual text content (as of now 'content')
    (e.g.: '{text: content}')
```

#### Path Parameters

* ``/pipeline/configuration``  
  (query params here: ``process`` (if a configuration from a specific pipeline is requested) &
  ``default`` (defaults to ``true`` so that the endpoint returns a default configuration; if set to ``false`` it can be
  combined with `process` to request a specific configuration))  
  _ToDo: proper format for this entry_

#### Query Parameters

* `process`: overrides the `corpus_name` given in the config
* `lang` (`de` or `en`): if declared here and no model provided in `config`, will use default language specific models
  for `preprocessing` and `clustering` (default: en)
* `skip_present`: (`true` or `false`) - whether to skip already saved steps for this particular process name (default:
  true)
* `skip_steps`: (comma separated list of either one or multiple of ``[data, embedding, clustering, graph]``) - pipeline
  steps that should be omitted if prerequisites are fulfilled; overrules ``skip_present``
* `return_statistics`: (`true` or `false`) - whether to wait for pipeline to finish and return statistics for the
  created graphs in the end or just silently start the pipeline (default: false)

``data``, ``labels`` and ``configs`` need to be provided like in the respective base endpoints except for the configs
need to be specified accordingly (if not provided, default values will be used):

* ``preprocessing``: ``config`` -> ``data_config``
* ``embedding``: ``config`` -> ``embedding_config``
* ``clustering``: ``config`` -> ``clustering_config``
* ``graph``: ``config`` -> ``graph_config``

(see example below)

_ToDo_: describe json config possibilities when header ``Content-Type:application/json``

#### HTTP Requests

```
POST http://SOME_IP:SOME_PORT/pipeline?process=default&lang=en&skip_present=true
Content-Type: multipart/form-data; boundary="boundary"

--boundary
Content-Disposition: form-data; name="data"; filename="DATA.zip"
Content-Type: application/zip

< PATH/TO/DATA.zip

--boundary
Content-Disposition: form-data; name="data_config"; filename="DATA_CONFIG.yaml"
Content-Type: application/x-yaml

< PATH/TO/DATA/CONFIG.yaml
--boundary
Content-Disposition: form-data; name="embedding_config"; filename="EMBEDDING_CONFIG.yaml"
Content-Type: application/x-yaml

< PATH/TO/EMBEDDING/CONFIG.yaml
--boundary
Content-Disposition: form-data; name="clustering_config"; filename="CLUSTERING_CONFIG.yaml"
Content-Type: application/x-yaml

< PATH/TO/CLUSTERING/CONFIG.yaml
--boundary
Content-Disposition: form-data; name="graph_config"; filename="GRAPH_CONFIG.yaml"
Content-Type: application/x-yaml

< PATH/TO/GRAPH/CONFIG.yaml
--boundary--
```

or without data upload, but rather document server:

```
POST http://SOME_IP:SOME_PORT/pipeline?process=default&lang=en&skip_present=true
Content-Type: multipart/form-data; boundary="boundary"

--boundary
Content-Disposition: form-data; name="document_server_config"; filename="DOCUMENT_SERVER_CONFIG.yaml"
Content-Type: application/x-yaml

< PATH/TO/DOCUMENT_SERVER/CONFIG.yaml

[...]

--boundary
Content-Disposition: form-data; name="graph_config"; filename="GRAPH_CONFIG.yaml"
Content-Type: application/x-yaml

< PATH/TO/GRAPH/CONFIG.yaml
--boundary--
```

### `/processes`

Gets the name of all stored processes.

#### HTTP Requests

```
GET http://SOME_IP:SOME_PORT/processes
```

### `/status`

Gets status of a specific process

#### Query Parameters

* ``process``: name of the process (e.g. ``corpus_name`` in config); if not provided, uses 'default'

#### HTTP Requests

```
GET http://SOME_IP:SOME_PORT/status?process=PROCESS_NAME
```

### `/status/document-server`

Checks if a data server (specified with a `document_server_config` file) is reachable

#### HTTP Requests

```
POST http://SOME_IP:SOME_PORT/document-server
Content-Type: multipart/form-data; boundary="boundary"

--boundary
Content-Disposition: form-data; name="document_server_config"; filename="DOCUMENT_SERVER_CONFIG.yaml"
Content-Type: application/x-yaml

< PATH/TO/DOCUMENT_SERVER/CONFIG.yaml

--boundary--
```

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
