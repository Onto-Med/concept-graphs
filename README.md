# Concept Graphs

## Docker Image
1. `docker image build -t concept-graphs-api .` (the resulting image is appr. 12GB)
2. `docker run -p 9007:9007 concept-graphs-api`

If you start the container as described, the address for the `curl` command would be `http://0.0.0.0:9007`.  
Right now, the path where all results (the processed documents, the embeddings, etc.) are stored is under `/rest_api/concept-graphs/tmp` (if you want to access them via `docker volumes`).
However, they are serialized as Python Objects and need to be loaded with:
1. processed documents: `src/data_functions/DataProcessingFactory.load(PATH/TO/DOCUMENT_OBJECT)`
2. phrase embeddings: `src/embedding_functions/SentenceEmbeddingsFactory.load(PATH/TO/DOCUMENT_OBJECT, PATH/TO/EMBEDDING_OBJECT)`
3. phrase cluster: `src/cluster_functions/PhraseClusterFactory.load(PATH/TO/CLUSTER_OBJECT)`

## Endpoints
### `/preprocessing`
upload text data to be preprocessed (i.e. extraction of phrases)  

`curl -X POST -F data=@"PATH/TO/DATA.zip" -F config=@"PATH/TO/CONFIG.yaml" -F labels=@"PATH/TO/LABELS.yaml" http://SOME_IP:SOME_PORT/preprocessing`  

* `data`  (mandatory) : the text files provided as one zip file
* `config` (optional) : configurations for the preprocessing step provided as yaml file (if not provided, default values will be used)
* `labels` (optional) : the label (if any) of each data text file provided as yaml file (line-wise) (e.g. an entry looks like this: `file1: LABEL1`) 

### `/embedding`
embed the extracted phrases into a vector space  

`curl -X GET http://SOME_IP:SOME_PORT/embedding`  
or  
`curl -X POST -F config=@"PATH/TO/CONFIG.yaml" http://SOME_IP:SOME_PORT/embedding`  

* `config` (optional) : configurations for the embedding step provided as yaml file (if not provided, default values will be used)

The first time this endpoint is called, the respective model (as given in config or the default one) will be downloaded

### `/clustering`

## Example Config Files (YAML)
### `/preprocessing`
```
# Name of the corpus; can be chosen freely (but acts as reference point between the different endpoint actions)
corpus_name: default
# Name of the spaCy model; pre-installed: de_dep_news_trf, de_core_news_sm (German) & en_core_web_trf (English)
spacy_model: de_dep_news_trf
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
```

### `/embedding`
```
# Name of the corpus; can be chosen freely (but acts as reference point between the different endpoint actions)
corpus_name: default
# (for German: Sahajtomar/German-semantic)
model_name: sentence-transformers/paraphrase-albert-small-v2
# Number of processes that will be spawned (how many cores will be utilized)
n_process: 1
# Right now, only 'umap' is supported or 'None' (the latter, however, is not advised) 
down_scale_algorithm: umap
# With the prefix 'scaling_' you can tune the various parameters for the 'down_scale_algorithm' if desired
scaling_*
```
