# concept-graphs

## Endpoints
### `/preprocessing`
upload text data to be preprocessed (i.e. extraction of phrases)  

`curl -X POST -F data=@"PATH/TO/DATA.zip" -F config=@"PATH/TO/CONFIG.yaml" -F labels=@"PATH/TO/LABELS.yaml" http://SOME_IP:SOME_PORT/preprocessing`  

* `data`  (mandatory) : the text files provided as one zip file
* `config` (optional) : configurations for the preprocessing step provided as yaml file (if not provided, default values will be used)
* `labels` (optional) : the label (if any) of each data text file provided as yaml file (line-wise) (e.g. an entry looks like this: `file1: LABEL1`) 

### `/embedding`
embed the extracted phrases into a vector space  

`curl -X POST -F config=@"PATH/TO/CONFIG.yaml" http://SOME_IP:SOME_PORT/preprocessing`  

* `config` (optional) : configurations for the embedding step provided as yaml file (if not provided, default values will be used)

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
# 
model_name: sentence-transformers/paraphrase-albert-small-v2
# Number of processes that will be spawned (how many cores will be utilized)
n_process: 1
# Right now, only 'umap' is supported or 'None' (the latter, however, is not advised) 
down_scale_algorithm: umap
# With the prefix 'scaling_' you can tune the various parameters for the 'down_scale_algorithm' if desired
scaling_*
```
