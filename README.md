[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-46a2f1.svg)](https://docs.astral.sh/ruff/)

# Concept Graphs

Concept Graphs is a Flask-based API for building, storing, inspecting, and extending concept graphs from document corpora.

The application processes text documents through a pipeline:

1. **Preprocessing**: extracts noun chunks / phrases from documents.
2. **Embedding**: embeds extracted phrases into a vector space.
3. **Clustering**: groups related phrases into concept clusters.
4. **Graph creation**: creates one graph per concept cluster.
5. **Optional integration**: stores phrase/document information in an external vector store.
6. **Optional RAG**: initializes a retrieval-augmented generation component over processed documents.

The API also exposes endpoints to inspect process status, retrieve graph data, add documents to existing graphs, and ask questions via RAG.

> The implementation is based on the Concept Graphs approach described in the references below [1].

## Requirements

The project uses **Python 3.11** and `uv`.

Install dependencies with:
```bash
uv sync
```
## Running locally

Start the API directly:
```bash
uv run python main.py
```
By default, the Flask application listens on:
```text
http://localhost:9010
```
The OpenAPI UI is available at:
```text
http://localhost:9010/
http://localhost:9010/openapi
```
## Docker

Build and start the services:
```bash
docker compose build
docker compose up -d
```
Depending on the compose configuration, the API is usually exposed at:
```text
http://localhost:9007
```
Generated results are written to the configured storage directory inside the container, typically mounted from the `results` Docker volume.

## Data model and processes

Most API operations are associated with a **process** name. A process represents one corpus and its generated pipeline artifacts.

If no process is supplied, the API uses:
```text
default
```
Use the `process` query parameter to select a process:
```bash
curl "http://localhost:9010/status?process=my_corpus"
```
Process names are normalized by the server.

## OpenAPI specification

The OpenAPI specification is served by the application UI and is defined in:
```text
api/concept-graphs-api.yml
```
Configured server URLs include:
```text
http://top-prod:9007
http://localhost:9007
http://localhost:9010
```
## Pipeline

### `POST /pipeline`

Starts a complete concept-graph pipeline.

The endpoint accepts either:

- `multipart/form-data`
- `application/json`

A pipeline can read documents from:

- an uploaded zip file, or
- an external document server.

If a vector store is configured and reachable, the API can additionally run an integration step. If the vector store is not reachable, the API falls back to pickle-based storage where possible.

### Query parameters

| Name | Type | Default | Description |
|---|---:|---:|---|
| `process` | string | `default` | Name of the corpus/process. |
| `language` | string | `en` | Language of the documents. Common values are `en` and `de`. |
| `skip_present` | boolean | `true` | Skip already completed serialized steps. |
| `skip_steps` | string | | Comma-separated list of steps to skip. Supported values include `data`, `embedding`, `clustering`, `graph`, and `integration`. |
| `return_statistics` | boolean | `false` | If `true`, waits for the pipeline to finish and returns graph statistics. This may take a long time. |

### Multipart request
```bash
curl -X POST "http://localhost:9010/pipeline?process=my_corpus&language=en&skip_present=true" \
  -F data=@"./documents.zip" \
  -F data_config=@"./data-config.yaml" \
  -F embedding_config=@"./embedding-config.yaml" \
  -F clustering_config=@"./clustering-config.yaml" \
  -F graph_config=@"./graph-config.yaml"
```
Supported multipart fields:

| Field | Required | Description |
|---|---:|---|
| `data` | conditionally | Zip file containing input text documents. Required unless `document_server_config` is provided. |
| `document_server_config` | conditionally | YAML config for loading documents from an external document server. Required unless `data` is provided. |
| `vectorstore_server_config` | no | YAML config for an external vector store. |
| `labels` | no | YAML file mapping document names/ids to labels. |
| `data_config` | no | Preprocessing configuration. |
| `embedding_config` | no | Embedding configuration. |
| `clustering_config` | no | Clustering configuration. |
| `graph_config` | no | Graph creation configuration. |

### JSON request
```bash
curl -X POST "http://localhost:9010/pipeline?process=my_corpus&language=en" \
  -H "Content-Type: application/json" \
  -d @pipeline-config.json
```
A JSON pipeline configuration may contain:
```json
{
  "name": "my_corpus",
  "language": "en",
  "document_server": {
    "url": "http://localhost",
    "port": 9008,
    "index": "documents",
    "size": 30,
    "other_id": "id",
    "label_key": "label",
    "replace_keys": {
      "text": "content"
    }
  },
  "vectorstore_server": {
    "url": "http://localhost",
    "port": 8882
  },
  "config": {
    "data": {},
    "embedding": {},
    "clustering": {},
    "graph": {}
  }
}
```
### Response

If `return_statistics=false`, the endpoint starts the pipeline asynchronously and returns `202 Accepted` with the current process status.

If `return_statistics=true`, the endpoint waits for the pipeline thread to finish and returns graph statistics.

## Pipeline configuration

### `GET /pipeline/configuration`

Returns either a default pipeline configuration or a stored configuration for a process.
```bash
curl "http://localhost:9010/pipeline/configuration?default=true&language=en"
```
Query parameters:

| Name | Type | Default | Description |
|---|---:|---:|---|
| `default` | boolean | `true` | If `true`, returns the default configuration for the selected language. |
| `process` | string | `default` | Process name, used when `default=false`. |
| `language` | string | `en` | Language for the default configuration. |

Examples:
```bash
curl "http://localhost:9010/pipeline/configuration?default=true&language=en"
curl "http://localhost:9010/pipeline/configuration?default=false&process=my_corpus"
```
## Preprocessing inspection

The standalone preprocessing creation endpoint is no longer exposed. Preprocessing is run through `/pipeline`.

The following endpoints inspect stored preprocessing results.

### `GET /preprocessing/statistics`

Returns basic statistics for a processed corpus.
```bash
curl "http://localhost:9010/preprocessing/statistics?process=my_corpus"
```
### `GET /preprocessing/noun_chunks`

Returns extracted noun chunks / phrase chunks.
```bash
curl "http://localhost:9010/preprocessing/noun_chunks?process=my_corpus"
```
## Embedding inspection

Embedding is run through `/pipeline`.

### `GET /embedding/statistics`

Returns statistics for the stored embedding object.
```bash
curl "http://localhost:9010/embedding/statistics?process=my_corpus"
```
## Clustering inspection

Clustering is run through `/pipeline`.

### `GET /clustering/concepts`

Returns the concepts found during clustering.
```bash
curl "http://localhost:9010/clustering/concepts?process=my_corpus&top_k=15&distance=0.6"
```
Query parameters:

| Name | Type | Default | Description |
|---|---:|---:|---|
| `process` | string | `default` | Process name. |
| `top_k` | integer | `15` | Number of representative phrases to return for each concept. |
| `distance` | number | `0.6` | Cosine distance threshold for representatives. |

## Graphs

Graph creation is run through `/pipeline`.

### `GET /graph/statistics`

Returns basic graph statistics for a process.
```bash
curl "http://localhost:9010/graph/statistics?process=my_corpus"
```
### `GET /graph/{graph_id}`

Returns nodes and adjacency information for a specific graph.
```bash
curl "http://localhost:9010/graph/0?process=my_corpus"
```
To request a rendered graph where supported:
```bash
curl "http://localhost:9010/graph/0?process=my_corpus&draw=true"
```
Query parameters:

| Name | Type | Default | Description |
|---|---:|---:|---|
| `process` | string | `default` | Process name. |
| `draw` | boolean | `false` | If `true`, returns a rendered graph instead of JSON where supported. |

## Adding documents to existing graphs

### `POST /graph/document/add`

Adds one or more documents to an existing process and integrates their phrases into the graphs built for that corpus.

The request body must be JSON.
```bash
curl -X POST "http://localhost:9010/graph/document/add?process=my_corpus" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "en",
    "documents": [
      {
        "id": "doc-001",
        "name": "example.txt",
        "content": "The document text goes here.",
        "label": "optional-label"
      }
    ],
    "vectorstore_server": {
      "url": "http://localhost",
      "port": 8882
    }
  }'
```
Request fields:

| Field | Type | Required | Description |
|---|---|---:|---|
| `language` | string | yes | Document language. |
| `documents` | array | yes | Documents to add. |
| `documents[].id` | string | no | External document id. |
| `documents[].name` | string | yes | Document name. |
| `documents[].content` | string | yes | Document text. |
| `documents[].label` | string | no | Optional document label. |
| `vectorstore_server` | object | no | Vector store connection settings. |
| `document_server` | object | no | Reserved for document-server based additions. |

The endpoint starts an asynchronous document-addition thread.

### `GET /graph/document/add/status`

Returns the status or result of a document-addition task.
```bash
curl "http://localhost:9010/graph/document/add/status?process=my_corpus"
```
Possible responses include:

- `200 OK`: task finished and returned a result
- `202 Accepted`: task is still running
- `404 Not Found`: no document-addition task exists for the process

### `DELETE /graph/document/<DOCUMENT_ID>`

The path exists internally but document deletion is not implemented.

## Process management

### `GET /processes`

Returns all known stored processes.
```bash
curl "http://localhost:9010/processes"
```
### `GET /status`

Returns the status of a specific process.
```bash
curl "http://localhost:9010/status?process=my_corpus"
```
### `GET /processes/{process}/stop`

Requests that a running process be stopped.
```bash
curl "http://localhost:9010/processes/my_corpus/stop"
```
Optional query parameter:

| Name | Type | Default | Description |
|---|---:|---:|---|
| `hard_stop` | boolean | `false` | If `false`, attempts a graceful stop. |

### `DELETE /processes/{process}/delete`

Deletes a process from the in-memory cache and removes serialized artifacts for finished steps.
```bash
curl -X DELETE "http://localhost:9010/processes/my_corpus/delete"
```
Optional query parameter:

| Name | Type | Default | Description |
|---|---:|---:|---|
| `hard_stop` | boolean | `false` | If the process is running, stop it before deletion. |

## Document server status

### `POST /status/document-server`

Checks whether a configured document server is reachable and contains data.

JSON example:
```bash
curl -X POST "http://localhost:9010/status/document-server" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://localhost",
    "port": 9008,
    "index": "documents",
    "size": 30
  }'
```
Multipart example:
```bash
curl -X POST "http://localhost:9010/status/document-server" \
  -F document_server_config=@"./document-server-config.yaml"
```
A typical document server configuration contains:
```yaml
url: "http://localhost"
port: 9008
index: "documents"
size: 30
other_id: "id"
label_key: "label"
replace_keys:
  text: content
```
## RAG

The API can initialize one active RAG component for a process and answer questions over retrieved document chunks.

### `POST /rag/init`

Initializes the RAG component.
```bash
curl -X POST "http://localhost:9010/rag/init?process=my_corpus&force=false" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "",
    "language": "en",
    "vectorstore_server": {
      "url": "http://localhost",
      "port": 8882
    },
    "chatter": {
      "chatter": "src.rag.chatters.blablador.BlabladorChatter"
    },
    "prompt_template": {
      "templates": {
        "en": "Answer the question using the context: {context}\nQuestion: {question}"
      },
      "input_variables": ["context", "question"]
    }
  }'
```
Query parameters:

| Name | Type | Default | Description |
|---|---:|---:|---|
| `process` | string | `default` | Process name. |
| `force` | boolean | `false` | Reinitialize/refill the vector-store index even if it already exists. |

If the backing chunk vector store is empty, initialization starts a background task to fill it. The RAG component becomes ready after initialization completes.

### `GET /status/rag`

Checks whether the active RAG component is ready for a process.
```bash
curl "http://localhost:9010/status/rag?process=my_corpus"
```
### `GET /rag/question`

Asks a question using the active RAG component.
```bash
curl "http://localhost:9010/rag/question?process=my_corpus&q=What%20is%20this%20corpus%20about%3F"
```
### `POST /rag/question`

Asks a question and optionally restricts retrieval to selected document ids.
```bash
curl -X POST "http://localhost:9010/rag/question?process=my_corpus&q=What%20does%20document%20A%20say%3F" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_ids": ["doc-001", "doc-002"],
    "limit": 15
  }'
```
Request body fields:

| Field | Type | Default | Description |
|---|---|---:|---|
| `doc_ids` | array of strings | `[]` | Optional list of document ids to restrict retrieval. |
| `limit` | integer | `15` | Maximum number of document chunks to retrieve. |

A successful response contains:
```json
{
  "answer": "...",
  "info": "..."
}
```
`info` contains serialized metadata for the retrieved reference documents.

## Example configuration files

The recommended way to retrieve a complete current configuration is:
```bash
curl "http://localhost:9010/pipeline/configuration?default=true&language=en"
```
The sections below show the main configuration concepts.

### Pipeline JSON configuration
```json
{
  "name": "default",
  "language": "en",
  "document_server": {
    "url": "http://localhost",
    "port": 9008,
    "index": "documents",
    "size": 30,
    "other_id": "id",
    "label_key": "label",
    "replace_keys": {
      "text": "content"
    }
  },
  "vectorstore_server": {
    "url": "http://localhost",
    "port": 8882
  },
  "config": {
    "data": {
      "spacy_model": "en_core_web_trf",
      "n_process": 1,
      "file_extension": "txt",
      "file_encoding": "utf-8",
      "use_lemma": false,
      "prepend_head": false,
      "head_only": false,
      "case_sensitive": false,
      "disable": null,
      "tfidf_filter": {
        "enabled": false,
        "min_df": 1,
        "max_df": 1,
        "stop": null
      },
      "negspacy": {
        "enabled": true,
        "configuration": {
          "scope": 1,
          "language": "en",
          "feat_of_interest": "NC"
        }
      }
    },
    "embedding": {
      "model": "sentence-transformers/paraphrase-albert-small-v2",
      "n_process": 1,
      "storage": {
        "method": "vectorstore",
        "config": {
          "normalizeEmbeddings": false,
          "annParameters": {
            "spaceType": "dotproduct",
            "parameters": {
              "efConstruction": 1024,
              "m": 16
            }
          }
        }
      }
    },
    "clustering": {
      "algorithm": "kmeans",
      "downscale": "umap",
      "missing_as_recommended": true,
      "deduction": {
        "enabled": true,
        "k_min": 2,
        "k_max": 100,
        "n_samples": 15,
        "sample_fraction": 25,
        "regression_poly_degree": 5
      },
      "scaling": {
        "n_neighbors": 10,
        "n_components": 100,
        "min_dist": 0.1
      },
      "clustering": {}
    },
    "graph": {
      "cluster": {
        "distance": 0.7,
        "min_size": 4
      },
      "graph": {
        "cosine_weight": 0.6,
        "merge_threshold": 0.9,
        "graph_weight_cut_off": 0.6,
        "unroll": false,
        "simplify": 0.5,
        "simplify_alg": "significance",
        "sub_clustering": false
      },
      "restrict_to_cluster": true
    }
  }
}
```
### Document server YAML configuration
```yaml
url: "http://localhost"
port: 9008
index: "documents"
size: 30
other_id: "id"
label_key: "label"
replace_keys:
  text: content
```
### Vector store YAML configuration
```yaml
url: "http://localhost"
port: 8882
```
## Stored artifacts

Pipeline artifacts are stored below the configured file storage directory, which defaults to:
```text
tmp/
```
Each process has its own subdirectory.

Depending on configuration and reachable external services, embeddings and integration data may be stored either through the configured vector store or serialized locally.

## Common workflow

Start a full pipeline from an uploaded zip file:
```bash
curl -X POST "http://localhost:9010/pipeline?process=my_corpus&language=en&return_statistics=false" \
  -F data=@"./documents.zip"
```
Check progress:
```bash
curl "http://localhost:9010/status?process=my_corpus"
```
Inspect concepts:
```bash
curl "http://localhost:9010/clustering/concepts?process=my_corpus&top_k=10"
```
Inspect graph statistics:
```bash
curl "http://localhost:9010/graph/statistics?process=my_corpus"
```
Fetch a graph:
```bash
curl "http://localhost:9010/graph/0?process=my_corpus"
```
Initialize RAG:
```bash
curl -X POST "http://localhost:9010/rag/init?process=my_corpus" \
  -H "Content-Type: application/json" \
  -d @rag-config.json
```
Ask a question:
```bash
curl "http://localhost:9010/rag/question?process=my_corpus&q=Summarize%20the%20main%20topics."
```
## Notes and limitations

- Pipeline execution can take a long time for large corpora.
- Some operations run asynchronously. Use `/status`, `/processes`, `/graph/document/add/status`, and `/status/rag` to inspect progress.
- Only one active RAG component is held by the application at a time.
- Document deletion from graphs is not implemented.
- Graph quality depends heavily on corpus size, extracted phrase quality, embeddings, and clustering settings.
- Very small corpora may not produce useful concept clusters or graphs.

## References

**[1]** Matthies, F. et al. *Concept Graphs: A Novel Approach for Textual Analysis of Medical Documents.* In: Röhrig, R. et al., editors. Studies in Health Technology and Informatics. IOS Press; 2023. Available from: https://ebooks.iospress.nl/doi/10.3233/SHTI230710

**[2]** NetworkX: https://networkx.org/

**[3]** Dianati, N. *Unwinding the hairball graph: Pruning algorithms for weighted complex networks.* Physical Review E. 2016;93(1). Available from: https://link.aps.org/doi/10.1103/PhysRevE.93.012304

**[4]** Chapman, Bridewell, Hanbury, Cooper, Buchanan. *NegEx - A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries.* https://doi.org/10.1006/jbin.2001.1029