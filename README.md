[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-46a2f1.svg)](https://docs.astral.sh/ruff/)  
Docker Image: 
``
docker pull ghcr.io/onto-med/concept-graphs/concept-graphs-api:1.1.1
``
# Concept Graphs

Concept Graphs is a Flask API for building, storing, inspecting, extending, and querying concept graphs from document corpora.

A corpus is processed through a pipeline:

1. **Preprocessing**: load documents and extract noun chunks / phrase chunks.
2. **Embedding**: encode extracted phrases into vectors.
3. **Clustering**: group related phrases into concepts.
4. **Graph creation**: build concept graphs from phrase/document relations.
5. **Optional integration**: write graph-cluster metadata to an external vector store.
6. **Optional RAG**: initialize retrieval-augmented generation over processed document chunks.

The API also supports process management, graph inspection, adding documents to existing graphs, deleting added document provenance, asking questions through RAG, and LLM-based query expansion.

The implementation is based on the Concept Graphs approach described in the references.

---

## Requirements

- Python `>=3.11,<3.12`
- [`uv`](https://docs.astral.sh/uv/)
- Optional external services:
  - Marqo/vector store
  - document index server
  - RAG backend/chatter service, depending on configuration

Install dependencies:

```bash
uv sync
```

Run checks:

```bash
uv run --group test ruff format .
uv run --group test ruff check .
uv run --no-sync pytest -q
```

---

## Running locally

Start the Flask development server:

```bash
uv run python main.py
```

Default local URL:

```text
http://localhost:9010
```

Swagger UI / OpenAPI UI:

```text
http://localhost:9010/
http://localhost:9010/openapi
```

The OpenAPI document lives at:

```text
api/concept-graphs-api.yml
```

---

## Application factory and runtime state

The app uses a Flask application factory:

```python
from main import create_app

app = create_app()
```

Production servers should call the factory, for example with Waitress:

```bash
waitress-serve --call --port=9007 main:create_app
```

Shared runtime state is attached to the Flask app under:

```python
app.extensions["concept_graphs_context"]
```

That context contains grouped runtime state for:

- running processes and threads
- active pipeline step objects
- file storage configuration
- per-process active RAG state

---

## Docker

Build and start the default compose setup:

```bash
docker compose build
docker compose up -d
```

The production image runs Waitress with the factory entrypoint:

```text
main:create_app --call
```

The production compose setup should use the code baked into the image. It should not bind-mount the project over `/rest_api`, because doing so can hide the image's build-time `.venv`.

The local development compose file may bind-mount the source tree for iterative work.

Typical API URL in Docker setups:

```text
http://localhost:9007
```

### Providing extra localized prompt profiles in Docker

Prompt profiles are normal YAML files below `conf/`:

```text
conf/rag/localization/<profile>.yml
conf/query-expansion/localization/<profile>.yml
```

When using the production image, do **not** bind-mount the whole project over `/rest_api`. Instead, mount only the additional profile file or profile directory into the matching `conf/` subdirectory.

Example custom RAG profile file on the host:

```text
./local-conf/rag/localization/fr.yml
```

```yaml
input_variables:
  - summaries
  - question

template: |
  Réponds à la question à partir des SOURCES fournies.
  Réponds toujours en français.

  QUESTION:
  {question}

  SOURCES:
  {summaries}

  RÉPONSE FINALE EN FRANÇAIS:
```

Compose example:

```yaml
services:
  concept-graphs-api:
    image: ghcr.io/onto-med/concept-graphs/concept-graphs-api:1.0.0
    volumes:
      - ./local-conf/rag/localization/fr.yml:/rest_api/conf/rag/localization/fr.yml:ro
      - ./local-conf/query-expansion/localization/fr.yml:/rest_api/conf/query-expansion/localization/fr.yml:ro
```

`docker run` example:

```bash
docker run --rm -p 9007:9007 \
  -v "$PWD/local-conf/rag/localization/fr.yml:/rest_api/conf/rag/localization/fr.yml:ro" \
  ghcr.io/onto-med/concept-graphs/concept-graphs-api:1.0.0
```

Use the mounted profile by passing the profile/language in the request, for example:

```json
{
  "language": "fr",
  "prompt_template": {
    "profile": "fr"
  }
}
```

For query expansion, use:

```json
{
  "language": "fr",
  "prompt": {
    "profile": "fr"
  }
}
```

---

## Processes and storage

Most endpoints use a `process` query parameter. A process represents one corpus and its stored artifacts.

If omitted, the process defaults to:

```text
default
```

Example:

```bash
curl "http://localhost:9010/status?process=my_corpus"
```

Pipeline artifacts are stored below the configured file storage directory, defaulting to:

```text
tmp/
```

Each process has its own directory:

```text
tmp/<process>/
```

Typical storage behavior:

| Artifact / data | Stored where |
|---|---|
| Pipeline step pickles | `tmp/<process>/` |
| Graph pickle | `tmp/<process>/<process>_graph.pickle` |
| Phrase embeddings | local pickle and/or Marqo, depending on config |
| Integration metadata | Marqo/vector store when configured |
| Document-addition graph provenance | graph pickle node attributes |
| Document-addition vector provenance | Marqo entry metadata |
| Full documents added through `/graph/document/add` | not currently inserted into the external document server |

Document addition currently updates the graph/vector-store side only:

| Target | Added by document addition? |
|---|---:|
| Concept graph pickle | yes |
| Vector store / Marqo phrase index | yes |
| Existing processed data pickle | no |
| External document index server | no |

---

## Pipeline

### `POST /pipeline`

Starts a full concept-graph pipeline.

The endpoint accepts either:

- `application/json`
- `multipart/form-data`

Documents can come from either:

- an uploaded zip file, or
- an external document server.

### Pipeline query parameters

| Name | Type | Default | Description |
|---|---:|---:|---|
| `process` | string | `default` | Corpus/process name. |
| `language` | string | `en` | Document language, for example `en` or `de`. |
| `skip_present` | boolean | `true` | Reuse already serialized step artifacts where possible. |
| `skip_steps` | string | | Comma-separated steps to skip: `data`, `embedding`, `clustering`, `graph`, `integration`. |
| `return_statistics` | boolean | `false` | If `true`, waits for completion and returns graph statistics. This can take a long time. |

### JSON request

```bash
curl -X POST "http://localhost:9010/pipeline?process=my_corpus&language=en" \
  -H "Content-Type: application/json" \
  -d @conf/pipeline-config_en.json
```

A full JSON config has this general shape:

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

### Multipart request

```bash
curl -X POST "http://localhost:9010/pipeline?process=my_corpus&language=en&skip_present=true" \
  -F data=@"./documents.zip" \
  -F data_config=@"./data-config.yaml" \
  -F embedding_config=@"./embedding-config.yaml" \
  -F clustering_config=@"./clustering-config.yaml" \
  -F graph_config=@"./graph-config.yaml" \
  -F vectorstore_server_config=@"./vectorstore-server.yaml"
```

Supported multipart fields:

| Field | Required | Description |
|---|---:|---|
| `data` | conditionally | Zip file containing input text documents. Required unless a document server config is provided. |
| `document_server_config` | conditionally | Config file for loading documents from an external document server. Required unless uploaded data is provided. |
| `vectorstore_server_config` | no | External vector-store config. |
| `labels` | no | Optional labels mapping file. |
| `data_config` | no | Preprocessing config. |
| `embedding_config` | no | Embedding config. |
| `clustering_config` | no | Clustering config. |
| `graph_config` | no | Graph creation config. |

### Pipeline configuration

Get the default configuration:

```bash
curl "http://localhost:9010/pipeline/configuration?default=true&language=en"
```

Get a stored process configuration:

```bash
curl "http://localhost:9010/pipeline/configuration?default=false&process=my_corpus"
```

Available defaults are stored in `conf/`, for example:

```text
conf/pipeline-config_de.json
```

---

## Inspecting pipeline artifacts

### Preprocessing

```bash
curl "http://localhost:9010/preprocessing/statistics?process=my_corpus"
curl "http://localhost:9010/preprocessing/noun_chunks?process=my_corpus"
```

### Embedding

```bash
curl "http://localhost:9010/embedding/statistics?process=my_corpus"
```

### Clustering

```bash
curl "http://localhost:9010/clustering/concepts?process=my_corpus&top_k=15&distance=0.6"
```

Parameters:

| Name | Type | Default | Description |
|---|---:|---:|---|
| `top_k` | integer | `15` | Number of representative phrases per concept. |
| `distance` | number | `0.6` | Cosine distance threshold. |

### Graphs

```bash
curl "http://localhost:9010/graph/statistics?process=my_corpus"
curl "http://localhost:9010/graph/0?process=my_corpus"
curl "http://localhost:9010/graph/0?process=my_corpus&draw=true"
```

Parameters:

| Name | Type | Default | Description |
|---|---:|---:|---|
| `process` | string | `default` | Process name. |
| `draw` | boolean | `false` | Return a rendered graph where supported. |

---

## Adding documents to existing graphs

### `POST /graph/document/add`

Adds one or more documents to an existing process and integrates their phrases into already built graphs.

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
| `documents[].id` | string | no | External document ID. If missing, a UUID is generated. |
| `documents[].name` | string | yes | Document name. |
| `documents[].content` | string | yes | Document text. |
| `documents[].label` | string | no | Optional document label. |
| `vectorstore_server` | object | no | Vector-store connection settings. |
| `document_server` | object | no | Currently not used for inserting added documents. |

The endpoint starts an asynchronous document-addition thread.

Check the result:

```bash
curl "http://localhost:9010/graph/document/add/status?process=my_corpus"
```

Document-addition provenance is stored as:

- graph node `documents` attributes in the graph pickle
- `documents` metadata on Marqo/vector-store entries

Example Marqo provenance metadata:

```json
{
  "documents": [
    {
      "id": "doc-001",
      "offsets": [[0, 42]]
    }
  ],
  "source": "document_addition"
}
```

### `DELETE /graph/document/{document_id}`

Removes document provenance from graphs and, where vector-store provenance is available, from Marqo entries.

```bash
curl -X DELETE "http://localhost:9010/graph/document/doc-001?process=my_corpus"
```

Optional query parameters:

| Name | Type | Default | Description |
|---|---:|---:|---|
| `remove_unreferenced_nodes` | boolean | `true` | Remove graph nodes that have no document references after deletion. |
| `delete_unreferenced_embeddings` | boolean | `false` | Delete vector-store entries that have no document provenance after deletion. |

Optional JSON body if vector-store settings are not available from the saved embedding object:

```json
{
  "vectorstore_server": {
    "client_url": "http://localhost:8882",
    "index_name": "my_corpus"
  }
}
```

---

## Process management

List known/running processes:

```bash
curl "http://localhost:9010/processes"
```

Get one process status:

```bash
curl "http://localhost:9010/status?process=my_corpus"
```

Stop a process:

```bash
curl "http://localhost:9010/processes/my_corpus/stop?hard_stop=false"
```

Delete a process and serialized artifacts:

```bash
curl -X DELETE "http://localhost:9010/processes/my_corpus/delete?hard_stop=false"
```

---

## Document server status

Check whether a configured document server is reachable and contains data.

JSON request:

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

Multipart request:

```bash
curl -X POST "http://localhost:9010/status/document-server" \
  -F document_server_config=@"./document-server-config.yaml"
```

Typical document-server config:

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

---

## RAG

The API can initialize one active RAG component per process/corpus. Each process uses its own RAG vector-store index named `<process>_rag`.

### `POST /rag/init`

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
      "profile": "en"
    }
  }'
```

Provider-specific OpenAI-compatible request options can be forwarded through the chatter config with `extra_body`, for example to disable model-specific thinking/reasoning when the provider supports it:

```json
{
  "chatter": {
    "model": "alias-fast",
    "temperature": 0.0,
    "extra_body": {
      "chat_template_kwargs": {
        "enable_thinking": false
      }
    }
  }
}
```

The exact `extra_body` shape is provider/model-specific.

RAG prompt profiles are file-based by default:

```text
conf/rag/localization/
  en.yml
  de.yml
```

Each profile contains a `template` and `input_variables`. The built-in variables are `{question}` and `{summaries}`. The old inline request-body shape with `templates` and `input_variables` is still supported for experiments/backwards compatibility; `prompt_template.template` can also override the resolved file profile directly.

Query parameters:

| Name | Type | Default | Description |
|---|---:|---:|---|
| `process` | string | `default` | Process name. |
| `force` | boolean | `false` | Reinitialize/refill the vector-store index even if it already exists. |

If the backing chunk vector store is empty, initialization starts a background task. Check readiness with:

```bash
curl "http://localhost:9010/status/rag?process=my_corpus"
```

### `GET /rag/question`

```bash
curl "http://localhost:9010/rag/question?process=my_corpus&q=What%20is%20this%20corpus%20about%3F"
```

### `POST /rag/question`

Use POST to restrict retrieval to selected document IDs and/or set a chunk limit:

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
| `doc_ids` | array of strings | `[]` | Restrict retrieval to selected document IDs. |
| `limit` | integer | `15` | Maximum number of chunks to retrieve. |

Successful response:

```json
{
  "answer": "...",
  "info": "..."
}
```

---

## Query expansion

The API can generate categorized LLM query-expansion candidates and optionally ground them against terminology/source files.

Endpoint:

```text
POST /query-expansion
```

The default implementation uses LangChain and validates the LLM output with Pydantic models. A PydanticAI generator remains available for future/custom use, but is not the default runtime path.

Supported built-in categories are stable API identifiers:

```text
synonym
medication
diagnosis
symptom
procedure
abbreviation
broader_term
narrower_term
related_term
```

Prompt profiles live in:

```text
conf/query-expansion/localization/
  en.yml
  de.yml
```

The profile is selected from `prompt.profile` or, if omitted, from `language`. Prompt templates and category descriptions can also be overridden per request.

### Blablador example

Prefer passing provider tokens through a header instead of the JSON body:

```bash
curl -X POST "http://localhost:9010/query-expansion" \
  -H "Content-Type: application/json" \
  -H "X-LLM-API-Key: YOUR_BLABLADOR_API_KEY" \
  -d '{
    "term": "Myokardinfarkt",
    "language": "de",
    "categories": ["synonym", "medication", "symptom"],
    "limit_per_category": 5,
    "llm": {
      "model": "alias-fast",
      "options": {
        "provider": "blablador",
        "base_url": "https://api.helmholtz-blablador.fz-juelich.de/v1/",
        "temperature": 0.0
      }
    },
    "grounding": {
      "include_llm_only": true
    }
  }'
```

Accepted API-key headers include:

```text
Authorization: Bearer <token>
X-LLM-API-Key: <token>
X-API-Key: <token>
X-Blablador-API-Key: <token>
```

### Prompt override example

```json
{
  "term": "Myokardinfarkt",
  "language": "de",
  "categories": ["synonym"],
  "llm": {
    "model": "alias-fast",
    "options": {
      "provider": "blablador"
    }
  },
  "prompt": {
    "profile": "de",
    "category_descriptions": {
      "synonym": "Synonyme, laienverständliche Begriffe und häufige Schreibvarianten."
    }
  }
}
```

Grounding sources are optional. Currently implemented source type:

```json
{
  "name": "local-medical-terms",
  "type": "local",
  "path": "conf/query-expansion/grounding/medical_terms.example.yml"
}
```

Local grounding loads YAML or JSON and exact-matches generated candidates against each entry's `term` and `synonyms` after lowercasing and whitespace normalization. Optional `category` / `categories` metadata restricts grounding to matching stable category IDs.

Example YAML:

```yaml
terms:
  - id: C001
    term: Myokardinfarkt
    synonyms:
      - Herzinfarkt
      - MI
    category: diagnosis

  - id: C002
    term: Aspirin
    synonyms:
      - ASS
      - Acetylsalicylsäure
    category: medication

  - id: C003
    term: Brustschmerz
    synonyms:
      - Thoraxschmerz
    categories:
      - symptom
      - related_term
```

If `category` / `categories` is omitted, the entry can ground candidates from any category. Ungrounded candidates are returned with status `llm_only` unless disabled via grounding options.

---

## Fixture generation for tests

A helper script can run the pipeline on a folder of documents and write fixtures:

```bash
uv run --no-sync python test/data/scripts/run_pipeline_on_folder.py \
  test/data/documents/grascco/ \
  --process grascco \
  --language de \
  --file-storage-dir test/data/results \
  --pipeline-config conf/pipeline-config_de.json \
  --skip-steps integration
```

Useful flags:

| Flag | Description |
|---|---|
| `--pipeline-config` | Full JSON pipeline config. |
| `--skip-present` | Reuse existing artifacts. This is the default. |
| `--no-skip-present` | Recompute artifacts. |
| `--skip-steps integration` | Skip external vector-store integration. |

Current tests use fixtures under:

```text
test/data/results/grascco
```

---

## API overview

Business endpoints are documented in Swagger UI and `api/concept-graphs-api.yml`.

Main endpoint groups:

| Group | Endpoints |
|---|---|
| Pipeline | `POST /pipeline`, `GET /pipeline/configuration` |
| Artifacts | `/preprocessing/*`, `/embedding/*`, `/clustering/*`, `/graph/*` |
| Graph documents | `POST /graph/document/add`, `GET /graph/document/add/status`, `DELETE /graph/document/{document_id}` |
| Processes | `GET /processes`, `GET /status`, `GET /processes/{process}/stop`, `DELETE /processes/{process}/delete` |
| Status | `POST /status/document-server`, `GET /status/rag` |
| RAG | `POST /rag/init`, `GET/POST /rag/question` |
| Query expansion | `POST /query-expansion` |

Documentation/static routes such as `/`, `/openapi`, and static files are intentionally not considered business API endpoints.

---

## Development notes

Code style and linting use Ruff:

```bash
uv run --group test ruff format .
uv run --group test ruff check .
```

Current Ruff rule groups include:

- `E`: pycodestyle errors
- `F`: pyflakes
- `I`: import sorting
- `UP`: pyupgrade modernization

Run the full validation set:

```bash
uv run --group test ruff format .
uv run --group test ruff check .
uv run --no-sync python -m compileall -q main.py src test
uv run --no-sync pytest -q
```

---

## Known limitations

- Pipeline execution can take a long time for large corpora.
- Some operations are asynchronous. Use `/status`, `/processes`, `/graph/document/add/status`, and `/status/rag` to inspect progress.
- RAG state is held in memory per process; it must be reinitialized after an application restart.
- Document addition does not currently insert full documents into the external document server.
- Graph quality depends heavily on corpus size, extracted phrase quality, embedding model, and clustering settings.
- Very small corpora may not produce useful concept clusters or graphs.

---

## References

**[1]** Matthies, F. et al. *Concept Graphs: A Novel Approach for Textual Analysis of Medical Documents.* In: Röhrig, R. et al., editors. Studies in Health Technology and Informatics. IOS Press; 2023. Available from: https://ebooks.iospress.nl/doi/10.3233/SHTI230710

**[2]** NetworkX: https://networkx.org/

**[3]** Dianati, N. *Unwinding the hairball graph: Pruning algorithms for weighted complex networks.* Physical Review E. 2016;93(1). Available from: https://link.aps.org/doi/10.1103/PhysRevE.93.012304

**[4]** Chapman, Bridewell, Hanbury, Cooper, Buchanan. *NegEx - A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries.* https://doi.org/10.1006/jbin.2001.1029
