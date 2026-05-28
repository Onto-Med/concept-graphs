# `main_methods.py` Split Status

## Completed

`main_methods.py` has been split into focused service modules and removed from the project root.

## New modules

```text
src/api/services/pipeline_params.py
src/api/services/document_server.py
src/api/services/configuration.py
src/api/services/artifact_responses.py
src/api/services/process_management.py
src/api/services/rag_vectorstore.py
```

## Function mapping

### `pipeline_params.py`

```python
get_pipeline_query_params
get_dict_expression
get_query_param_help_text
get_omit_pipeline_steps
```

### `document_server.py`

```python
get_data_server_config
check_data_server
check_es_source_for_id
is_skip_doc
get_documents_from_es_server
```

### `configuration.py`

```python
read_config
load_configs
read_exclusion_ids
```

### `artifact_responses.py`

```python
data_get_statistics
embedding_get_statistics
clustering_get_concepts
graph_get_statistics
build_adjacency_obj
graph_get_specific
graph_create
```

### `process_management.py`

```python
delete_pipeline
```

### `rag_vectorstore.py`

```python
initialize_chunk_vectorstore
fill_chunk_vectorstore
```

## Import updates

Updated imports in:

```text
src/api/pipeline.py
src/api/routes/artifacts.py
src/api/routes/graph_documents.py
src/api/routes/pipeline.py
src/api/routes/processes.py
src/api/routes/status.py
src/api/routes/rag.py
```

No Python files import `main_methods` anymore.

## Root directory result

The root now contains only:

```text
main.py
main_utils.py
```

## Validation

Ran Black successfully:

```bash
uv run --group test black src/api/services src/api/routes src/api/pipeline.py main.py main_utils.py
```

Ran compile check successfully:

```bash
uv run python -m compileall -q main.py main_utils.py src test
```

Ran Flask smoke checks successfully:

```text
GET /                                                 -> 200
GET /openapi                                          -> 200
GET /status/rag                                       -> 404 when RAG is not initialized
GET /processes                                        -> 404 when no processes exist
GET /pipeline/configuration?default=true&language=en  -> 200
GET /rag/question?q=test                              -> 404 when RAG is not initialized
```

## Full pytest status

`uv run pytest -q` still fails for the same pre-existing unrelated test issues:

- missing `ig` in pruning tests
- incomplete `TestGraphCreator` fixture setup
- missing pickle fixtures under `tmp/`

These failures do not appear related to the `main_methods.py` split.
